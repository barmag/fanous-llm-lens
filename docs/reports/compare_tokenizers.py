"""Reproduce the MSA-vs-Masri tokenizer-fitness diagnostics for the report.

Trains ``bpe``, ``unigram``, ``wordpiece``, ``morfessor`` and ``morphological`` on a large
mixed corpus, then runs the **honest, gold-light** diagnostics from
``fanous_lens.tokenizers.evaluate`` on a **disjoint** held-out eval set, stratified by
register. Prints the tables that back ``docs/reports/tokenizer-comparison.md``.

Why no precision / F1 (2026-06-29 design rethink): the morpheme gold marks *clitics only*,
not inflection, and is weak on Masri. Against an incomplete gold you cannot measure
precision — a boundary placed where the gold is silent is indistinguishable from a true
boundary the gold missed — so an F1 built on it rewards agreement with the gold's blind
spots, and a ``morphological`` tokenizer (whose vocab is the gold) scores 1.0 by tautology.
We report instead:

- **fertility** = tokens / whitespace-word, at equal realized vocab (parity reported).
- **unk %** = fraction of ids that are ``[UNK]`` (information lost to OOV).
- **clitic recall** = of the clitic boundaries the gold is *confident* about, the fraction
  the tokenizer placed (±1 char, greedy one-to-one). Honest only **read with fertility**:
  recall is gamed by over-segmentation. ``beyond %`` = share of a tokenizer's intra-word
  seams that fall where the gold is silent — descriptive (may be correct inflection/dialect
  splitting), never scored as error.
- **coverage** = 1 − gold-skipped/words (Masri's is lower, so its recall saw less signal).
- **consistency** (gold-free): does a fixed morpheme tokenize the same across host words?
  ``top-share`` 1.0 / ``entropy`` 0.0 = perfectly stable → localizable feature.

These are diagnostics, not a verdict: whether morpheme alignment *buys* interpretability is
a hypothesis only the Phase A probe settles. Run: ``uv run python docs/reports/compare_tokenizers.py``
"""

from __future__ import annotations

import itertools
import json
import sys

from fanous_lens.tokenizers.corpora import load_corpora
from fanous_lens.tokenizers.evaluate import clitic_recall, gold_for, morpheme_consistency
from fanous_lens.tokenizers.morphological import morpheme_boundaries
from fanous_lens.tokenizers.train import get_tokenizer, train_tokenizer

APPROACHES = ["bpe", "unigram", "wordpiece", "morfessor", "morphological"]
N_TRAIN = 3_000  # per register; 6_000 total — enough to saturate vocab_size
N_EVAL = 200  # per register; disjoint from train
VOCAB_SIZE = 8_000

# Representative single words for the report appendix.
EXAMPLE_WORDS = [
    ("MSA", "وسيذهبون", "wa-sa-yaḏhabūn · and they will go"),
    ("MSA", "بالقلم", "bi-l-qalam · with the pen"),
    ("MSA", "كتبها", "katab-hā · he wrote it"),
    ("MSA", "المدرسة", "al-madrasa · the school"),
    ("Masri", "كتابه", "kitāb-u · his book (shared enclitic)"),
    ("Masri", "بالعربية", "bi-l-ʿarabiyya · by car (shared proclitics)"),
    ("Masri", "بيكتب", "bi-yiktib · he is writing (progressive بـ)"),
    ("Masri", "هيروح", "ha-yrūḥ · he will go (future هـ)"),
    ("Masri", "بتاعنا", "bitāʿ-na · ours (analytic possessive)"),
]

# Shared morphemes + real host words for the gold-free consistency check.
CONSISTENCY_ITEMS = [
    ("ال", ["الكتاب", "المدرسة", "الولد", "الطاولة", "البيت", "الجامعة"]),
    ("و", ["وكتب", "والكتاب", "وبيت", "وعايزين", "وسيذهبون"]),
    ("ب", ["بالقلم", "بالعربية", "بالكتاب", "بالمدرسة"]),
    ("كتاب", ["الكتاب", "كتابه", "كتابها", "وكتاب", "كتاب"]),
    ("بيت", ["البيت", "بيته", "بيتهم", "وبيت", "بيت"]),
]


def realized_vocab(approach: str, config: dict) -> int:
    if approach in ("bpe", "unigram", "wordpiece"):
        return len(config["model"]["vocab"])
    return len(config["vocab"])


def unk_id_of(approach: str, config: dict) -> int:
    # Unigram serializes vocab as a list of [token, logprob]; BPE/WordPiece as a dict.
    if approach == "unigram":
        return next(i for i, (t, _s) in enumerate(config["model"]["vocab"]) if t == "[UNK]")
    if approach in ("bpe", "wordpiece"):
        return config["model"]["vocab"]["[UNK]"]
    return config["vocab"]["[UNK]"]


def unk_rate(encode, sentences: list[str], unk_id: int) -> float:
    n_tok = n_unk = 0
    for text in sentences:
        ids, _offsets = encode(text)
        n_tok += len(ids)
        n_unk += sum(1 for i in ids if i == unk_id)
    return round(n_unk / n_tok, 4) if n_tok else 0.0


def segmentation(encode, word: str) -> str:
    _ids, offsets = encode(word)
    return "·".join(word[s:e] for s, e in offsets)


def gold_segmentation(word: str) -> str:
    cuts = [0, *morpheme_boundaries(word), len(word)]
    return "·".join(word[a:b] for a, b in itertools.pairwise(cuts))


def print_examples(encoders: dict) -> None:
    print("\n=== APPENDIX: representative segmentations ===")
    for reg, word, gloss in EXAMPLE_WORDS:
        print(f"\n{reg}: {word}  — {gloss}")
        print(f"  {'gold':13}: {gold_segmentation(word)}")
        for a in APPROACHES:
            print(f"  {a:13}: {segmentation(encoders[a], word)}")


def main() -> None:
    msa_all, masri_all = load_corpora(max_msa=N_TRAIN + N_EVAL, max_masri=N_TRAIN + N_EVAL)
    msa_eval = msa_all[N_TRAIN : N_TRAIN + N_EVAL]
    masri_eval = masri_all[N_TRAIN : N_TRAIN + N_EVAL]
    train_corpus = msa_all[:N_TRAIN] + masri_all[:N_TRAIN]
    print(
        f"train={len(train_corpus)}  eval MSA={len(msa_eval)} Masri={len(masri_eval)}", flush=True
    )

    gold = {"MSA": [gold_for(t) for t in msa_eval], "Masri": [gold_for(t) for t in masri_eval]}
    evals = {"MSA": msa_eval, "Masri": masri_eval}

    rows: dict[str, dict] = {}
    encoders: dict = {}
    for a in APPROACHES:
        print(f"training {a} ...", file=sys.stderr, flush=True)
        cfg = train_tokenizer(a, train_corpus, vocab_size=VOCAB_SIZE)
        enc = get_tokenizer(a, cfg)
        encoders[a] = enc
        uid = unk_id_of(a, cfg)
        row: dict = {"vocab": realized_vocab(a, cfg)}
        for reg in ("MSA", "Masri"):
            r = clitic_recall(enc, evals[reg], gold[reg])
            r["unk_rate"] = unk_rate(enc, evals[reg], uid)
            row[reg] = r
        row["consistency"] = morpheme_consistency(enc, CONSISTENCY_ITEMS)
        rows[a] = row

    hdr = (
        f"{'approach':14}{'vocab':>7}{'reg':>7}{'fert':>7}{'unk%':>7}"
        f"{'recall':>8}{'beyond%':>8}{'gold':>7}{'cov':>7}"
    )
    print("\n" + hdr)
    for a, d in rows.items():
        for reg in ("MSA", "Masri"):
            m = d[reg]
            print(
                f"{a:14}{d['vocab']:>7}{reg:>7}{m['fertility']:>7}{m['unk_rate'] * 100:>7.2f}"
                f"{m['recall']:>8}{m['beyond_gold_rate'] * 100:>8.1f}{m['gold_seams']:>7}{m['coverage']:>7}"
            )

    print(f"\n{'approach':14}{'top-share':>11}{'entropy(bits)':>15}{'n_morph':>9}")
    for a, d in rows.items():
        c = d["consistency"]
        print(f"{a:14}{c['mean_top_share']:>11}{c['mean_entropy']:>15}{c['n_morphemes']:>9}")

    print_examples(encoders)
    print("\nJSON:\n" + json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
