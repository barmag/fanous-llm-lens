"""Reproduce the MSA-vs-Masri comparison of the five candidate tokenizers.

Trains ``bpe``, ``unigram``, ``wordpiece``, ``morfessor`` and ``morphological`` on a
large mixed corpus, then scores each on a **disjoint** held-out eval set, stratified by
register. Prints the table that backs ``docs/reports/tokenizer-comparison.md``.

Metrics (all per register, never averaged across registers):

- **fertility** = tokens / whitespace-word — segmentation granularity (compare only at
  equal realized vocab; we report the realized vocab so the reader can check parity).
- **unk %** = fraction of emitted ids that are ``[UNK]`` — read *together* with fertility,
  since heavy UNK lowers fertility while destroying information.
- **precision / recall / f1** of intra-word morpheme seams vs the camel-tools gold
  (``morpheme_boundaries``), with greedy one-to-one matching under a ±1-char tolerance,
  restricted to the gold-covered words in both the prediction and the gold.
- **coverage** = 1 − (gold-skipped words / words) — how much of the register the MSA gold
  could actually segment (the Masri denominator is smaller; its F1 is a lower bound).

``morphological`` is the **oracle** for the alignment metric — its vocab is built from the
same camel-tools segmentation as the gold, so it scores F1≈1.0 by construction. It is
shown for reference and excluded from the *alignment* ranking, but is a legitimate
comparand on fertility / UNK.

Determinism: the HF trainers and camel-tools are deterministic and morfessor's batch
training is seedless here, so no RNG seed is required; re-running on the same cached
corpus reproduces the table. Run:  ``uv run python docs/reports/compare_tokenizers.py``
"""

from __future__ import annotations

import json
import sys

from camel_tools.tokenizers.word import simple_word_tokenize

from fanous_lens.tokenizers.corpora import load_corpora
from fanous_lens.tokenizers.morphological import morpheme_boundaries_with_coverage
from fanous_lens.tokenizers.train import get_tokenizer, train_tokenizer

APPROACHES = ["bpe", "unigram", "wordpiece", "morfessor", "morphological"]
N_TRAIN = 3_000  # per register; 6_000 total — enough to saturate vocab_size
N_EVAL = 200  # per register; disjoint from train
VOCAB_SIZE = 8_000
TOLERANCE = 1  # ±1 surface char when matching a predicted seam to a gold seam


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


def word_starts(text: str) -> set[int]:
    """Offsets that begin a whitespace-delimited word (never an intra-word seam)."""
    starts: set[int] = set()
    in_word = False
    for i, ch in enumerate(text):
        if ch == " ":
            in_word = False
        else:
            if not in_word:
                starts.add(i)
            in_word = True
    return starts


def gold_for(text: str) -> tuple[list[int], list[tuple[int, int]], int, int]:
    """(gold intra-word seams, covered-word spans, n_words, n_skipped). Computed once."""
    seams, n_words, n_skipped = morpheme_boundaries_with_coverage(text)
    search_from, spans = 0, []
    for w in simple_word_tokenize(text):
        idx = text.find(w, search_from)
        if idx < 0:
            continue
        search_from = idx + len(w)
        _b, _n, sk = morpheme_boundaries_with_coverage(w)
        if sk == 0:
            spans.append((idx, idx + len(w)))
    covered = [s for s in seams if any(a < s < b for a, b in spans)]
    return covered, spans, n_words, n_skipped


def predicted_seams(text: str, offsets: list[tuple[int, int]], spans: list[tuple[int, int]]):
    """Token-start offsets strictly inside a gold-covered word.

    The Whitespace pre-tokenizer guarantees no token crosses a word boundary for any of
    the five tokenizers, so an intra-word seam is exactly a token start that is not a
    word start.
    """
    ws = word_starts(text)
    cand = {s for s, _e in offsets if s not in ws and 0 < s < len(text)}
    return sorted(s for s in cand if any(a < s < b for a, b in spans))


def greedy_match(pred: list[int], gold: list[int], tol: int = TOLERANCE) -> int:
    """Greedy one-to-one matches within ±tol; each pred and gold seam consumed once."""
    gold = sorted(gold)
    used = [False] * len(gold)
    matched = 0
    for p in sorted(pred):
        for j, g in enumerate(gold):
            if not used[j] and abs(p - g) <= tol:
                used[j] = True
                matched += 1
                break
    return matched


def eval_register(encode, sentences, gold, unk_id) -> dict:
    n_tokens = n_words = n_unk = 0
    sum_m = sum_p = sum_g = 0
    n_w_total = n_w_skip = 0
    for text, (gold_seams, spans, nw, sk) in zip(sentences, gold, strict=True):
        ids, offsets = encode(text)
        n_tokens += len(ids)
        n_words += len(text.split())
        n_unk += sum(1 for i in ids if i == unk_id)
        n_w_total += nw
        n_w_skip += sk
        pred = predicted_seams(text, offsets, spans)
        sum_m += greedy_match(pred, gold_seams)
        sum_p += len(pred)
        sum_g += len(gold_seams)
    p = sum_m / sum_p if sum_p else 0.0
    r = sum_m / sum_g if sum_g else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "fertility": round(n_tokens / n_words, 3) if n_words else 0.0,
        "unk_rate": round(n_unk / n_tokens, 4) if n_tokens else 0.0,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "gold_seams": sum_g,
        "coverage": round(1 - n_w_skip / n_w_total, 3) if n_w_total else 0.0,
    }


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
    for a in APPROACHES:
        print(f"training {a} ...", file=sys.stderr, flush=True)
        cfg = train_tokenizer(a, train_corpus, vocab_size=VOCAB_SIZE)
        enc = get_tokenizer(a, cfg)
        uid = unk_id_of(a, cfg)
        rows[a] = {
            "vocab": realized_vocab(a, cfg),
            "MSA": eval_register(enc, evals["MSA"], gold["MSA"], uid),
            "Masri": eval_register(enc, evals["Masri"], gold["Masri"], uid),
        }

    hdr = f"{'approach':14}{'vocab':>7}{'reg':>7}{'fert':>7}{'unk%':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'gold':>7}{'cov':>7}"
    print("\n" + hdr)
    for a, d in rows.items():
        for reg in ("MSA", "Masri"):
            m = d[reg]
            print(
                f"{a:14}{d['vocab']:>7}{reg:>7}{m['fertility']:>7}{m['unk_rate'] * 100:>7.2f}"
                f"{m['precision']:>7}{m['recall']:>7}{m['f1']:>7}{m['gold_seams']:>7}{m['coverage']:>7}"
            )
    print("\nJSON:\n" + json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
