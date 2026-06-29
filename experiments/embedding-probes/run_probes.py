"""Phase A — probe the zero-layer embedding spaces for linguistic features.

The Phase A question (the verdict Phase B deferred to): does a morpheme-aligned tokenization
make a linguistic feature **more linearly recoverable** from a model's embedding space than a
frequency-only subword tokenization?

**Design (departs from the spec's per-token-`W_E` probe, deliberately).** Probing single token
embeddings is degenerate for these features: closed-class markers have a handful of positive
token *types* (no generalization), and *which* token carries a feature differs per tokenizer
(morphological has a clean ``ال`` token; BPE does not), so labels are not comparable. Instead:

- **Unit = a word.** Represent each word as the **mean of its token embeddings** under a
  tokenizer's ``W_E`` — uniform across tokenizers (BPE pools subwords; morphological pools
  morphemes; pooling is the standard cross-granularity comparison).
- **Features = high-cardinality morphology**, labelled by camel-tools (tokenizer-independent):
  - **definiteness** — many positive types; the feature where alignment should matter *most*
    (a tokenizer that splits ``ال`` gives every definite word a shared article-token component,
    so the feature is trivially available; that dedicated-token mechanism *is* the hypothesis).
  - **number (singular vs plural)** — a harder, *non-tautological* feature: plurality is marked
    by internal patterns / suffixes, not one clean clitic, so it tests whether the embedding
    *geometry* encodes the feature, not just whether a marker token is present.
- **Type-level train/test split** so the probe is scored on unseen word types (no memorization).
- **Controls:** a random label (must sit at AUC ≈ 0.5) and word-length (must be decodable —
  proves the probe + embeddings work). Stratified by **register** (MSA, Masri), never averaged.

Run: ``HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev --extra tokenizers \\
    python experiments/embedding-probes/run_probes.py``
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from train_embedding_model import TrainConfig, train

from fanous_lens.tokenizers.corpora import load_corpora
from fanous_lens.tokenizers.morphological import word_morph_features_batch
from fanous_lens.tokenizers.train import get_tokenizer

APPROACHES = ["bpe", "unigram", "wordpiece", "morfessor", "morphological"]
SEED = 0
MIN_PER_CLASS = 60  # skip a (feature, register) cell with fewer positives or negatives


def held_out_word_types(n_msa: int, n_masri: int, skip: int) -> dict[str, list[str]]:
    """Unique word types from a held-out corpus slice, per register (disjoint from training)."""
    from camel_tools.tokenizers.word import simple_word_tokenize

    msa, masri = load_corpora(max_msa=skip + n_msa, max_masri=skip + n_masri)
    out = {}
    for reg, sents in (("MSA", msa[skip:]), ("Masri", masri[skip:])):
        seen: dict[str, None] = {}
        for s in sents:
            for w in simple_word_tokenize(s):
                if len(w) >= 2 and any("؀" <= c <= "ۿ" for c in w):
                    seen.setdefault(w, None)
        out[reg] = list(seen)
    return out


def label_words(words: list[str]) -> dict[str, np.ndarray]:
    """Per-word labels: definiteness (0/1), number plural (0/1, None dropped), length, random."""
    feats = word_morph_features_batch(words)
    rng = np.random.default_rng(SEED)
    definite = np.array([1 if f["definite"] else 0 for f in feats])
    number = np.array(
        [1 if f["number"] == "p" else (0 if f["number"] == "s" else -1) for f in feats]
    )
    length = np.array([len(w) for w in words])
    length_bin = (length > np.median(length)).astype(int)
    random = rng.integers(0, 2, size=len(words))
    return {"definite": definite, "number": number, "length": length_bin, "random": random}


def pooled_vectors(w_e: torch.Tensor, encode, words: list[str]) -> np.ndarray:
    """Mean token embedding per word (zeros for the rare empty encoding)."""
    d = w_e.shape[1]
    rows = np.zeros((len(words), d), dtype=np.float32)
    for i, w in enumerate(words):
        ids, _offsets = encode(w)
        if ids:
            rows[i] = w_e[ids].mean(dim=0).numpy()
    return rows


def probe_auc(x: np.ndarray, y: np.ndarray) -> float | None:
    """Type-split logistic-regression AUC; None if a class is too small."""
    mask = y >= 0
    x, y = x[mask], y[mask]
    if y.sum() < MIN_PER_CLASS or (len(y) - y.sum()) < MIN_PER_CLASS:
        return None
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(x_tr, y_tr)
    proba = clf.predict_proba(x_te)[:, 1]
    return round(float(roc_auc_score(y_te, proba)), 3)


def main() -> None:
    words = held_out_word_types(n_msa=4_000, n_masri=4_000, skip=20_000)
    labels = {reg: label_words(words[reg]) for reg in ("MSA", "Masri")}
    for reg in ("MSA", "Masri"):
        d = labels[reg]
        print(
            f"{reg}: {len(words[reg])} types | definite +{int(d['definite'].sum())} | "
            f"plural +{int((d['number'] == 1).sum())} sing {int((d['number'] == 0).sum())}",
            flush=True,
        )

    features = ["definite", "number", "length", "random"]
    table: dict[str, dict] = {}
    for approach in APPROACHES:
        cfg = TrainConfig(approach=approach, max_msa=20_000, max_masri=10_000, max_steps=3_000)
        print(f"\ntraining {approach} (device={cfg.device}) ...", flush=True)
        res = train(cfg, save=True)
        w_e = res["W_E"]
        encode = get_tokenizer(approach, res["tok_config"])
        table[approach] = {"final_loss": res["final_loss"]}
        for reg in ("MSA", "Masri"):
            vecs = pooled_vectors(w_e, encode, words[reg])
            table[approach][reg] = {f: probe_auc(vecs, labels[reg][f]) for f in features}
        print(f"  {approach}: {table[approach]}", flush=True)

    print("\n=== PHASE A PROBE AUC ===")
    hdr = f"{'approach':14}{'reg':>7}" + "".join(f"{f:>10}" for f in features) + f"{'loss':>9}"
    print(hdr)
    for approach, d in table.items():
        for reg in ("MSA", "Masri"):
            cells = "".join(
                f"{(d[reg][f] if d[reg][f] is not None else '—'):>10}" for f in features
            )
            print(f"{approach:14}{reg:>7}{cells}{d['final_loss']:>9}")
    print("\nJSON:\n" + json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
