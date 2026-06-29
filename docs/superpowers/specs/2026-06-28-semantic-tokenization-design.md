# Semantic Tokenization for Arabic MSA and Dialects

**Date:** 2026-06-28
**Status:** Design — approved, awaiting implementation
**Experiment:** Phase B (tokenizer benchmarking) → Phase A (embeddings-only model training)

## Problem

BPE tokenizers trained on English-heavy or even multilingual corpora (mGPT) fail to produce tokens that align with meaningful linguistic units for Arabic — especially for Masri (Egyptian Arabic) and minority dialects. Token boundaries fracture morphemes, roots, and grammatical markers, making mechanistic interpretability work (probing, activation patching, circuit discovery) harder because the features of interest (negation, plurality, tense, dialect) are spread across arbitrary byte-level splits rather than clean token boundaries.

## Goal

Build and systematically compare 5 tokenization approaches on their ability to produce **interpretability-aligned tokens** — tokens whose boundaries correspond to linguistic units (morphemes, roots, prefixes, suffixes, grammatical markers). Then train a zero-layer transformer on the best approach and measure whether probe-able linguistic features are linearly decodable from the embedding space.

## Approaches

### Approach 1: Morphological-aware tokenization (primary/recommended)
- **Method:** Use `camel-tools` Arabic morphological analyzer to segment into morphemes. Each morpheme becomes a token. Fall back to character-level for forms the analyzer cannot parse.
- **Dependency:** `camel-tools` (pip installable, has Masri dialect model)
- **Strengths:** Directly delivers morpheme-aligned tokens; analyzer is MSA-strong and has dialect support
- **Weaknesses:** Analyzer dependency; slower preprocessing; dialect accuracy not perfect
- **Mitigation:** Use camel-tools's Masri model; evaluate MSA and Masri accuracy separately; fallback to char-level for unparseable forms

### Approach 2: Unigram LM (SentencePiece) — statistical baseline
- **Method:** Train `sentencepiece` with `--type=unigram` on the MSA+Masri corpus. Vocab size ~8k.
- **Strengths:** Well-understood, used by T5/mT5; tends toward cleaner subword boundaries than BPE
- **Weaknesses:** Boundaries are frequency-driven, not linguistically grounded

### Approach 3: WordPiece — second statistical baseline
- **Method:** Train HuggingFace `WordPiece` tokenizer on same corpus. Vocab size ~8k.
- **Strengths:** Likelihood-gated merging; slightly more word-like than BPE
- **Weaknesses:** Same frequency-driven limitation; less common for Arabic

### Approach 4: Bayesian morphological segmentation (Morfessor) — unsupervised baseline
- **Method:** Train `morfessor` on corpus to learn unsupervised morphological segmentation. Convert segmentations to tokenizer vocabulary.
- **Strengths:** No analyzer dependency; learns from data; adapts to dialect forms
- **Weaknesses:** Less accurate than camel-tools on known MSA forms; non-standard integration

### Approach 5: Vanilla BPE (control / "before" baseline)
- **Method:** Train standard BPE on same corpus, matching mGPT's approach. Vocab size ~8k.
- **Strengths:** Reproduces existing mGPT behavior; serves as control
- **Weaknesses:** Known poor semantic alignment

## Phase B: Tokenizer Benchmarking

### Evaluation framework

> **Revised 2026-06-29 (binding).** The original framework treated morpheme-alignment **F1**
> as the headline fitness score. That is unsound and has been removed. The gold (camel-tools
> `calima-msa-r13`, `d3tok`) marks **clitics only** — not inflection (it leaves `يذهبون` whole,
> not `يذهب`+`ون`) — and is weak on Masri. Against an **incomplete** gold **precision is
> unmeasurable**: a boundary placed where the gold is silent is indistinguishable from a true
> boundary the gold missed, so any F1 rewards agreement with the gold's blind spots and the
> `morphological` tokenizer (whose vocab *is* the gold) scores 1.0 by tautology. Phase B is a
> **cheap CPU diagnostic tier**, not a verdict; the verdict is Phase A's probe (see "Phase
> relationship" below). Implemented in `tokenizers/evaluate.py`; reproduced by
> `docs/reports/compare_tokenizers.py`; written up in `docs/reports/tokenizer-comparison.md`.

**Metric 1: Clitic boundary recall — reported WITH fertility (no precision, no F1)**
- Gold: camel-tools `d3tok` clitic boundaries, restricted to words it can reconstruct, on a
  held-out set of MSA + Masri sentences. Recall only: *of the boundaries the gold is sure
  exist, how many did the tokenizer place?* (greedy one-to-one, ±1 char).
- **Always paired with fertility** (tokens/word): recall alone is gamed by over-segmentation
  (char-level scores 1.0). Also report `beyond-gold rate` — share of intra-word cuts where the
  gold is silent — as a *descriptor*, never as error (it is often correct inflection/dialect).
- Report per register (MSA, Masri); never average. Masri recall is a **lower bound** (partial
  gold); always cite coverage with it.

**Metric 2: Morpheme consistency (gold-free) — the localizability signal**
- For a set of shared morphemes, measure across host words how often the morpheme maps to the
  same token(s): `top-share` (↑) and signature `entropy` in bits (↓).
- High consistency → the morpheme is at a predictable, stable token position → a feature for it
  can be **localized**, which is the actual interpretability question. Gold supplies the
  *list* of morphemes, not the scoring, so this is not bottlenecked by the gold's completeness.

**Metric 3: Token-count efficiency (fertility)**
- Mean tokens per word, per register. Reported alongside recall (Metric 1) so over-segmentation
  is always visible; not a standalone "lower is better".

**Metric 4: OOV / UNK rate**
- Fraction of emitted ids that are `[UNK]` on held-out text, per register. Note: for the
  per-piece encoders (morfessor, morphological) UNK is *independent* of fertility.

### Corpus
- **MSA:** Wikipedia 20231101.ar (~600k articles, ~400M tokens) — from Stage 1b
- **Masri:** `amgadhasan/arabic_tweets_dialects` filtered to EG (~200k tweets, ~50M tokens) — from Stage 1b
- **Test set:** 30 minimal pairs (existing) + 500 MSA sentences (held-out Wikipedia) + 500 Masri sentences (held-out tweets)

### Deliverables
1. A comparison table: tokenizer × diagnostic × register (recall+fertility, beyond-gold, consistency, UNK)
2. Per-tokenizer visualization: token boundaries on a few example sentences, color-coded by linguistic feature
3. A **shortlist (not a winner)**: rule out dominated tokenizers; carry the rest into Phase A. Phase B can *eliminate*, not *crown* — the diagnostics legitimately disagree (e.g. recall-at-fertility vs consistency), and only the probe resolves which property matters.

### Phase relationship (binding)
Morpheme alignment is a **hypothesis**, not the definition of fitness: BPE may never split `ال`
yet leave a model perfectly probeable for definiteness. So **Phase A's probe is the primary
fitness judge**; Phase B is the cheap CPU pre-filter that narrows the field and surfaces
tensions. Do not present any Phase B number as the verdict.

## Phase A: Embeddings-Only Model Training

### Model
- Zero-layer transformer: just a token embedding table `W_E` (vocab_size × d_model) + learned positional embeddings
- `d_model = 256` (small enough to train fast, large enough for meaningful probes)
- Train on the MSA+Masri corpus using standard next-token prediction (cross-entropy loss)

### Training details
- **Optimizer:** Adam, lr=1e-3, cosine schedule
- **Batch size:** 512 tokens
- **Steps:** 100k (~50M tokens seen)
- **Hardware:** ROCm on Strix Halo iGPU
- **Checkpoint:** every 10k steps

### Probe evaluation
Train 10 linear probes (logistic regression) on the embedding space `W_E[t]` for each token position:
- **Feature probes:** negation, future, past-tense, plural, feminine, dialect (MSA vs Masri), wh-question, progressive aspect, 1sg pronoun, possessive
- **Control probes:** token length (in chars), token frequency, random label (should stay at chance)
- **Metric:** probe accuracy (ROC-AUC) per feature per tokenizer variant
- **Key comparison:** does the morphological-aware tokenizer produce higher probe accuracy than BPE on the same embedding size?

### Deliverables
1. Probe accuracy table: tokenizer variant × feature × AUC
2. Visualization: embedding space PCA colored by feature (for the best tokenizer)
3. Quantified claim: "Morphological-aware tokenization improves negation probe accuracy from X to Y"

## Architecture & Dependencies

```
fanous-lens/
  tokenizers/              # New module
    __init__.py
    morphological.py       # camel-tools wrapper
    train.py               # Training scripts for all approaches
    evaluate.py            # Evaluation framework (metrics 1-4)
  experiments/
    tokenization-benchmark/  # Phase B notebooks/scripts
    embedding-probes/        # Phase A notebooks/scripts
  data/
    tokenizer-eval/         # Gold-standard segmentations, test sets
```

Key dependencies:
- `camel-tools` — morphological analysis
- `sentencepiece` — Unigram LM training
- `tokenizers` (HuggingFace) — WordPiece, BPE training
- `morfessor` — Bayesian segmentation
- `torch` — embedding model training
- `sklearn` — probe classifiers

## Risks

1. **camel-tools Masri accuracy** — if the analyzer fails on too many Masri forms, the morphological approach degrades. Mitigation: measure MSA vs Masri accuracy separately; fallback to char-level.
2. **Embedding-only model may not learn meaningful features** — without attention, the model may learn surface statistics rather than linguistic features. This is actually a feature, not a bug: it measures what the *tokenization* alone contributes to feature separability.
3. **Corpus size** — 50M Masri tokens may not be enough for stable embeddings. Mitigation: train on combined MSA+Masri (450M total); report per-register metrics.
4. **Morfessor integration** — morfessor doesn't produce a standard tokenizer vocabulary. Mitigation: convert its segmentations to a vocabulary by collecting all unique segments.

## Success criteria

> Revised 2026-06-29: the old criterion ("Phase B produces a clear ranking on morpheme
> alignment F1") presumed both the unsound F1 and that morphological would win it. Replaced.

- **Phase B (CPU diagnostic):** produces honest per-register diagnostics (clitic recall +
  fertility, beyond-gold, consistency, UNK) and a defensible **shortlist** — at minimum it
  rules out dominated tokenizers. It is *not* expected to crown a winner; the recall-vs-consistency
  disagreement is an accepted, documented outcome. *(Met: bpe/wordpiece shown dominated;
  morfessor leads recall-at-fertility, unigram leads consistency — see `tokenizer-comparison.md`.)*
- **Phase A (probe, the verdict):** for the shortlisted tokenizers, report probe AUC per
  feature per tokenizer with control probes at chance. The success bar is a **clear,
  feature-localized separation** for at least one shortlisted tokenizer over BPE on ≥5 of 10
  features (AUC Δ > 0.1) — *or*, equally publishable, the null result that alignment does **not**
  improve probe accuracy (which would refute the morpheme-alignment hypothesis cleanly).
