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

**Metric 1: Morpheme boundary precision & recall**
- Gold standard: camel-tools morphological segmentation on a held-out test set of 500 MSA sentences + 500 Masri sentences
- For each tokenizer, measure: what fraction of token boundaries coincide with morpheme boundaries?
- Report precision, recall, F1 by register (MSA, Masri)

**Metric 2: Token-level concept consistency**
- Define 10 linguistic features: negation, future, past-tense, plural, feminine, possessive, 1sg pronoun, 2sg pronoun, wh-question, progressive aspect
- For each feature, measure: across all occurrences, how often does the feature map to the same token(s)?
- High consistency → the feature is reliably represented at a predictable token position → good for probing

**Metric 3: Token-count efficiency**
- Mean tokens per triple on the 30 MSA/Masri minimal pairs (reuse notebook 01 framework)
- Lower is better (but secondary to alignment metrics)

**Metric 4: OOV rate**
- Fraction of tokens in held-out text that appear <2 times in training corpus. Lower is better.

### Corpus
- **MSA:** Wikipedia 20231101.ar (~600k articles, ~400M tokens) — from Stage 1b
- **Masri:** `amgadhasan/arabic_tweets_dialects` filtered to EG (~200k tweets, ~50M tokens) — from Stage 1b
- **Test set:** 30 minimal pairs (existing) + 500 MSA sentences (held-out Wikipedia) + 500 Masri sentences (held-out tweets)

### Deliverables
1. A comparison table: tokenizer × metric × register
2. Per-tokenizer visualization: token boundaries on a few example sentences, color-coded by linguistic feature
3. Recommendation: which tokenizer(s) to take into Phase A

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

- Phase B produces a clear ranking: morphological-aware ≥ Unigram ≥ WordPiece ≥ Morfessor ≥ BPE on morpheme alignment F1
- Phase A shows that the morphological-aware tokenizer yields higher probe accuracy for at least 5 of 10 linguistic features compared to BPE, with AUC improvement >0.1
- The experiment is reproducible in <30 min on the Strix Halo iGPU
