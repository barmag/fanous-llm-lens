# Stage 1c Redesign: Learn the Vectors, Then Probe for Dialect

**Date:** 2026-06-03
**Status:** Design — awaiting implementation
**Scope:** `notebooks/education/stage1_c_subword_reference.ipynb` (full rewrite),
`notebooks/education/stage1_c_subword_experiment.ipynb` (full rewrite, scaffolded),
and a new fast unit test for the analysis logic.

## Problem

The current Stage 1c notebook loads `mGPT`, averages the pretrained embeddings of a
hand-picked word list, and runs unsupervised 3D PCA hoping MSA and Masri synonyms cluster
by meaning. They do not. The result is noise, for structural reasons:

- **Raw embeddings are not semantic.** A zero-layer view of a pretrained model exposes
  token embeddings, which mostly encode surface/distributional statistics (frequency,
  string length, shared characters). Meaning emerges only after attention/MLP layers.
  Asking raw embeddings to cluster *synonyms* is asking for something the data cannot give.
- **Averaging fractured subwords smears the signal.** Synonyms with different BPE fracture
  patterns average to unrelated points.
- **PCA is unsupervised.** It returns the directions of largest variance — here frequency
  and length — not the feature (dialect) the lesson is about.

Two earlier rescue ideas are ruled out by the curriculum: the "dialect tax / fracture"
story is already taught in another notebook, and multi-layer contextual semantics is a
later stage. The embeddings trilogy is **count chars (1a) → count word transitions (1b)
→ learn a vector per subword (1c)**, and 1c must deliver genuine, dialect-focused value
without straying into either neighbour's territory.

## Intended concept (the one "aha")

Train a zero-layer transformer on our own MSA+Masri text so the embeddings reflect *our*
dialect data, then ask one sharp question: **is dialect linearly encoded in the learned
embeddings?** The lesson rides on a contrast between two views of the *same* trained `W_E`:

- **Unsupervised PCA** surfaces the loudest variance (frequency/length) → dialect colours
  stay **mixed**.
- **A supervised linear probe** finds the one axis that separates dialect → a clean split
  with an **accuracy number**.

Takeaway, in one line: *PCA shows what is biggest; a probe shows what you asked for.*
This both delivers a real dialect result and teaches the PCA-vs-probe distinction — the
core technique the old notebook was missing.

## Design

### Model & data

- **Corpus.** Reuse Stage 1b's loaders verbatim for consistency: Wikipedia `20231101.ar`
  (MSA, streamed) and `amgadhasan/arabic_tweets_dialects` filtered to `EG` (Masri), with
  the same whitespace/Arabic-only cleaning (the regex writes the Arabic range using
  `ء-ي` unicode escapes, the repo convention for non-Arabic readers). Keep the
  two cleaned strings separate
  (`msa_text`, `masri_text`) — dialect labels depend on per-stream counts.

- **Tokenizer.** Train a small **BPE** tokenizer (~3,000 vocab) on the *combined* corpus
  using HuggingFace `tokenizers`. No model download — the notebook is fully self-contained.
  Log the vocab size and a few example MSA/Masri tokenisations (CLAUDE.md tokenizer-aware
  convention). Encode each stream to subword id sequences (`msa_ids`, `masri_ids`).

- **Model.** A literal zero-layer transformer: `embedding (W_E) → unembedding (W_U) →
  softmax over the next token`. This is a subword **bigram** next-token predictor. Train
  with cross-entropy (PyTorch, CPU, fixed seed, embedding dim 64, a few epochs over the
  `(current_token → next_token)` pairs pooled from both streams). Print the loss so the
  learner sees it fall. The trained `W_E` (a NumPy array, shape `[vocab, 64]`) is the
  artefact the analysis inspects. Runs well under the 10-minute budget on CPU.

### Analysis

- **Dialect labels (free, from frequency).** For each subword id, count occurrences in
  `msa_ids` vs `masri_ids`. Keep ids above a small frequency floor (e.g. total count
  >= 5). Label by ratio: `MSA` if the MSA share exceeds an upper threshold, `Masri` if it
  falls below a lower threshold, else `Shared`. Thresholds are explicit named constants.

- **PCA view.** 2-component PCA of the labelled tokens' `W_E` rows; scatter coloured by
  dialect (MSA / Masri / Shared). This is the "naive" mixed picture.

- **Probe view.** A logistic-regression linear probe trained on `W_E` rows to classify
  `MSA` vs `Masri` (Shared rows are held out of training and shown for context). Split the
  labelled tokens into train/test; report **held-out accuracy** as the headline number.
  Take the probe's weight vector as the dialect direction, project every token (including
  Shared) onto it, and draw a 1-D histogram per dialect with the decision boundary marked.
  Shared tokens should land between the two classes.

- **Render.** One `plotly` figure, **side-by-side**: left = PCA scatter (mixed), right =
  probe-axis histogram (split) annotated with the accuracy. Colour map: MSA red, Masri
  blue, Shared green (matches the existing notebook). `fig.show()`.

### Code shape

Keep helpers small and single-purpose so the analysis is unit-testable without training:

- `train_bpe(texts, vocab_size)` → trained tokenizer.
- `train_zero_layer(ids, vocab, dim, ...)` → `W_E` (NumPy). Self-contained PyTorch loop.
- `dialect_labels(msa_ids, masri_ids, min_count, hi, lo)` → `{token_id: "MSA"|"Masri"|"Shared"}`.
- `probe_dialect(W_E, labels)` → `(accuracy, direction, projections)`. No training of the
  embedding model inside — operates on a given `W_E`, so tests can inject a synthetic one.
- `plot_pca_vs_probe(W_E, labels, probe_result)` → builds the side-by-side figure.
- **Driver** wires corpus → BPE → train → labels → PCA + probe → plot.

### Notebook structure (reference)

1. Colab badge (unchanged, already correct).
2. Markdown: title + intro in the new framing; trilogy recap; hypothesis stated upfront;
   **upfront limits** ("these are toy embeddings from a zero-layer model trained on a small
   corpus; deep contextual semantics come in a later stage").
3. Code: install (`tokenizers`, `datasets`, `scikit-learn`, `plotly`, `pandas`, `torch`).
4. Markdown: name the BPE + corpus step.
5. Code: load corpus, train BPE, encode streams, log vocab size + example tokenisations.
6. Markdown: name the zero-layer model (embedding+unembedding bigram predictor).
7. Code: build pairs, train the model (seeded), print falling loss, expose `W_E`.
8. Markdown: explain free dialect labels and the PCA-vs-probe contrast.
9. Code: labels → PCA scatter + logistic probe (accuracy) + side-by-side figure.
10. Markdown: recap + handoff ("PCA missed dialect; the probe found it at X% — dialect is
    linearly encoded even in a zero-layer model. Next stage: multi-layer semantics.").

Apply the validated pedagogical patterns throughout: name-then-experiment, upfront limits,
RTL display for Arabic blocks, shape spine (annotate array shapes), recap+handoff.

### Experiment notebook

Mirror the reference structure, but replace the BPE-train, model-train, and analysis code
cells with TODO scaffolds + hints (no implementation), consistent with the other
`*_experiment.ipynb` notebooks.

## Verification

- **Fast unit tests** (`tests/education/test_stage1c_probe.py`, CPU, no network, no
  training): load the analysis code cell, inject a synthetic *separable* `W_E` + labels,
  and assert (1) the probe reaches high accuracy on separable data, (2) PCA returns the
  right shape, (3) `dialect_labels` assigns MSA/Masri/Shared correctly on synthetic counts,
  (4) the figure has two side-by-side panels. Mirrors the Stage 1b test pattern (load cell,
  inject synthetic globals, mock `Figure.show`, exec).
- **Manual end-to-end:** `verify_notebooks.py c` runs the whole notebook on the real
  datasets (mocked `Figure.show`); expect a non-trivial probe accuracy and both panels
  populated.

## Out of scope

- Stage 1a (char) and Stage 1b (word) notebooks — unchanged.
- POS / part-of-speech probing (considered; dropped to keep a single dialect-focused
  lesson and avoid manual labelling).
- Pretrained-embedding (mGPT) probing and contextual/multi-layer embeddings (the latter is
  a later stage).
- Any new plotting dependency beyond what the trilogy already uses.
