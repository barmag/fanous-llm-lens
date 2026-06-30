# Phase A (depth) — does adding layers reconstruct what the tokenizer obscured?

**Driver:** [`run_probes_depth.py`](../../experiments/embedding-probes/run_probes_depth.py) ·
**Model:** [`train_depth_model.py`](../../experiments/embedding-probes/train_depth_model.py) ·
**Results:** [`phase_a_depth_results.json`](../../experiments/embedding-probes/phase_a_depth_results.json)
**Date:** 2026-06-29 · **Base commit:** `3b5a679` · **Hardware:** Strix Halo iGPU, torch 2.5.1+rocm6.2

Phase A ([phase-a-probes.md](phase-a-probes.md)) showed the zero-layer pooled-mean probe is
**insensitive to training** — it scores the *tokenization*, not the model. The natural next
question: does **depth** change that? Add 1–2 transformer blocks (attention + MLP), read the
residual stream at layers 0/1/2, and measure the quantity that was ≈0 at zero layers — the
**trained − untrained increment** (paired, 3 seeds, mean ± range). The prediction was that depth
would *reconstruct the fused `ون`/`ات` plural suffix*, lifting `number` on `morphological`.

**It did not — and the failure is the result.** Depth adds nothing at inference for either
feature. The only training increment is **at the embedding layer (L0)**, on `definiteness`, for the
tokenizers that spread the article across many tokens. Once decomposed honestly, the data say:
*the layers don't reassemble morphology; training reshapes the embedding table, and only when the
feature has a shared, generalizing surface signature to align.*

This is an **isolated-word / pooled-mean** probe (words encoded alone; see
[`train_depth_model.py`](../../experiments/embedding-probes/train_depth_model.py) docstring). The
caveats in §5 bound every null below.

---

## 1. Setup

- **Model:** `d_model=256`, **2 pre-norm blocks** (4-head causal attention + GELU MLP), tied head,
  same 30 k MSA+Masri corpus / 3 000 steps as Phase A. One probe per residual-stream depth
  **L0 (embeddings+pos), L1, L2**, read off the *same* 2-layer model.
- **Increment, paired per seed:** construct the model → probe **untrained** (random init) at all
  depths → train the *same* model → probe again. Same-seed pairing gives a clean Δ; tokenizer
  build is cached once per approach, so 3 seeds only multiply the fast training loop.
- **Probe:** Phase A's held-out word **types**, labels, controls, and type-split — reused verbatim
  — plus **StandardScaler** before the logistic probe (load-bearing once depths differ in scale).
- **Controls hold at every depth:** `random` = 0.50 throughout; `length` decodable (0.72–0.84).
  The probe is valid at all three depths.

---

## 2. Finding A — for fused inflection (`number`), depth does nothing; the Phase-A ranking is intact

| tokenizer | number L0 | L1 | L2 | shape |
|-----------|:---:|:---:|:---:|---|
| unigram | **0.80** | 0.78 | 0.78 | best, *erodes* with depth |
| wordpiece | 0.73 | 0.71 | 0.71 | erodes |
| bpe | 0.72 | 0.70 | 0.70 | erodes |
| morfessor | 0.64 | 0.63 | 0.63 | erodes |
| morphological | **0.62** | 0.60 | 0.60 | worst, erodes |

Two robust facts (MSA; Masri tracks within ±0.02):

- **Depth does not help anyone on `number`. It slightly *hurts*.** The plural signal is most
  recoverable at the **embedding** layer and is washed out — not built up — by passing through
  blocks and mean-pooling. Every per-seed Δ at L1/L2 is ≈0 or negative (e.g. bpe
  Δ−0.007 [−0.010, −0.003]; unigram Δ−0.000).
- **Phase A's tokenization ranking survives 2 layers unchanged:** unigram 0.78 ▸ subwords ~0.70 ▸
  morphological 0.60, a stable ~0.18 gap. Depth does not compress it.

The fused suffix is **weakly recoverable, not destroyed**: morphological `number` shows a small but
real *embedding-level* training increment (L0 Δ+0.035 [+0.029, +0.040], clears the noise floor) —
training imparts some generalizing (type-split) plural signal to the fused embeddings. But it caps
at 0.62 and **depth erodes it** (L2 Δ+0.015). Two shallow layers do not substitute for a tokenizer
that keeps the suffix.

---

## 3. Finding B — a real training increment, but at the **embedding**, not from the layers

| tokenizer | definite L0 | L1 | L2 | Δ at L0 | added by depth (L0→L2) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| morphological | **0.93** | 0.94 | 0.93 | **+0.002** | +0.000 |
| unigram | 0.89 | 0.90 | 0.90 | +0.022 | +0.004 |
| bpe | 0.81 | 0.82 | 0.82 | **+0.047** | +0.012 |
| wordpiece | 0.80 | 0.82 | 0.82 | **+0.071** | +0.014 |

The increment is **real and large** for bpe/wordpiece (wordpiece L0 Δ+0.071 [+0.057, +0.098]) — but
**it is already present at L0**. The extra gain from running through both blocks (L0→L2) is
+0.012 / +0.014 — **inside the ±0.014 noise floor**, with per-seed bands that overlap. So the
honest mechanism is **not** "attention/MLP reassemble definiteness at inference." It is
**training reshaping the embedding table**, read at L0; the layers add nothing measurable on top.

Why the increment sorts the way it does (mechanism, grounded in §4's segmentations):

- **morphological** gives definiteness a **single dedicated `ال` token**, so every definite word
  pools the same article vector — a constant offset that is maximal **even untrained** (0.93,
  Δ≈0). Nothing for training to add. This is the Phase-A "one-token artifact," confirmed.
- **bpe / wordpiece** have **no standalone article token**; definiteness is spread across *many
  distinct* word-initial subwords (`الم`, `الكتاب`, `المعل`, …). Untrained, those vectors are
  unaligned; **training pulls them toward a shared definiteness direction** (Δ+0.05–0.09) — and
  that alignment lives in the embeddings.

Suggestive cross-experiment note (not a clean claim): Phase A's zero-layer model showed bpe
definite Δ−0.006 (training did nothing at the embedding); here the same L0 probe shows Δ+0.047. The
difference is that **2 layers were present during training**, so backprop through them could shape
the embeddings. But Phase A used no StandardScaler and this is a different (2-layer) model, so
isolating "depth-during-training shapes embeddings" cleanly needs a scaler-controlled zero-layer
rerun. Flagged, not asserted.

---

## 4. A mechanism consistent with both findings (and verified on the segmentations)

Why is `definiteness` recoverable from every tokenizer's embeddings while fused `number` is not,
when both surface markers (`ال`, `ون`) are physically present in the word? Tokenizing 5 definite +
5 sound-plural words (subword tokenizers fresh; morphological per [phase-a-probes.md](phase-a-probes.md) §4):

| word | bpe | unigram | wordpiece | morphological |
|------|-----|---------|-----------|---------------|
| المعلم (def) | الم·علم | الم·علم | المعل·##م | ال·معلم |
| الكتاب (def) | الكتاب | الكتاب | الكتاب | ال·كتاب |
| معلمون (pl) | م·علم·**ون** | م·علم·**ون** | معل·##مون | معلمون |
| طالبات (pl) | طالب·**ات** | طالب·**ات** | طالب·##ات | طالبات |

> **Proposed rule:** a feature is recoverable when it has a **shared, high-frequency, generalizing
> surface signature** in the token stream — and not when it is **fused into lexically-unique whole
> tokens**.

- **`ال` (definiteness)** is **word-initial in every tokenizer** — a standalone token in
  morphological, glued into the first subword elsewhere, but *always at the same position with the
  same form*. That consistency is a learnable signature that generalizes to held-out types — so
  every tokenizer recovers it (and training aligns the distributed cases, §3).
- **`ون`/`ات` (number)** is **split into a shared suffix token** by the subword tokenizers
  (`…·ون`, `…·ات`) — recoverable — but **fused into the stem** by `morphological`, producing
  thousands of distinct whole-word plural tokens (`معلمون`, `طالبات`) with **no shared carrier**.
  "Plural" is then spread across lexically-unique singletons, and the **type-split blocks the only
  remaining route, per-type memorization**. Not recoverable.

So the governing variable is **not** "aligned vs not" but **shared-surface-signature (recoverable)
vs fused-into-unique-tokens (lost)**. Alignment is one way to win or lose that lottery, feature by
feature — and `morphological` wins it for the clitic and loses it for the inflection.

---

## 5. Caveats (these bound the nulls in §2)

- **Mean-pooling may dilute a post-mixing signal.** That `number` *erodes* with depth is exactly
  what a mean-pool would do to a feature attention relocates onto one position. A last-token or
  per-position readout could change Finding A; the null is "not recoverable *by a mean-pool of an
  isolated word at d_model 256 / 2 layers*," not "not recoverable, period."
- **The LM objective doesn't reward poolable morphology.** Next-token prediction never asks the
  model to make plurality linearly recoverable from a word's mean vector, so absence of the feature
  is partly absence of pressure, not only of capacity.
- **Small model.** 2 layers, 256-d. A deeper/wider model, or in-context words (the deferred
  robustness pass), could move Finding A.
- **§4 is a mechanism consistent with the data, not a proof.** The segmentations verify its
  premises; it has not been tested by intervention (e.g. forcing a shared plural token into
  `morphological` and re-probing — a clean follow-up).
- **Byte-premium confound (untested).** Each model trains on the same *sentences* but, because
  tokenizers differ in fertility, **not** the same number of tokens or the same per-step signal —
  so cross-tokenizer AUC gaps conflate feature-exposure with effective-data effects. The published
  decompositions ([tokenization-lit-note.md](tokenization-lit-note.md): Inequities 2025;
  Why-morph-complex 2024) find data/fertility dominate at scale, so a **fertility-matched** rerun
  (equalize *tokens*, not sentences) is needed before the ranking reads as a pure tokenization
  property. Our refusal to compare LM loss across tokenizers (Phase A §5) is the same caution on a
  different axis.

---

## 6. Verdict against the hypothesis

| prediction | outcome |
|---|---|
| `number × morphological` increment largest, **grows with depth** | **Falsified.** No depth gain for anyone on number; it erodes. Ranking unchanged. |
| `definite` Δ ≈ 0 everywhere (one-token artifact) | **Half right.** ≈0 only for the *aligned* tokenizer; **largest increment** on bpe/wordpiece — but at **L0**, not from depth. |
| gap `unigram − morphological` (number) compresses with depth | **Falsified.** Stable ~0.18 through 2 layers. |
| circuit moves embedding → layers with depth | **Falsified.** Both features are read at the embedding; the layers add nothing at inference. The only learned gain is an **embedding-level** definiteness alignment (depth-enabled via backprop). |

**One-line claim:** *Two layers do not let a small model reconstruct morphology its tokenizer
fused — `number` stays an embedding-level property the layers can't add to (and the Phase-A
tokenizer ranking holds through depth). The only training increment is at the embedding: aligning
the many distinct article-bearing tokens into a shared definiteness direction, large for tokenizers
without a dedicated `ال` token and ≈0 for the aligned one. Recoverability is governed by whether the
feature keeps a shared surface signature, not by depth.*

---

## 7. Reproduce

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev --extra tokenizers \
    python experiments/embedding-probes/run_probes_depth.py
```
