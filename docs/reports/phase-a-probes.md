# Phase A — does aligned tokenization buy interpretability? The probe answers

**Driver:** [`run_probes.py`](../../experiments/embedding-probes/run_probes.py) ·
**Baseline:** [`run_baseline.py`](../../experiments/embedding-probes/run_baseline.py)
**Model:** [`train_embedding_model.py`](../../experiments/embedding-probes/train_embedding_model.py)
**Results:** [`phase_a_results.json`](../../experiments/embedding-probes/phase_a_results.json) ·
[`phase_a_baseline.json`](../../experiments/embedding-probes/phase_a_baseline.json)
**Date:** 2026-06-29 · **Commit:** `69c9c84` · **Hardware:** Strix Halo iGPU, torch 2.5.1+rocm6.2

This is the **verdict** Phase B deferred to. Phase B ([tokenizer-comparison.md](tokenizer-comparison.md))
showed morpheme-alignment cannot be the *definition* of fitness — it is a **hypothesis**: *does
aligning tokens to morphemes make linguistic features more recoverable inside a model?* The
probe answers it — and a built-in control reframes what "inside a model" even means here.

---

## 1. Setup

- **Model — zero-layer transformer** (embeddings + positional + LayerNorm + tied head, **no
  attention, no MLP**). All learnable structure lives in `W_E`. One model per tokenizer, 30 k
  MSA+Masri sentences, next-token, 3 000 steps, d_model 256, on the iGPU.
- **Probe — pooled word embeddings.** Each held-out word = mean of its token embeddings
  (uniform across tokenizers). Logistic-regression probe → ROC-AUC on a **type-level** held-out
  split (no memorization).
- **Features (camel-tools gold, tokenizer-independent):** **definiteness** (a *clitic* — the
  `ال` article) and **number** (an *inflection* — plurality, marked by stem-internal
  patterns/suffixes, not one clean clitic). MSA 37 k types (def +10 k, plural +6.5 k); Masri
  19.7 k (def +3.5 k, plural +2.7 k).
- **Controls:** `random` (must be ≈ 0.50), `length` (must be decodable), **and an untrained-`W_E`
  baseline** (§2) — the decisive one.

---

## 2. The decisive control: training barely moves the probe

Before reading any ranking as "what the model learned," compare each trained AUC to the **same
probe on a fresh, untrained `W_E`** (random init, same tokenizer, same labels/split):

| tokenizer | definite: untrained → trained (Δ) | number: untrained → trained (Δ) |
|-----------|:---:|:---:|
| bpe (MSA) | 0.761 → 0.755 (−0.006) | 0.669 → 0.663 (−0.006) |
| unigram (MSA) | 0.872 → 0.860 (−0.012) | 0.773 → 0.762 (−0.011) |
| wordpiece (MSA) | 0.734 → 0.747 (+0.013) | 0.693 → 0.684 (−0.009) |
| morfessor (MSA) | 0.711 → 0.704 (−0.007) | 0.628 → 0.626 (−0.002) |
| morphological (MSA) | 0.931 → 0.929 (−0.002) | 0.586 → 0.596 (+0.010) |

**Training reshaped the embeddings a lot, yet the probe did not move.** The loss fell steeply
during training (≈ 88 → 27 on the smoke run) and the five tokenizers reached *different* final
losses (5.46–7.74), so `W_E` was substantially rewritten — and still every probe Δ is within
**±0.014** (single-seed noise), average slightly negative. The probe is **invariant to whatever
geometry training imposed**.

Why: a pooled *mean* of token embeddings is a (here near-random) linear projection of the word's
**bag of tokens**, so by Johnson–Lindenstrauss the probe ≈ a bag-of-tokens probe — it fires when
the *tokens* covary with the feature, regardless of their learned values. So the honest claim is
not "training does nothing" (the model demonstrably learned — the loss says so) but: **at
zero-layer depth, pooled-mean linear probing is insensitive to training — it scores the
*tokenization*, not the model.** Phase A here ranks *tokenizations*; it is a principled,
gold-free fitness probe (it scores the real feature, no morpheme-boundary gold needed) whose
zero-layer reading the model does not influence. Measuring what *learning* adds needs depth —
with attention/MLP the trained−untrained increment becomes the interesting quantity, and is the
natural next experiment.

---

## 3. Probe AUC (trained; ≈ untrained per §2)

| tokenizer | reg | **definite** | **number** | length | random |
|-----------|-----|-------------:|-----------:|-------:|-------:|
| morphological | MSA | **0.929** | 0.596 | 0.718 | 0.500 |
| morphological | Masri | **0.901** | 0.594 | 0.690 | 0.502 |
| unigram | MSA | 0.860 | **0.762** | 0.695 | 0.504 |
| unigram | Masri | 0.851 | **0.767** | 0.669 | 0.509 |
| wordpiece | MSA | 0.747 | 0.684 | 0.659 | 0.499 |
| bpe | MSA | 0.755 | 0.663 | 0.648 | 0.503 |
| morfessor | MSA | 0.704 | 0.626 | 0.643 | 0.499 |

`random` ≈ 0.50 (no leakage), `length` decodable (probe works). AUCs are real signal.

---

## 4. The answer: alignment exposes the morphemes it isolates — and hides those it fuses

The hypothesis is **conditional**, and the condition is *which boundaries the tokenizer draws*:

Both features are tokenization properties (§2) — the distinction is **not** inventory-vs-learned
but **one-token artifact vs distributed-across-inventory**:

1. **Definiteness (a clitic): morphological's 0.93 is a one-token artifact — do not credit it.**
   A *single dedicated token*, `ال`, literally carries the label: every definite word pools the
   same `ال`-vector, a constant offset any probe recovers (untrained 0.931 ≈ trained 0.929). It
   is the Phase-A form of the Phase-B tautology (`morphological` matches the gold because its
   vocab *is* the gold). A dedicated `ال` token genuinely makes definiteness maximally
   localizable — but the score is handed over by one token, so the *gap* over other tokenizers
   is not a meaningful comparison.

2. **Number (an inflection): morphological is the WORST (0.59), and here the ranking is
   meaningful** — because number is **not** a one-token artifact. No single token equals
   "plural"; the signal is distributed across *whether the inventory isolates the sound-plural
   suffix as a reusable token*. Verified on the trained tokenizers: unigram and bpe split it
   (`المعلمون` → `…·ون`, `الطالبات` → `…·ات`), giving plurals a shared suffix token; morphological
   **fuses** it in the stem (`ال·معلمون`, `ال·طالبات`), so plurality has no shared carrier and is
   least separable. (Broken plurals — `رجال`, `كتب` — are kept whole by *all* tokenizers, which
   caps the ceiling for everyone and is why no tokenizer exceeds ~0.77.) Because no one token
   short-circuits the probe, the ordering **unigram 0.76 ▸ subwords ~0.69 ▸ morphological 0.59**
   is a trustworthy tokenization comparison, not a tautology.

So **morpheme-aligned tokenization is not a universal interpretability win.** It makes the
morphemes it *isolates* maximally available (clitics → definiteness) and the morphology it
*fuses* least separable (inflection → number). A tokenizer's interpretability profile is
**feature-specific**, fixed by where it cuts.

---

## 5. The probe ranking overturns Phase B

Phase B's cheap diagnostics did not predict this:

- **morfessor led Phase B** on recall-at-fertility — yet it is the **weakest** tokenizer on both
  probes (def 0.70, num 0.63). Clitic-recovery efficiency did not translate into separable
  features.
- **unigram led Phase B consistency** — and it translates: best at number, second at
  definiteness, the **best all-rounder**. The consistency signal tracked probe separability
  better than recall-at-fertility did.

(We do **not** compare LM loss across tokenizers: cross-entropy is computed over different
vocabularies and sequence statistics, so the numbers are not comparable and support no claim.)

This is exactly why Phase B refused to crown a winner and demoted alignment to a hypothesis: a
boundary metric is not a fitness verdict. The probe adjudicates and reorders the cheap-tier
ranking — and, via §2, cautions that the model's learned geometry is invisible to this
zero-layer probe, so these are tokenization comparisons, not statements about a trained network.

---

## 6. Caveats

- **This zero-layer pooled-mean probe is insensitive to training (§2)** — `W_E` was substantially
  rewritten (loss fell, finals differ) yet AUC held within ±0.014. So read every number as a
  *tokenization* property, not a learned one. The learned-representation question needs a deeper
  model; that is the natural next experiment (re-probe with layers; watch the increment appear).
- **morphological's definiteness win is a one-token artifact** (`ال`; untrained = trained), not a
  creditable gap. The meaningful, non-tautological result is **number**, where morphological is
  worst because it fuses the sound-plural suffix.
- **Two features** (definiteness, number) span the clitic/inflection contrast that matters most;
  negation, tense, dialect remain for a fuller sweep.
- **Single seed**; gaps are large vs expected seed noise, but error bars are a fair follow-up.
  The untrained baseline already de-risks the convergence worry (it bounds what training did).

---

## 7. Implication: pick the tokenizer for the feature you study

There is no single best tokenizer for Arabic-morphology interpretability:

- **clitic-level** phenomena (definiteness, prepositions, conjunctions, attached pronouns) → a
  **morpheme-aligned** tokenizer makes them cleanly localizable (a dedicated token);
- **inflectional** phenomena (number, gender, tense) → a **subword** tokenizer (unigram here)
  exposes them better, because alignment leaves inflection fused in the stem;
- **unigram is the best all-rounder** — strong on both, and Phase B's consistency signal flagged
  it. The defensible default when one tokenizer must serve mixed work.

The Masri story tracks MSA closely: the same clitic-vs-inflection asymmetry holds in both
registers, Masri a few points lower. Notably the *probe*, unlike the boundary metric, does not
depend on the partial Masri gold.

---

## 8. Reproduce

```bash
# trained probes (iGPU; morphological's camel-tools leg ~10 min)
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev --extra tokenizers \
    python experiments/embedding-probes/run_probes.py
# untrained baseline (CPU; reads the saved checkpoints + results)
uv run --extra rocm --extra dev --extra tokenizers \
    python experiments/embedding-probes/run_baseline.py
```
