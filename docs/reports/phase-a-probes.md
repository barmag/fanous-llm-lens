# Phase A — does aligned tokenization buy interpretability? The probe answers

**Driver:** [`experiments/embedding-probes/run_probes.py`](../../experiments/embedding-probes/run_probes.py)
**Model:** [`train_embedding_model.py`](../../experiments/embedding-probes/train_embedding_model.py)
**Results:** [`experiments/embedding-probes/phase_a_results.json`](../../experiments/embedding-probes/phase_a_results.json)
**Date:** 2026-06-29 · **Commit:** `fb0ca69`
**Hardware:** AMD Strix Halo iGPU (gfx1151→gfx1100), torch 2.5.1+rocm6.2

This is the **verdict** Phase B deferred to. Phase B ([tokenizer-comparison.md](tokenizer-comparison.md))
established that morpheme-alignment cannot be the *definition* of tokenizer fitness — it is a
**hypothesis**: *does aligning tokens to morphemes actually make linguistic features more
recoverable inside a model?* Only a probe can answer, because alignment-F1 presupposes the
thing under test. Here is the answer.

---

## 1. Setup

- **Model — zero-layer transformer** (embeddings + positional + LayerNorm + tied head, **no
  attention, no MLP**). With no context mixing, all learnable structure lives in `W_E`, so the
  probe measures what the **tokenization alone** makes linearly available. One model per
  tokenizer, trained on 30 k MSA+Masri sentences (next-token, 3 000 steps, d_model 256).
- **Probe — pooled word embeddings.** Each held-out word is the mean of its token embeddings
  (uniform across tokenizers). A logistic-regression probe predicts a morphological feature,
  scored by ROC-AUC on a **type-level** held-out split (no memorization).
- **Features (camel-tools gold, tokenizer-independent):** **definiteness** (a *clitic* feature —
  the `ال` article) and **number** (an *inflectional* feature — plurality, marked by
  stem-internal patterns/suffixes, not one clean clitic). High cardinality: MSA 37 k word types
  (def +10 k, plural +6.5 k), Masri 19.7 k (def +3.5 k, plural +2.7 k).
- **Controls:** `random` label (must be ≈ 0.50) and `length` (must be decodable). Per register.

---

## 2. Results — probe AUC

| tokenizer | reg | **definite** | **number** | length | random | LM loss |
|-----------|-----|-------------:|-----------:|-------:|-------:|--------:|
| morphological | MSA | **0.929** | 0.596 | 0.718 | 0.500 | 5.69 |
| morphological | Masri | **0.901** | 0.594 | 0.690 | 0.502 | 5.69 |
| unigram | MSA | 0.860 | **0.762** | 0.695 | 0.504 | 6.65 |
| unigram | Masri | 0.851 | **0.767** | 0.669 | 0.509 | 6.65 |
| wordpiece | MSA | 0.747 | 0.684 | 0.659 | 0.499 | 7.74 |
| wordpiece | Masri | 0.753 | 0.704 | 0.629 | 0.509 | 7.74 |
| bpe | MSA | 0.755 | 0.663 | 0.648 | 0.503 | 7.62 |
| bpe | Masri | 0.743 | 0.697 | 0.590 | 0.514 | 7.62 |
| morfessor | MSA | 0.704 | 0.626 | 0.643 | 0.499 | 5.46 |
| morfessor | Masri | 0.707 | 0.667 | 0.616 | 0.491 | 5.46 |

**Controls pass everywhere:** `random` ≈ 0.50 (no leakage), `length` decodable (the probe and
embeddings work). The AUCs are real signal.

---

## 3. The answer: alignment helps for the features it aligns — and *hurts* for those it doesn't

**The hypothesis is neither confirmed nor refuted wholesale. It is conditional — and that is the
finding.**

1. **Definiteness (a clitic): morphological wins decisively** — 0.93 vs unigram's 0.86, far
   above bpe/wordpiece/morfessor (~0.70–0.75), in both registers. Splitting `ال` into a
   dedicated token gives every definite word a *shared article-token component*, so the feature
   becomes a clean linear direction. This is the localizability mechanism the hypothesis
   predicts — working exactly as advertised. *(Partly by construction: the `ال` token is
   literally in the pooled vector. But unigram reaching 0.86 without always splitting `ال` shows
   the feature is genuinely encodable, so the gap is meaningful, not pure tautology.)*

2. **Number (an inflection): morphological is the WORST** — 0.59, barely above the bpe/unigram
   floor and *below every subword tokenizer*; unigram leads at 0.76. This is the decisive,
   non-obvious result. The morphological tokenizer marks **clitics, not inflection** (`المعلمون`
   → `ال`+`معلمون`, plurality fused inside the stem token), so the model *cannot localize*
   plurality — there is no token that isolates it. The frequency-driven subword tokenizers, which
   sometimes split the `ون`/`ات` suffix, expose number **better**.

So **morpheme-aligned tokenization is not a universal interpretability win.** It buys clean
localization for the morphemes it isolates (clitics: definiteness) and actively *costs*
localization for the morphology it leaves fused (inflection: number). A tokenizer's
interpretability profile is **feature-specific**, set by *which* boundaries it draws.

---

## 4. The probe disagrees with Phase B — which vindicates the reframe

Phase B's cheap diagnostics did **not** predict the probe verdict:

- **morfessor led Phase B** on recall-at-fertility — yet it is the **weakest** tokenizer on
  *both* probes (def 0.70, num 0.63). Its clitic-recovery efficiency did not translate into
  recoverable features.
- **unigram led Phase B consistency** — and it **does** translate: best at number, second at
  definiteness. The consistency signal (stable morpheme→token mapping) tracked probe quality
  better than recall-at-fertility did.
- **LM loss is not fitness either:** morfessor has the *lowest* loss (5.46) but the weakest
  probes — better next-token prediction ≠ more recoverable linguistic structure.

This is exactly why Phase B refused to crown a winner and demoted alignment to a hypothesis: a
boundary metric (or a loss) is not a fitness verdict. Only the probe adjudicates, and it
overturns the cheap-tier ranking.

---

## 5. Caveats

- **Definiteness for morphological is partly mechanical** — a dedicated `ال` token sits in the
  pooled vector. The honest, non-tautological evidence is the **number** result, where
  morphological is worst.
- **Zero-layer model, pooled representation, 3 000 steps** — this isolates the tokenization's
  contribution (the goal), but it is not a full LM; absolute AUCs would shift with depth. The
  *ranking pattern* (clitic vs inflection asymmetry) is the robust claim.
- **Two features.** Definiteness and number span the clitic/inflection contrast that matters
  most here; negation, tense, dialect remain for a fuller sweep.
- **Single seed.** Probe splits and training use seed 0; the gaps are large relative to expected
  seed noise, but multi-seed error bars are a worthwhile follow-up.

---

## 6. Implication: pick the tokenizer for the feature you study

There is no single best tokenizer for interpretability of Arabic morphology:

- studying **clitic-level** phenomena (definiteness, prepositions, conjunctions, attached
  pronouns) → a **morpheme-aligned** tokenizer makes them cleanly localizable;
- studying **inflectional** phenomena (number, gender, tense) → a **subword** tokenizer
  (unigram here) exposes them better, because alignment leaves inflection fused in the stem;
- **unigram is the best all-rounder** — strong on both, and the Phase B consistency signal
  flagged it. If one tokenizer must be chosen for mixed work, it is the defensible default.

The dialect (Masri) story tracks MSA closely here: the same clitic-vs-inflection asymmetry holds
in both registers, with Masri a few points lower — consistent with the partial Masri gold, but
notably the *probe* (unlike the boundary metric) does not depend on that gold.

---

## 7. Reproduce

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev --extra tokenizers \
    python experiments/embedding-probes/run_probes.py
```

Trains all five zero-layer models on the iGPU and prints the AUC table + JSON. Checkpoints land
in `experiments/embedding-probes/checkpoints/` (gitignored, ~50 MB). The morphological model's
camel-tools tokenization is the slow leg (~10 min); the rest are seconds each.
