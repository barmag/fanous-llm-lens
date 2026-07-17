# Induction ablation — causal follow-up to "induction heads in the wild"

**Date:** 2026-07-17
**Status:** approved design, pre-implementation
**Artefact:** `notebooks/in_context_learning/induction_ablation.ipynb` (new notebook)
**Predecessor:** `induction_heads_in_the_wild.ipynb` (PR #3, branch `induction-heads-in-the-wild`)

## Motivation

The wild notebook ends on a named limitation: every result is correlational — a
matrix looks right, a score is high, a rank is 1. The blog draft closes with an
explicit promise: *"Knock out L4H11 and L5H5, or L3H2 and L4H6, and watch whether
the induction score collapses. That is the next notebook."* This is that notebook.

Two causal paths were considered:

1. **Ablation (knock-out)** — primary. Known trap: induction behavior lives in a
   cluster, not one head (GPT-2 L6H9 scores 0.917 vs L5H5's 0.930), and the IOI
   literature documents backup heads that strengthen when the primary is ablated.
   Single-head ablation may under-collapse; the design makes that the finding, not
   a failure.
2. **Tuning toward the hand-crafted config (knock-in)** — stretch act only, in the
   form of OV eigenvalue surgery. Known trap: superposition — Pythia L4H6 spends
   ~31% of its OV eigenvalue mass on non-copying directions, so any weight edit
   conflates "removed copying" with "removed everything else". The design turns
   that conflation into the measurement.

## Hypotheses

- **H1 (knock-out):** if L5H5 (GPT-2) / L4H6 (Pythia) is the model's induction
  head, mean-ablating it collapses induction behavior: loss on the repeated half
  of random-token sequences returns toward the unrepeated baseline. Predicted
  wrinkle: partial collapse for single-head ablation, because of the cluster.
- **H2 (mediation):** if K-composition is causal, mean-ablating the prev-token
  head (L4H11 / L3H2) kills the *downstream* induction head's attention stripe to
  matched positions — an effect one layer removed from the intervention. This is
  the causal test of the wild notebook's star finding (K-composition rank 1/60
  and 1/48).
- **H3 (stretch, knock-in):** if Pythia L4H6's negative OV eigenvalue mass is
  non-copying work in superposition, removing those directions preserves or
  improves synthetic induction while degrading natural-text loss elsewhere.

## Act structure

GPT-2 small and Pythia-160m measured side-by-side in every act (same convention
as the wild notebook). All heads and thresholds inherited from the wild
notebook: GPT-2 prev-token L4H11, induction L5H5; Pythia prev-token L3H2,
induction L4H6; induction-score gate 0.2.

- **Act 0 — Setup + the yardstick.** Load both models via TransformerLens
  `HookedTransformer.from_pretrained` (LayerNorm-folded processing, same as
  before). Rebuild the repeated-random-token batch with the same generator/seed
  as the wild notebook. Define the two baselines every later act is measured
  against: mean loss on first (unrepeated) half vs second (repeated) half. The
  gap between them is the induction behavior, in nats.
- **Act 1 — Calibrate the scalpel.** Define mean-ablation: replace a head's
  `hook_z` output with its mean over a reference batch. Show zero-ablation
  alongside on one head, once, as the pedagogical beat about staying
  on-distribution — then use mean-ablation everywhere after.
- **Act 2 — Knock out the induction head (escalation ladder).**
  - **2a:** ablate the top induction head alone; report loss-gap recovery.
  - **2b (hydra check):** under that ablation, re-measure every remaining head's
    induction attention score; ask whether runners-up strengthen (GPT-2's
    L6H9, 0.917 clean; Pythia's runner-up read off the wild notebook's Act 1
    heatmap) — backup-head behavior made observable.
  - **2c:** ablate the whole above-gate cluster; expect near-full collapse.
- **Act 3 — Knock out the feeder (mediation).** Ablate only the prev-token head.
  Readouts: the induction head's own attention-to-matched-position score, and
  the model-level loss gap. Directly causal-tests K-composition.
- **Act 4 — Does it matter on real text?** Natural-text per-token loss delta
  under cluster ablation, on the same corpus sample as the wild notebook, split
  by repeated-bigram positions (where induction can act) vs everywhere else.
  Plus one compact direct-logit-attribution beat: the matched token's logit
  contribution from the induction head (its output projected through `W_U`),
  clean vs feeder-ablated.
- **Act 5 (stretch) — Purify the copier.** OV eigenvalue surgery on Pythia
  L4H6: eigendecompose the 64×64 full-OV circuit, reconstruct `W_V·W_O` keeping
  only positive-real-part eigendirections, write back into the model
  (snapshot the original `W_V`/`W_O` first so the cell restores and stays
  re-runnable). Measure the induction loss gap and natural-text loss both
  before and after.
  Superposition made causal: improvement on induction + damage elsewhere is the
  H3-consistent outcome; any other outcome is reported as measured.
- **Act 6 — Verdict + honest gaps.** One table: intervention × model × loss-gap
  recovered. Answer the blog's promise sentence directly. Name what remains
  untested.

## Methods

**Metrics (defined once in Act 0, reused everywhere):**

- **Induction loss gap** = mean cross-entropy on the first (unrepeated) half −
  mean cross-entropy on the second (repeated) half, over a batch of
  `rand_tokens ⊕ rand_tokens` sequences. Positive by construction when
  induction works: the repeated half is cheaper to predict. Ablation effect
  reported as collapse fraction `(gap_clean − gap_ablated) / gap_clean`
  (0% = no causal effect, 100% = full collapse).
- **Induction attention score** = attention mass at offset −(half_len − 1),
  where `half_len` is the length of one repeated half — identical to the wild
  notebook's Act 1 definition so hydra-check numbers are comparable across
  notebooks.
- **Natural-text delta** = per-token loss increase, split by repeated-bigram
  positions vs rest.
- **DLA** = matched-token logit contribution of the induction head via
  `W_U`-projection of its per-position output.

**Mean-ablation reference:** per-head mean `hook_z` vector averaged over batch
and position, computed on a separate reference batch drawn from the same
distribution as the eval batch (synthetic reference for Acts 2–3, natural-text
reference for Act 4). Zero-ablation appears once, in Act 1, as contrast.

**Implementation:** TransformerLens hooks on `blocks.{L}.attn.hook_z`, slicing
head `H`. No new dependencies. `uv run --no-sync` for all execution (venv has
ROCm torch + TransformerLens).

**Budgets:** forward passes only, on 124M/160M models — inside the iGPU memory
envelope and the <10-min single-seed bar. The hydra check reuses cached
attention patterns from a single ablated forward pass, not a per-head sweep.
No training loops.

## Conventions

- Seeds + commit SHA logged in the notebook.
- Each act opens with a paper hook: Olsson et al. 2022 (ablations and per-token
  induction losses), Wang et al. 2023 IOI (backup/hydra heads), Elhage et al.
  2021 framework (OV eigenvalues) — and *why it matters here*, not just what
  the code does.
- Save results before judging them: no hard pass/fail gate that makes a
  negative unobservable. A weak single-head collapse is the hydra finding.
- Small, single-purpose, re-runnable cells; numbered sub-steps when a cell
  grows.
- Clear outputs before commit.
- Smoke-test scale first (small batch, a few sequences) before the full batch.

## Failure handling

- **Act 5 instability:** eigenvalue surgery on LayerNorm-folded, low-rank
  factored weights may not reconstruct cleanly (complex eigenvectors of a
  non-symmetric real matrix must be recombined into a real matrix; rank may
  drop). If the surgered model degenerates, report the attempt with the named
  reason and cut the act from the blog. The act is explicitly a stretch.
- **Mean-ablation reference mismatch:** if synthetic-reference means put Act 4's
  natural-text runs visibly off-distribution, recompute means on natural text
  and say so in the markdown.

## Out of scope

- Gradient-based fine-tuning toward the toy config (heavier "tuning" path —
  deferred; eigenvalue surgery is the chosen knock-in form).
- Path patching / full causal-mediation graphs (IOI-style) — a later notebook
  if mediation results warrant it.
- Resample/patch ablation as the primary tool (mentioned as a pointer, not
  implemented).
- Any Masri-specific evaluation — this notebook stays on the English blog
  thread; the dialect track continues separately.

## Deliverables

- The notebook, running end-to-end on one seed in <10 min on the iGPU.
- Plots: escalation-ladder bar chart (loss-gap recovery per intervention per
  model); hydra-check before/after induction-score scatter; mediation
  attention-stripe comparison; natural-text delta split; Act 5 spectrum
  before/after with the two loss numbers.
- One verdict table (intervention × model × recovery).
- Blog section source material for the follow-up post.
