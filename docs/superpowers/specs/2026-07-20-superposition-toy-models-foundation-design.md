# Toy Models of Superposition — Foundation Notebook (design)

**Date:** 2026-07-20
**Status:** approved design; implementation plan to follow
**Anchor paper:** Elhage et al., *Toy Models of Superposition*, Transformer Circuits
Thread, Sept 2022. Saved locally at
[`docs/papers/elhage2022-toy-models-superposition.html`](../../papers/elhage2022-toy-models-superposition.html)
(complete self-contained HTML; the article is HTML-native and has no official PDF).

## North star fit

Mechanistic-interpretability tooling for small models. This notebook is the **first
rung of a superposition track**. Its deliverable is *understanding the mechanism* —
how and why a network packs more features than it has dimensions — not an Arabic/Masri
application. Masri is deliberately out of scope for book one and may become its own
future track once the mechanism is owned. This choice was made explicitly by the user:
the mechanism is where the pedagogical value is.

## Thesis (stated at the top of the notebook)

**A nonlinear model given sparse inputs will represent more features than it has
dimensions — "superposition" — and sparsity is the knob that turns it on. The cost is
interference.**

The notebook is a single self-contained narrative that runs end-to-end on one seed in
<10 min (in practice, seconds — the toys are tiny). It follows the repo's
investigation-notebook style (cf. `in_context_learning/icl_from_scratch.ipynb`,
`induction_heads_in_the_wild.ipynb`), not the education-ladder experiment/reference
pairing.

## Scope

### In (Cluster A — "superposition is real and sparsity drives it")
- The linear-vs-ReLU toy model, faithful to the paper.
- The flagship demonstration: 5 features embedded in 2 dimensions, dense → orthogonal,
  sparse → pentagon.
- `WᵀW` interference heatmaps and 2D feature-direction ("star") plots.
- A sparsity sweep that quantifies the transition.

### Out (deferred to a possible book two, `superposition_geometry.ipynb`)
- *Why* features pick pentagons vs tetrahedrons vs other uniform polytopes.
- The full phase diagram (Section 3 of the paper).
- Feature dimensionality (fraction-of-a-dimension per feature).
- Computation in superposition (the abs-value circuit).
- Relationship to adversarial examples / grokking / the strategic picture.

The pentagon **is shown** in book one as the reveal, but treated as an *observation*;
its mechanics are what get deferred. Book two is a "split if it grows" option, not a
commitment.

### Explicitly out (both books)
- No Arabic/MSA/Masri content. Stated as out-of-scope in the notebook itself.
- No real model, no TransformerLens. Pure synthetic toy.

## Model (faithful to the paper)

Notation: `n` = number of features, `m` = hidden dimensions (`m < n`), `W` has shape
`[m, n]` (column `W[:, i]` is feature `i`'s direction), `b` has shape `[n]`.

- **Data.** Each feature `xᵢ = 0` with probability `S` (the sparsity), otherwise drawn
  uniformly from `[0, 1]`. All features share one sparsity `S` (the paper's focus case).
  A single scalar **importance** `Iᵢ` weights each feature's loss term; importance
  decays across features (exact decay borrowed from the paper's Colab during
  implementation — e.g. `Iᵢ = r^i`).
- **Linear model.** `x' = WᵀW x`. Can represent at most `m` features orthogonally;
  superposition is impossible by construction. Superposition ⟺ `WᵀW` is not invertible.
- **ReLU output model.** `x' = ReLU(WᵀW x + b)`. Matches the linear model on dense data
  but superposes as sparsity rises, sacrificing the least-important features first.
- **Loss.** Importance-weighted MSE: `Σᵢ Iᵢ (xᵢ − x'ᵢ)²`, averaged over the batch.

Exact hyperparameters (importance decay ratio, sparsity grid points, training steps,
optimizer/LR, seed) are pinned against the paper's Colab in the implementation plan, per
the repo's "verify against the primary source, not memory" rule.

## Narrative arc

- **Act 0 — The puzzle.** Polysemantic neurons in real models: why would a model *want*
  more features than neurons? Hook to the paper's Background & Motivation. State the
  thesis and the plan.
- **Act 1 — The linear ceiling.** Build the linear toy `x' = WᵀW x`. Inspect `WᵀW`. A
  linear map holds at most `m` features orthogonally — superposition impossible. Train on
  n=5, m=2 → it keeps the top-2 features by importance (PCA-like), discards 3. This is
  the baseline/ceiling.
- **Act 2 — Add the ReLU, stay dense.** Introduce `x' = ReLU(WᵀW x + b)`. On dense inputs
  (`S = 0`) it behaves the same — top-2 orthogonal, 3 dead. Proves the ReLU *alone* is
  not the trick; sparsity is required.
- **Act 3 — Crank sparsity → the pentagon.** Sweep `S = 0 → 0.8 → 0.9`. Watch 2
  orthogonal → 4 antipodal pairs → 5-as-a-pentagon. Two visuals per regime: the W-columns
  in 2D (star plot) and the `WᵀW` heatmap (diagonal = represented, off-diagonal =
  interference). The existence proof — the money shot.
- **Act 4 — Sparsity phase, quantified.** Grid-sweep `S`; plot number-of-features-
  represented (via W-column norms) against sparsity. Turns "sparsity drives it" into a
  curve, not a vibe.
- **Recap + handoff.** What we proved (superposition is real; sparsity drives it;
  interference is the cost). What we deferred to book two, and why each deferred item
  needs its own home.

## Code units (small, single-purpose, re-runnable)

- `ToyModel(n_features, n_hidden, use_relu=True)` — `nn.Module`; params `W [m, n]`,
  `b [n]`; `forward` returns `x'`. `use_relu=False` gives the linear model.
- `make_batch(n_features, sparsity, batch_size, generator)` — sparse uniform features.
- `train(model, importance, sparsity, steps, ...)` — importance-weighted MSE; returns
  the trained model and a loss trace.
- Visualization helpers, each one plot: `plot_features_2d(W)` (star plot),
  `plot_WtW(W)` (interference heatmap), `plot_sparsity_sweep(results)`.

Each cell is small and guards its own work; heavy cells (the sweeps) cache so re-runs are
fast, per the repo's idempotent/checkpoint-cached convention. At this scale caching is a
nicety, not a necessity.

## Honesty & success criteria

- **Honest negatives are results.** The paper notes these toys hit local minima and
  "energy-level jumps." If a clean pentagon does not form at our seed, the notebook
  reports what it got and *why* (seed sensitivity / local minimum), never a claimed pass.
  Always save results first, then report the metric. No hard pass/fail gate that would
  hide a negative.
- **Success looks like:** the pentagon reproduces at high sparsity; the linear model and
  the dense ReLU model both show top-2 orthogonal with the rest dead; the sparsity sweep
  shows features rising with sparsity. Result is a plot or a one-line numeric claim, not a
  vibes paragraph.
- **Runtime:** end-to-end on a single seed in <10 min; CPU is sufficient, GPU optional.

## Notebook conventions

- Markdown is pedagogical only — no process-talk, no citing repo conventions inside cells.
- Paper-hooked section openers: what the paper did and why it matters *here*.
- Small, numbered sub-steps when a cell grows hard to scan.
- Clear all outputs before commit.

## Dependencies

torch, numpy, matplotlib — all already present in the repo. No new dependencies. No
TransformerLens, no real model, no network access at run time.
