# Superposition Geometry — book two design

**Date:** 2026-07-23
**Notebook:** `notebooks/superposition/superposition_geometry.ipynb`
**Reference:** Elhage et al. 2022, *Toy Models of Superposition* — "The Geometry of
Superposition" and "Learning Dynamics" sections (local copy:
`docs/papers/elhage2022-toy-models-superposition.html`).
**Predecessor:** book one, `notebooks/superposition/toy_models_of_superposition.ipynb`
(merged via PR #6). Book one's recap deferred four threads; this notebook takes three of
them and leaves computation-in-superposition for a possible book three.

## Hypothesis

Superposition is not amorphous. Features arrange themselves into **uniform polytopes**
(digons, triangles, tetrahedra, pentagons, square antiprisms), measurable as fractional
**feature dimensionality** clinging to ½, ⅔, ¾, ⅖, ⅜. Training reaches these
configurations through discrete **energy-level jumps** visible in the loss curve, and
**non-uniformity** deforms the geometry smoothly until it snaps to a different
configuration.

## Scope

### In
1. **Feature dimensionality + plateau sweep** — define
   `Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)²`, sweep sparsity, reproduce the sticky-plateau plot of
   `D* = m/‖W‖²_F` vs `1/(1−S)` plus the per-feature scatter.
   *(Amended 2026-07-23 against the Colab: the paper text says n=400, m=30, but the
   Colab's actual geometry experiment uses n=200, m=20 with 20 sparsity instances
   trained simultaneously in one batched loop — `feature_probability =
   20^−linspace(0,1,20)`, constant importance, AdamW lr=1e-3 constant, 10k steps,
   batch 1024. We follow the Colab and adopt its batched-instances device as a lib
   cell; it also collapses the sweep's runtime from ~20 sequential runs to one.)*
2. **Polytope identification** — show the plateaus ARE polytopes via interference-graph
   components (the paper's tegum factors) and per-component PCA + Gram-matrix angles.
3. **Energy-level jumps** — one training run in the digon regime; per-feature
   dimensionality trajectories aligned with the loss curve; discrete drops coincide
   with jumps.
4. **Non-uniform superposition (one-feature perturbation only)** — the paper's
   pentagon-stretch experiment: n=5, m=2, uniform `1−S = 0.05`, vary one feature's
   sparsity; deform smoothly, then snap.

### Out
- Computation in superposition (abs-value circuit) — book three candidate.
- Correlated/anti-correlated feature geometry (the paper's harder non-uniform case).
- The phase diagram (Section 3) — unless a sliver serves narrative.
- Compressed-sensing theory, adversarial-examples connection.
- Arabic/MSA/Masri content — stated as out-of-scope in the notebook, as in book one.
- Real models, TransformerLens.

## Code reuse decision

Book one's `ToyModel`/`make_batch`/`train` live as `# lib:`-marked cells inside its
notebook (no `src/` module). Book two **restates them as its own compact `# lib:` cells**
(~40 lines), noted in markdown as "same model as book one, restated so this notebook
stands alone." Self-contained, same exec-from-notebook testing convention, and the
restatement doubles as a recap. (Alternatives considered: promoting to
`fanous_lens.toy_models` — DRY but hides the build from the reader and leaves two
sources of truth; exec-ing book one's cells — hidden cross-notebook dependency.)

## Narrative arc (measure first, then explain)

- **Act 0 — The question book one left open.** The pentagon reappears. Book one
  *observed* it; it never said why five features in 2-D pick a *regular* pentagon.
  Tease: the model is solving a physics problem (Thomson problem).
- **Act 1 — Rebuild the toy.** Restated lib cells; quick retrain of the n=5, m=2
  pentagon as the known-good calibration case.
- **Act 2 — The instrument: feature dimensionality.** Define `Dᵢ` as a lib cell.
  Name-then-experiment: predict by hand (dedicated → 1, antipodal → ½, pentagon
  vertex → ⅖, dropped → 0), then measure on Act 1's and book-one-style solutions.
  Trust the instrument only after it reads known cases correctly.
- **Act 3 — The uniform sweep.** n=400, m=30, importance ≡ 1, sparsity grid pinned
  from the paper's Colab. Plot `D*` vs `1/(1−S)` + per-feature scatter. Payoff:
  dimensionality clings to ¾, ⅔, ½, ⅖, ⅜. Heavy cell, cached.
- **Act 4 — The plateaus are polytopes.** For models on each plateau: threshold
  `|Ŵᵢ·Wⱼ|` → interference graph → connected components (tegum factors) → PCA each
  small component into its own subspace and look. Gram matrices confirm the angles
  (tetrahedron → −⅓). Closes Act 0's question: a plateau at p/q is q features sharing
  p dimensions as a uniform polytope.
- **Act 5 — Energy-level jumps.** One run in the digon regime, checkpointing `Dᵢ`
  during training. Twin plot: dimensionality trajectories + loss. Discrete loss drops
  coincide with level jumps — geometry explains the local minima book one apologized
  for.
- **Act 6 — Non-uniform: stretch and snap.** n=5, m=2, uniform pentagon at
  `1−S = 0.05`; vary one feature's sparsity. Denser → neighbors repel; sparser → it
  shrinks; far enough → snap to a different configuration.
- **Recap + handoff.** One-line numeric claims per act. Handoff to book three:
  everything so far is *storage* — can a model compute through these polytopes?

Interlude markdown cells between acts where connective tissue helps, as in book one.
Runtime approach: **in-notebook + cache** (reader watches training happen; first pass
slow, re-runs fast).

## Code units (each small, single-purpose, `# lib:`-marked, tested)

Tested by the existing exec-from-notebook pattern in a new
`tests/superposition/test_geometry.py`:

- `ToyModel` / `make_batch` / `train` — restated from book one; tests re-verify here.
- `feature_dimensionality(W) -> [n]` — the `Dᵢ` formula. Unit tests against
  hand-constructed `W`: identity → 1s, antipodal pairs → ½s, exact pentagon → ⅖s,
  zero column → 0.
- `frobenius_dims_per_feature(W) -> float` — `D* = m/‖W‖²_F`.
- `interference_components(W, threshold) -> list[list[int]]` — connected components of
  the thresholded interference graph. Tested on constructed block-structured `W`.
- `project_component(W, component) -> [k, ≤3]` — PCA of a component's features into
  their own subspace for plotting.
- Plot helpers, one plot each: dimensionality scatter vs `1/(1−S)`, polytope gallery,
  dimensionality-trajectory + loss twin plot, pentagon-deformation strip.

**Caching:** Act 3 sweep and Act 5 trajectory write `.pt` caches under
`notebooks/superposition/cache/` (book one needed no cache at its scale, so this
notebook establishes the pattern; add the directory to `.gitignore`).
Seeds fixed and logged. Exact hyperparameters (sparsity grid, steps, optimizer/LR)
pinned against the paper's Colab during implementation, per the repo's
"verify against the primary source, not memory" rule.

## Honesty & success criteria

- Plateau values are **attractors, not guarantees** — the paper notes stalls in
  non-uniform configurations. Save results first, report as-is. If ⅜ (square
  antiprism) doesn't appear at our scale/seed, that's a named negative.
- Act 4 reports Gram-matrix angles numerically, not just pictures.
- Act 6's snap may land somewhere other than the paper's configuration — report what
  forms.
- No pass/fail gate anywhere that could hide a negative.

**Success looks like:**
- Per-feature scatter shows ≥3 distinct plateaus at predicted fractions.
- At least digon, triangle, tetrahedron, pentagon identified geometrically.
- ≥1 loss drop aligned with a dimensionality jump.
- Pentagon deformation monotone with the perturbed feature's sparsity until a snap.

**Runtime:** first pass ≤ ~30 min CPU — the batched-instances loop trains all 20
sparsities at once; smoke-test ~100 steps and extrapolate before launching the full
10k. Cached re-run <10 min end-to-end, per repo convention. GPU optional via the
approved ROCm stack.

**Convention notes:** narrative/reproduction notebook → ships executed with outputs.
Commit messages name results, not changes.
