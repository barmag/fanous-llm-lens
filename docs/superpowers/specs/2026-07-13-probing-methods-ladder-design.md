# Probing-methods ladder — design

**Date:** 2026-07-13 · **Status:** design approved, scaffolding rung p_a
**Branch:** `probing-methods-ladder`

## North star

A bilingual (RTL Arabic + English), Colab-runnable **probing-methods ladder** that
teaches *probing* one method at a time, the way the existing architecture ladder
(`notebooks/education/stage*`) teaches a transformer one component at a time. Where the
architecture ladder builds a *model* piece by piece, this ladder builds *probing skill*
piece by piece, each rung earning a new capability and each ending where the next needs
to begin.

The ladder stands on existing repo research as its worked payload — the Phase A
zero-layer probe ([`docs/reports/phase-a-probes.md`](../../reports/phase-a-probes.md))
and Phase-A-depth ([`docs/reports/phase-a-depth.md`](../../reports/phase-a-depth.md)) —
rather than inventing results. Those reports are the source of truth for the numbers a
rung teaches; the notebooks make the *method* that produced them legible and re-runnable.

## Placement & conventions

- **Location:** `notebooks/education/`, as a continuation of the education ladder.
  Reuses the ladder's infrastructure: `tiny.py` (tiny models / device probe),
  `corpus.py` (MSA Wikipedia + Masri EG-tweets streaming), `verify_notebooks.py`
  (end-to-end smoke test).
- **Rung prefix:** `probe_*`, sitting alongside `stage2_*`. Each rung ships a
  `*_reference` notebook (fully worked) and a `*_experiment` notebook (key cells hollowed
  to `# TODO` for self-study), matching the existing pattern.
- **Bilingual / RTL:** Arabic text blocks use `display(HTML(..., dir="rtl"))`; markdown is
  pedagogical (no process/meta talk about repo conventions).
- **Runtime bar:** each reference notebook runs end-to-end on a single seed in <10 min on
  the iGPU; heavy sweeps are offloaded to a sibling script and checkpoint-cached.

## Ladder shape

Spine is the **method** ladder (A); once the method is taught, a **feature** extension (B)
sweeps Arabic morphology. Rungs beyond p_a are sketched here for coherence but are **not**
built this session.

| Rung | Question it answers | What it adds | Payload source |
|---|---|---|---|
| **p_a** | What is a probe? | Logistic probe on pooled activations of a **zero-layer** model; AUC = linear recoverability; controls (random, length). The headline lesson: **a probe can score 0.9 while the model computed nothing** — the score read the *tokenization*, not the model. | Phase A |
| p_b | Can you trust it? | The rigor rung: untrained-`W_E` baseline, selectivity / control tasks, type-split. The trained−untrained increment is the honest quantity. | Phase A §2 |
| p_c | Where does it live? | Probe the residual stream layer-by-layer (add depth + MLP); accuracy-vs-depth. Confronts the honest null: **depth/MLP alone did not manufacture a learned signal** — the levers are readout (last-token vs mean-pool), task pressure, in-context words. | Phase-A-depth |
| p_d | Is the direction causal? | Steer / patch along the probe direction; correlation → causation. Bridges to circuits. | (new) |
| p_e+ (B) | Which features, and where? | Fix the method, sweep features: dialect → definiteness → number → tense/negation. | (new) |

### Why p_a is zero-layer (design rationale)

A zero-layer model is embeddings + positional + LayerNorm + tied head — all learnable
structure in `W_E`. This is deliberate, not a simplification:

1. **The mechanics are depth-independent** — extract → pool → logistic fit → AUC →
   controls is identical at any depth, so p_a teaches the whole workflow.
2. **It is the only place the core caveat is cleanly demonstrable.** Phase A showed the
   pooled zero-layer probe is *insensitive to training* (trained vs untrained AUC within
   ±0.014) yet reads **0.93 AUC on definiteness** — a probe scoring high while the model
   learned nothing measurable, because the score came from the tokenization. Zero
   computation is the cleanest possible control for "a probe measures the input encoding
   as much as the network."
3. **Adding an MLP does not fix this and would blur it.** Phase-A-depth ran 2 blocks
   (attention + GELU MLP) and found depth adds nothing at inference; the lever for a
   *learned* signal is readout/pressure/scale, not depth. That arc belongs at p_c, one
   lever at a time — not smuggled into rung 1.

## Rung p_a — structure (the deliverable)

The reference notebook opens **not** with a probe but with a **calibration section** that
earns every downstream choice from evidence (repo "smoke-test before you commit to cost"
discipline). We do not assume the MSA/Masri data on a tiny model yields above-control
signal — we measure it and let the numbers pick the rung's headline feature.

1. **Look at 5 rows** — sample MSA (Wikipedia) and Masri (EG tweets) sentences; confirm
   labels + RTL rendering.
2. **Calibration sweep** — build a zero-layer model (start: `d_model` 256, Phase-A-style,
   ~30k MSA+Masri sentences) and measure probe AUC **vs the control floor** (random ≈
   0.50, length decodable) for 2–3 candidate features (dialect, definiteness, number)
   across a small grid of dataset size / `d_model`. Heavy grid offloaded to
   `calibrate_probe.py`; notebook reads cached results.
3. **Decision gate** — pick the rung's headline feature + config from the numbers. If
   dialect clears the floor it becomes the ladder's through-line; if only morphology
   clears it, that is the honest finding and dialect is deferred to p_c. Either outcome is
   reported, not gated away.
4. **The teaching probe** — with the chosen config, walk one probe end-to-end (extract
   activations → pooled vector → logistic fit → AUC → control comparison) as the
   pedagogical payload, closing on the trained-vs-tokenization caveat and a recap/handoff
   to p_b.

The `_experiment` twin hollows the probe-fitting and control cells to `# TODO`.

## Scaffold artifacts (this session)

- `notebooks/education/probes.py` — shared, self-contained helper mirroring `tiny.py`'s
  style: `LinearProbe` (StandardScaler + logistic regression, ROC-AUC), pooling utils
  (mean-pool a word/sentence from token activations), and `controls` (random-label,
  length, untrained-`W_E` baseline). Depends only on numpy/sklearn/torch.
- `notebooks/education/probe_a_linear_reference.ipynb` — structure + bilingual markdown
  written; calibration + probe cells real where cheap, heavy sweep guarded behind a cached
  script call and a shrinking knob for CI.
- `notebooks/education/probe_a_linear_experiment.ipynb` — the `# TODO`-hollowed twin.
- `notebooks/education/calibrate_probe.py` *(optional / if the sweep exceeds the runtime
  bar)* — offline heavy sweep, like `train_stage2c.py`; writes a cached results JSON the
  notebook loads.
- `verify_notebooks.py` — register `probe_a` with a shrinking mock (small corpus, few
  steps, no-op plotly) so the reference runs in seconds in CI.

## Non-goals / deferred

- p_b–p_e notebooks, and the causal/steering machinery for p_d.
- The B feature-sweep beyond naming it.
- README roadmap edits beyond (at most) a one-line pointer to the new rung.
- Any claim about a *learned* representation from p_a — by construction p_a measures the
  tokenization; the learned-signal question is p_c's, answered against Phase-A-depth.

## Success criteria for this session

- Design doc committed.
- `probes.py` importable; `probe_a` reference + experiment notebooks present and
  structurally complete (markdown + cells), with the calibration-first arc.
- `probe_a` reference passes `verify_notebooks.py` under its mock (runs end-to-end).
- No result asserted beyond what calibration actually measures — honest negatives OK.
