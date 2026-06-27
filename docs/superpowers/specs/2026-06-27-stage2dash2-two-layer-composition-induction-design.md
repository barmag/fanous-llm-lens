# Stage 2dash² — Two-layer attention: composition & induction on real Arabic

**Date:** 2026-06-27
**Status:** Design (awaiting user review)
**Branch:** `stage2dash2-two-layer-composition-induction`
**Source paper:** Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021),
the two-layer attention-only section.

## North Star

Reproduce the framework paper's **two-layer attention-only result — induction heads arising
from attention-head composition — rigorously, at faithful scale, on real Arabic.** This is
the two-layer counterpart to Stage 2dash exactly as 2dash was the faithful-scale counterpart
to the toy single-layer rungs (2a/2b):

- **2dash** proved the *one-layer* result (a single attention layer = bigram direct path +
  per-head skip-trigram circuits), by hand, on a paper-class Arabic model loaded from an
  offline checkpoint.
- **2dash²** proves the *two-layer* result: the full composition algebra (Q/K/V-composition,
  virtual attention heads, term importance, eigenvalue copying analysis) culminating in an
  **induction head**, which is specifically the **K-composition** path — the layer-1 key
  reading a layer-0 previous-token head's output out of the residual stream.

2c already shows induction *qualitatively* on a tiny in-notebook model; 2dash² is the
*rigorous, faithful-scale* treatment 2c only sketched.

## Audience & register

The **"dash track" register**: mech-interp-literate, denser than the beginner ladder rungs.
This is a **deliberate, user-chosen departure** from the project's default no-ML-background
audience (`project_audience_first`). The opening cell states the reader prerequisite
explicitly:

> Assumes you have done **2c** (induction intuition at toy scale) and **2dash** (the
> by-hand circuit-decomposition method). This is the densest notebook in the curriculum.

All five pedagogical conventions from `feedback_pedagogical_scaffolding` still hold verbatim:
bilingual RTL (`<div dir="rtl">`, headers `## N. عربي · English`, Arabic prose first),
shape-spine annotations on every tensor (`← (dim1, dim2, ...)`), upfront limitations in
prose, name-then-show discipline, and a closing recap → handoff cell.

## Faithfulness caveat (must appear in the notebook)

The paper's `attn-only-2l` used learned positional embeddings (added to the residual) +
LayerNorm. Our model is **LN-free + shortformer** (inherited from 2dash). This is a
*principled deviation*: it makes the two-layer path expansion **exact** and induction
**purely content-based**. We reproduce the paper's **results** on a model deliberately
configured for **exact decomposition** — *not the paper's literal architecture*.
"Faithful-scale" ≠ "faithful architecture", stated in one sentence in the notebook.

Shortformer does not break induction: the previous-token head is a positional Q/K pattern
(which shortformer supports) whose OV writes *content* into the residual — exactly what the
layer-1 key reads via K-composition.

## Attribution correctness (must appear in the notebook)

§4 demonstrates all three composition types in weight space. §8 states plainly that
**induction is the K-composition path**. Q- and V-composition are real and shown, but are
*not* what builds the induction head. A "full algebra" notebook that blurs this teaches the
wrong thing.

## Deliverables (mirror the 2dash full pipeline)

1. **`notebooks/education/train_stage2dash2.py`** — offline, run-once training script
   (~hours on the iGPU, headless). Copy of `train_stage2dash.py` with minimal changes (below).
2. **Checkpoint** under `notebooks/education/checkpoints/stage2dash2/` — `model.pt`
   (state_dict + config), reused `tokenizer.json`, `metrics.json` (now including **per-head
   induction scores**). Gitignored (corpus is large). Local-first load → HF-hub fallback.
3. **`notebooks/education/stage2_dash2_composition_induction_reference.ipynb`** — the single
   dense reference notebook (10 sections) that loads the checkpoint and does the fast
   interpretability. **Reference-only** (no `_experiment` stub), matching 2dash.
4. **CI**: a `mock_stage2_dash2` in `verify_notebooks.py` (sets `FORCE_TINY=True`, no-ops
   plotly) and a registration line for the new reference notebook.
5. **Docs**: a `2dash²` row in the README education-ladder table + a roadmap checkbox.

## Training script — `train_stage2dash2.py`

Copy `train_stage2dash.py`; minimal faithful changes:

- **`n_layers = 2`** (config flows into the checkpoint's `config` dict; the notebook's
  `_model_from_ckpt` already rebuilds via `make_tiny_model(n_layers=c["n_layers"], ...)`, so
  no notebook-side change is needed for depth).
- Keep **`d_model=512`, 8 heads, attn-only, `normalization_type=None`,
  `positional_embedding_type="shortformer"`**. A second attn layer adds only ~1.5–3M params
  (embed/unembed dominate), so there is no reason to shrink `d_model`.
- **Reuse the existing 2dash `tokenizer.json` and cached `tokens.npy`** — identical 12k
  unicode-BPE vocab and identical corpus. This is what makes the **1-layer-vs-2-layer
  comparison on identical tokens** (the notebook's opening beat) clean and honest.
- **Throughput logging:** add a `tok/s` line to the training loop so the next genuine ~1h
  offline run reports throughput as a byproduct (no separate GPU-grabbing benchmark — see
  "Performance & substrate" below).
- **Precision flag:** add `--bf16` (recommended on) — bf16 autocast around the forward,
  params stay fp32 under AdamW. Lower memory + bandwidth (gentler on the unified-memory pool
  that drives the display) and some throughput gain, with no change to optimization dynamics.
- **Batch:** keep default `--batch 32`. Larger batch is a headless-only lever (it halves the
  number of gradient updates since steps are pinned to one epoch, so it needs LR re-tuning,
  and it stresses the display-driving iGPU). Document this in the script docstring.

### Verification gate (the key robustness move)

Induction heads in a 2-layer attn-only model on natural language are among the most robust
phenomena in mech interp; they emerge early and reliably well below 340M tokens. We do not
pressure-test whether they appear — we **assert** it before the checkpoint is declared good:

- Generate batches of `[N random in-vocab tokens][the same N tokens]`.
- For each head, compute the standard **induction score**: mean attention from position `i`
  in the second copy to position `(first-occurrence-of-that-token + 1)`. (TransformerLens
  ships `induction_score`; reuse or replicate it.)
- **Assert ≥1 head crosses a threshold** (e.g. 0.4) before saving the checkpoint.
- Write the full per-head induction-score vector into `metrics.json`. The notebook reads it
  in §8 to locate the induction head — so we *know* it emerged, and the notebook has the
  numbers it needs without recomputing from scratch.

## Notebook structure — 10 sections (one dense notebook)

1. **مقدمة · Why one layer isn't enough** — load the **2dash 1-layer** model, run the repeat
   task, watch it *fail* to copy novel tokens. Skip-trigrams can't copy unseen tokens →
   motivates a second layer. (Mirrors the paper's own motivation.)
2. **التفكيك لطبقتين · The two-layer path expansion** — the logit equation expanded into
   direct-path / individual-head / virtual-head terms.
3. **نحمّل الموديل · Load the 2-layer model & tokenizer** — local → HF fallback, `FORCE_TINY`
   path for CI. Reuses the shared `EVAL_TEXT` Arabic paragraph convention from 2dash.
4. **التركيب: Q · K · V** — all three composition types, measured in weight space
   (Frobenius-norm ratios, as in the paper).
5. **الرؤوس الافتراضية · Virtual attention heads** — composed head pairs as effective heads.
6. **أهمية الحدود · Term importance** — rank the expansion terms by contribution.
7. **تحليل القيم الذاتية للنسخ · Eigenvalue copying analysis** — positive-eigenvalue test on
   OV circuits.
8. **رأس الاستقراء = مسار الـ K-composition · Induction *is* the K-composition path** — the
   synthesis. Locate the induction head from `metrics.json`; identify the layer-0
   previous-token head it reads from; prove the K-composition by hand. **This section makes
   the attribution explicit (induction = K-composition, not Q/V).**
9. **بيشتغل على عربي جديد؟ · Fires on fresh Arabic** — held-out Arabic repeat sequence; show
   the induction head attends back to the repeated token and the logit lifts.
10. **الخلاصة والخطوة الجاية · Recap & handoff.**

## CI / `FORCE_TINY` contract

Mirror 2dash. `mock_stage2_dash2(ctx)` sets `ctx["FORCE_TINY"] = True` and no-ops
`plotly.graph_objects.Figure.show`. Under `FORCE_TINY` the notebook builds a **tiny,
network-free 2-layer model** (no checkpoint, no HF download) so CI runs in seconds.

The tiny model has **no trained induction head**, so every "induction found" / threshold
assertion in §8–9 must be **guarded under `FORCE_TINY`** (run the code paths to prove they
execute, skip the strong numeric assertions) — matching however 2dash guards its assertions.
Register the new reference notebook in `verify_notebooks.py`.

## Performance & substrate (lesson learned 2026-06-27)

The Strix Halo **iGPU drives the display**; a batch×precision throughput sweep saturated it
and crashed the window manager (`feedback_igpu_drives_display`). Therefore:

- **No dedicated GPU benchmark.** Throughput is measured for free via the `tok/s` log line
  added to `train_stage2dash2.py`, during the real (headless) training run.
- The GPU stack is the established combo: **torch 2.5.1+rocm6.2 +
  `HSA_OVERRIDE_GFX_VERSION=11.0.0`**, run via `uv run --extra rocm`
  (`project_stage2_conventions`).
- Run the ~1h training **headless / with the desktop idle**.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Induction head doesn't emerge | Verification gate asserts ≥1 head before checkpoint is saved; if it fails, training did not succeed and we know immediately. (Very low risk for this model class.) |
| Notebook too dense for one file | User chose one notebook over a 2-part split with the full section list in view. If it proves unreadable in practice, splitting at §7/§8 is a clean later cut. |
| "Faithful-scale" misread as "faithful architecture" | Explicit one-sentence caveat in the notebook (above). |
| Composition attribution blurred | §8 states induction = K-composition specifically; Q/V shown but not credited. |
| GPU work destabilizes desktop | All heavy GPU work is headless; no benchmark sweeps; bf16 reduces memory pressure. |

## Out of scope

- MLP / non-attention components (this is attn-only, as in the paper's two-layer section).
- Retraining the 1-layer 2dash model (reused as-is for §1).
- An `_experiment` companion notebook (2dash track is reference-only).
- Any Streamlit/UI surface.
