# ICL-from-scratch: training loop + logging — design

**Date:** 2026-07-12
**Notebook:** `notebooks/in_context_learning/icl_from_scratch.ipynb`
**Status:** approved, pending spec self-review + user review

## Problem

`icl_from_scratch.ipynb` reproduces Olsson et al. 2022 ("In-Context Learning and
Induction Heads") from zero: a 2-layer attention-only transformer trained on a
Pile-like corpus, instrumented with the paper's own diagnostics. Dataset
streaming, byte-level BPE tokenizer training, model config, and train/eval
sequence chunking are already written (cells 1-9). Nothing trains yet, and none
of the paper's nine metrics are implemented:

- PCA of token losses
- Loss over training
- Loss over training, broken down by context index
- Derivative of loss w.r.t. context index
- In-context learning (ICL) score
- Prefix-matching score
- Trace of QK eigenvalues
- Ablation to a "before-and-after" vector
- Ablation attribution to the ICL score

## Why this reproduction matters (context from prior art)

`notebooks/education/stage2c_induction_tinystories.ipynb`'s "Panel 0" already
tried a version of this on TinyStories and got an honest negative: aggregate
loss and a prefix-matching proxy showed no sharp phase transition, only a
gradual induction-score rise over ~2,000 steps. That notebook named two
reasons the test was weak rather than a refutation: (1) it measured aggregate
loss and a prefix-matching score, not the paper's actual ICL score (loss late
in context minus loss early in context); (2) TinyStories has no code and only
short documents, and code's near-verbatim repetition may be exactly what
sharpens the paper's signal. This notebook is designed to close both gaps:
`monology/pile-uncopyrighted` (Pile-CC + Gutenberg + GitHub, ~Pythia's own
training mix) as the corpus, and the paper's real per-position loss
decomposition instead of the aggregate-loss proxy.

## Scope decomposition

Nine metrics is too much for one spec. They decompose into five sub-projects
by data dependency — each reads the previous one's output, no re-instrumentation
needed later:

1. **Training loop + logging** (this spec) — the foundation. Every other
   sub-project reads its checkpoint and `metrics.json`.
2. **Loss/ICL-score curves** — loss over training, loss by context index, the
   derivative w.r.t. context index, and the ICL score itself. Pure
   post-processing of sub-project 1's `per_position_loss` matrix.
3. **Attention diagnostics** — prefix-matching score (mostly reuse of
   `induction_viz.prefix_matching_score`) and the QK eigenvalue trace (a static
   property of the trained weights, no training changes needed).
4. **PCA of token losses** — deeper loss diagnostic; may need per-sequence (not
   just batch-mean) loss capture, decided when this sub-project is scoped.
5. **Ablation** (before/after vector ablation + attribution to the ICL score) —
   the causal test on top of what sub-projects 2-3 establish. Needs hooks.
   Last because it's testing a claim the earlier panels first have to state.

This spec covers **sub-project 1 only**. Sub-projects 2-5 get their own
brainstorming pass when their turn comes.

## Audience and constraints

- Personal research notebook (not part of the bilingual curriculum ladder),
  same framing as `stage2c_induction_tinystories.ipynb`: built from zero,
  paper-hooked markdown per section, static and stepwise.
- Substrate: AMD Strix Halo iGPU, ROCm, unified memory — no CUDA-only
  assumptions (per `AGENTS.md`).
- In-notebook training (not an offline script like `train_stage2c.py`) —
  deliberate choice, consistent with this notebook's own stated philosophy
  ("build up piece by piece, watching each idea land") and dataset/tokenizer
  prep already being in-cell rather than in a script. Traded off against the
  CLAUDE.md "<10 min end-to-end" bar for a finished experiment: the first real
  training pass will likely exceed that, so the training cell is checkpoint-
  cached (skip retraining if `model.pt` already exists under `CACHE_DIR`) the
  same way the corpus/tokenizer cells already are — re-running the notebook
  after the first real pass stays fast.

## Design

### Model construction

`sys.path.insert(..., "notebooks/education")`; `import tiny`. Build with
`tiny.make_tiny_model(**MODEL_CONFIG)` — `tiny.py`'s parameter names
(`n_layers`, `n_heads`, `d_vocab`, `n_ctx`, `d_model`, `attn_only`,
`normalization_type`, `positional_embedding_type`) already match this
notebook's `MODEL_CONFIG` dict exactly. `tiny.device()` for device selection
(probes for a runnable ROCm kernel, falls back to CPU).

### Calibration cell (new, runs before the real training loop)

~20-50 steps at a candidate batch size; print measured tokens/sec and peak
memory. `n_ctx=2048` is 4x `stage2c`'s 512, so attention memory is roughly 16x
costlier per sequence — batch size gets picked from this evidence, not
guessed. Output of this cell: chosen batch size and the resulting step count
for the 150M-token budget.

### Training loop

AdamW (`weight_decay=0.05`, `betas=(0.9, 0.99)`), linear warmup over the first
2% of steps → cosine decay, grad-clip norm 1.0, bf16 autocast when on GPU —
same recipe as `train_stage2c.py`. Step count = `150_000_000 // (batch *
n_ctx)`. Fixed held-out eval batch, sliced from `eval_seqs` (already built in
cell 8-9), used for every eval read below — a fixed batch gives a
low-variance loss signal (diagnosed in stage2c: raw minibatch loss is too
noisy to see a phase transition in).

### Logging, every `eval_every` steps (default 100) and at the final step

On the fixed eval batch:

- scalar training-minibatch loss (reference only, expected noisy)
- scalar eval loss: `model(eval_batch, return_type="loss")` (mean) —
  low-variance signal
- **full per-position loss vector**: `model(eval_batch, return_type="loss",
  loss_per_token=True)`, mean over the batch dimension → a length-`(n_ctx-1)`
  array. This is the one new capture. It unlocks sub-project 2's loss-by-
  context-index curve, the derivative w.r.t. context index, and the ICL score
  (loss at token 500 minus loss at token 50, tracked over training) — and
  later sub-project 4's PCA — without needing to re-instrument training.
- induction score via `tiny.induction_scores(model)`, same cadence as
  `stage2c` — kept for continuity with the earlier notebook's diagnostic, not
  itself the focus of this sub-project.

Storage: trivial. A few hundred eval rows × ~2047 floats is on the order of a
few MB at most — no need for subsampling or compression.

### Metrics schema (`metrics.json`)

```json
{
  "steps": [...],
  "loss": [...],
  "eval_loss": [...],
  "per_position_loss": [[...], ...],
  "induction_score": [...],
  "corpus_tokens": ...,
  "training_tokens": ...,
  "vocab": ...,
  "n_params": ...,
  "minutes": ...,
  "seed": ...,
  "commit": ...
}
```

`per_position_loss` is a list of length-`(n_ctx-1)` rows, one per logged step,
aligned by index to `steps`.

### Checkpointing and idempotency

Saved under `CACHE_DIR` (`checkpoints/icl_pile`, already established by the
dataset-prep cells): `model.pt` (state dict + `MODEL_CONFIG`) and
`metrics.json`. The training cell checks for `model.pt` first and loads +
skips training if present — same pattern as the corpus/tokenizer cache cells
already in this notebook. A periodic in-progress checkpoint (every ~1000
steps, mirroring `train_stage2c.py`) guards against losing a long run to a
killed kernel.

### No hard gate

`train_stage2c.py` refuses to save `model.pt` if the induction score misses a
threshold. This notebook's entire point is testing whether the paper's phase
transition shows up on a Pile-like corpus — a hard gate would make a negative
result unobservable, and `stage2c`'s Panel 0 already established that a
negative result here is itself a valid, reportable finding. So: **always
save**, and report whatever the induction score / (later) ICL score turns out
to be. No pass/fail assertion.

### Out of scope for this sub-project

Attention pattern capture, prefix-matching score, QK eigenvalues, ablation —
sub-projects 3-5. They read this checkpoint later; no further changes to the
training loop are anticipated to support them.

## Testing

No new pytest coverage planned — this is a personal research notebook
following `stage2c`'s precedent (heavy training/eval logic lives in a notebook
cell, not a tested module, because there is no reusable module here — unlike
`induction_viz.py`, which sub-project 3 will likely extend and which does have
test coverage in `tests/education/`).
