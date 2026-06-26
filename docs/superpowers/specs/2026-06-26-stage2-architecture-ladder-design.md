# Stage 2: The Architecture Ladder — What Each Transformer Piece Adds, in Arabic

**Date:** 2026-06-26
**Status:** Design — awaiting implementation
**Scope:** Four new notebook pairs under `notebooks/education/`
(`stage2_a_single_block`, `stage2_b_multi_head`, `stage2_c_depth_induction`,
`stage2_d_mlp`), each `_reference` (complete) + `_experiment` (scaffolded with
placeholders); a new shared helper `notebooks/education/tiny.py`; and tests under
`tests/education/`.

## Problem / motivation

Stage 1 (the embeddings trilogy: count chars → count word transitions → learn a vector
per subword) leaves the reader with a model that has **no attention** — it is
context-blind. The reader has met embeddings but not the transformer block that makes a
transformer. Separately, the project agreed to re-converge on the 2021 *Mathematical
Framework for Transformer Circuits* (weights-first circuit analysis) rather than continue
purely observational probing. Stage 2 is the bridge: it builds a transformer **one
architectural piece at a time**, on Arabic data, so the reader watches each capability
emerge and arrives at the framework's signature circuit — the induction head — with the
vocabulary to understand it.

This is a **curriculum-first** design (see `project_audience_first.md`): the audience is a
Masri-reading Python user with no ML background. Each rung introduces exactly one new
idea, names the part before experimenting, and makes the math visible through output
before stating it formally.

## Intended concept (the spine)

**Capability-per-rung.** Each notebook adds one architectural piece and shows the one
capability it unlocks on Arabic data:

| Rung | Piece added | Capability unlocked | The one "aha" |
|---|---|---|---|
| (1c, exists) | a vector per subword | none — context-blind | dialect is linearly encoded in `W_E` |
| 2a | one attention block, one head | a token can **look at other tokens** | the head learns to point; prediction becomes context-aware |
| 2b | multiple heads | **several relations at once**, in parallel | different heads specialise on different Arabic relations |
| 2c | a second layer | **composition → induction** | depth lets one head use another head's output |
| 2d | an MLP | **per-token computation** | a token can transform itself; no information moves between positions |

Rungs **2a–2c are attention-only**, which is faithful to the 2021 framework (Elhage et
al. analyse attention-only transformers and explicitly set MLPs aside; the induction-head
capstone lives in a 2-layer attention-only model). Rung **2d then turns the MLP on** to
complete the transformer block, taught as its own variable rather than folded into the
attention rungs.

The arc is purely **architectural**: embeddings → +block (context) → +heads (parallel
relations) → +depth (composition) → +MLP (per-token computation). Data-effect stories
(e.g. BPE fracture / dialect tax) are deliberately **out of scope** here — they are
taught elsewhere and would smuggle a second variable into an architecture lesson.

## Approach

**Train each rung from scratch** on Arabic data, consistent with how 1c trains its
zero-layer model. The reader watches the new capability *emerge* when the architectural
piece is added — same data, same training recipe, one more architectural knob. This is
more honest than ablating a pretrained model and keeps everything on-substrate
(Strix Halo iGPU / Colab T4).

**Consistent substrate, math shown once.** Every rung is a TransformerLens
`HookedTransformer` so the reader learns one API and gets `run_with_cache` (free attention
extraction for the heatmaps, same idiom as nb06). The only thing that changes between
rungs is `n_layers` / `n_heads`. At rung 2a, one section computes the QK and OV products
**by hand** (raw matmuls on real Masri tokens) and shows the result equals what the cache
holds — "the heatmap you see *is* this matrix." Math is made visible once, then trusted.

**Attention-only (`attn_only=True`) for the attention ladder (2a–2c).** Folding the MLP
into those rungs would muddy "what does attention add"; staying attention-only also
matches the 2021 framework's analysis and keeps the toy models tiny. The MLP is then
introduced as its own final rung (2d, `attn_only=False`) so the curriculum teaches the
complete transformer block without conflating two variables.

### Shared helper: `notebooks/education/tiny.py`

A single self-contained module (depends only on pip-installable packages: `torch`,
`transformer_lens`, `numpy`). **Not** part of the `fanous_lens` package — nothing to
install for it to resolve.

- `make_tiny_model(n_layers, n_heads, d_model, attn_only=True, ...) -> HookedTransformer`
  — wraps a `HookedTransformerConfig` with tiny defaults; rungs 2a–2c differ only by
  `n_layers` / `n_heads` (attention-only), and 2d sets `attn_only=False` to add the MLP.
- `train(model, token_ids, ...) -> losses` — fixed, **seeded, deterministic** training
  loop that converges in <10 min on a Colab T4 / iGPU. Returns loss history (used
  internally; training curves are not a headline visual).
- `device() -> str` — generic `cuda if available else cpu`, so identical code runs on
  Colab and Strix Halo. Strix-specific `HSA_OVERRIDE_GFX_VERSION` is set in the
  environment, not in the notebook.
- `make_induction_data(...)` — synthetic repeat-sequence generator (`[A][B]…[A]→[B]`)
  over the toy vocab, for rung 2c.
- `make_natural_batches(...)` — batches natural MSA+Masri text (same HF sources as 1c:
  `wikimedia/wikipedia` ar + `arabic_tweets_dialects`) for training and for the 2c
  "does it fire in the wild" test.

The **identical, centralised training recipe** is what makes "this rung beats the one
below" a fair comparison — same data, optimizer, and seed, only architecture differs.

### Colab compatibility (hard requirement)

Every Stage 2 `_reference` and `_experiment` notebook must run top-to-bottom on a clean
Colab kernel with no missing references. The existing stage1 notebooks achieve this by
being fully self-contained (pip-install cell + HF data, no local imports). Stage 2 adds
exactly one local dependency, `tiny.py`, delivered by **wget of a single file** so no
package install is needed:

```python
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformer_lens datasets plotly
    !wget -q https://raw.githubusercontent.com/barmag/fanous-llm-lens/main/notebooks/education/tiny.py
import tiny
```

- **Local dev:** notebooks run from `notebooks/education/`, so `import tiny` resolves to
  the sibling file on any branch — no install, no path hacks.
- **Raw URL pins to `main`.** It therefore works on Colab only after the helper is merged
  to `main` — which is why `tiny.py` is the **first** branch → merge in the build order.
  Every later notebook branch wgets it from `main` while local dev uses the sibling file.
  There is no "missing reference" failure path once the helper is on `main`.
- The repo is public, so the raw URL is reachable.

## Per-rung notebook design (reference)

All `_reference` notebooks follow the validated scaffold
(`feedback_pedagogical_scaffolding.md`): RTL bilingual (`<div dir="rtl">`, headers
`## N. عربي · English`, Arabic prose first); name-the-part before experimenting; upfront
acknowledgment of toy-model limits; running shape spine (`← (dim1, dim2, ...)`); closing
recap → next-notebook handoff. Visual spine across all rungs: **RTL attention heatmaps ·
before/after next-token top-k · weight/vector geometry** (training curves de-prioritised).

### 2a — single block, single head *(capability: a token can look at others)*

- Train `n_layers=1, n_heads=1` on natural MSA+Masri via `tiny.train`.
- **By-hand beat:** for one Masri sentence, compute `QK` and `OV` with raw matmuls, then
  show the result equals the `run_with_cache` attention pattern — "the heatmap *is* this
  matrix." This is where the math becomes visible.
- Visuals: the first RTL attention heatmap; before/after next-token top-k vs the 1c
  embeddings-only model (context-blind → context-aware); OV pushing a direction in vocab
  space.

### 2b — multiple heads *(capability: several relations at once)*

- Same recipe, `n_heads=k` (small, e.g. 4).
- Visuals: small-multiples grid of per-head RTL heatmaps — the reader sees heads
  specialise on different Arabic relations (adjacency, agreement, article–noun);
  per-head OV geometry.

### 2c — second layer → induction *(capability: composition)*

- `n_layers=2`. **Synthetic-repeat task** (`[A][B]…[A]→[B]` over the toy vocab) → the
  induction stripe appears crisply; identify the prev-token head (L0) and induction head
  (L1) and show the K-composition that wires them. This is the "aha": depth lets one head
  use another head's output.
- **Then natural Masri** with the same trained model: does the learned circuit fire on
  real text, or only on the clean synthetic task? Honest report either way.
- Closer: *depth unlocks composition* — the capstone of the attention ladder. No
  fracture / data story.

### 2d — + MLP *(capability: per-token computation)*

- Take the 2-layer model and turn the MLP on (`attn_only=False`); train with the same
  recipe.
- Show the MLP does **per-position computation that moves no information between tokens** —
  contrast with attention, which is purely about moving information. Visuals: before/after
  next-token top-k vs the attention-only 2c model; the MLP's effect on a single token's
  residual (it transforms in place); per-token activations.
- Honest closer: this is where the *clean* circuit story of the 2021 framework stops —
  MLP-as-circuits is genuinely harder, and this is the natural **bridge to later
  feature / SAE work**. Completes "what a transformer block is made of" rather than
  ending a dead end.

### Experiment notebooks

Each `_experiment` is derived from its finished, approved `_reference` by hollowing out
the key cells into student `# TODO` placeholders (same relationship as stage1c). Built at
the **end** of each notebook's branch, once the reference is agreed.

## Verification

- **`tests/education/test_tiny.py`** (fast, CPU; lands with the helper): model builds at
  each rung shape; `train` reduces loss on a deterministic task; `make_induction_data`
  has the expected repeat structure.
- **`verify_notebooks.py`** extended per rung: each Stage 2 `_reference` executes
  top-to-bottom under the existing harness (CPU-small / mocked config so it is fast),
  asserting key cells produce the expected shapes — same gate the stage1 notebooks pass.
  The harness runs from `notebooks/education/` so `import tiny` resolves.
- **Per-notebook analysis test** where a rung has non-trivial logic (notably 2c's
  induction-head identification): a small unit test on that function, mirroring 1c's
  probe test.
- **Colab gate:** every `_reference` is confirmed runnable on a clean Colab kernel
  (wget setup cell) before merge.

## Build order & methodology

Iterative, one notebook (or the helper) per branch. Each unit: branch off `main` → build
→ **each passing test is its own commit** → user review and conversation → merge to
`main` only on user sign-off. Never edit `main` directly
(`feedback_git_workflow.md`). For a notebook pair: one branch covers the pair; build the
`_reference` to fully-working-and-tested first, then derive the `_experiment` with
placeholders, then merge.

1. `tiny.py` + `tests/education/test_tiny.py` *(must merge first — notebooks wget it from `main`)*
2. `stage2_a_single_block` (reference → experiment)
3. `stage2_b_multi_head` (reference → experiment)
4. `stage2_c_depth_induction` (reference → experiment)
5. `stage2_d_mlp` (reference → experiment)

## Out of scope

- BPE fracture / dialect-tax / any data-effect story (architecture lesson only; taught
  elsewhere).
- MLP *circuit analysis* (2d shows what the MLP does and contrasts it with attention, but
  reverse-engineering MLP features is deferred to later feature/SAE work — the 2021
  framework's clean story is attention-only).
- Pretrained models and ablation-based reveals (each rung is trained from scratch).
- Activation steering (the alternative fork that was not taken).
- The `fanous_lens` package API surface — `tiny.py` stays a standalone notebook helper
  for Colab portability.
