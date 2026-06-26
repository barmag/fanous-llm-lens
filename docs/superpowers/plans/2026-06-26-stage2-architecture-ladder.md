# Stage 2: Architecture Ladder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Masri/MSA curriculum that grows a transformer one architectural piece at a time — embeddings (exists) → +block → +heads → +depth/induction → +MLP — on Arabic data, culminating in the 2021-framework induction head.

**Architecture:** A single self-contained helper (`notebooks/education/tiny.py`) builds and trains tiny `HookedTransformer`s from scratch. Each rung is a Colab-runnable `_reference` notebook (complete) plus a scaffolded `_experiment` notebook. Notebooks reach the helper by `import tiny` (local sibling) / `wget` (Colab). Rungs 2a–2c are attention-only (faithful to the framework); 2d turns the MLP on.

**Tech Stack:** Python 3.11+, PyTorch, TransformerLens 3.1.0, `datasets` (HuggingFace), Plotly, pytest. `uv` for env, `ruff` for format/lint.

## Global Constraints

- **Substrate:** runs on AMD Strix Halo iGPU (ROCm) AND a clean Colab T4. Device selection generic: `cuda if available else cpu`. No CUDA-only libs.
- **Colab-runnable, no missing references:** every notebook runs top-to-bottom on a fresh Colab kernel. Setup cell: `if 'google.colab' in sys.modules:` → `!pip install -q transformer_lens datasets plotly` → `!wget -q https://raw.githubusercontent.com/barmag/fanous-llm-lens/main/notebooks/education/tiny.py` → `import tiny`.
- **`tiny.py` merges to `main` FIRST** (Task 1) — the wget URL pins to `main`, so no later notebook can run on Colab until the helper is there.
- **Attention-only for 2a–2c** (`attn_only=True`); **2d** sets `attn_only=False`. MLP circuit analysis is out of scope.
- **Architecture lesson only:** no BPE-fracture / dialect-tax / data-effect story in any rung.
- **Pedagogical scaffolding (mandatory, per `feedback_pedagogical_scaffolding.md`):** RTL bilingual (`<div dir="rtl">`, headers `## N. عربي · English`, Arabic prose first then English); name-the-part before experimenting; upfront acknowledgment of toy-model limits; running shape spine (every tensor cell prints `← (d1, d2, ...)`); closing recap → next-notebook handoff.
- **Visual spine across rungs:** RTL attention heatmaps · before/after next-token top-k · weight/vector geometry. Training curves de-prioritised.
- **Notebook data sources:** synthetic via `tiny.make_induction_data`; natural via the same HF sources 1c uses (`wikimedia/wikipedia` `20231101.ar`, `amgadhasan/arabic_tweets_dialects`).
- **Git:** never edit `main` directly. One branch per unit; each passing test is its own commit; reference built+approved before experiment; merge only on user sign-off. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Notebook hygiene:** clear outputs before commit (`jupyter nbconvert --clear-output`).

---

## File Structure

- `notebooks/education/tiny.py` — shared helper. `device()`, `make_tiny_model()`, `make_natural_batches()`, `make_induction_data()`, `train()`. Self-contained (torch, transformer_lens, numpy only).
- `notebooks/education/verify_notebooks.py` — MODIFY: neutralize `!wget`, add `notebooks/education/` to `sys.path`, add `mock_stage2_*` functions, register the new reference notebooks.
- `tests/education/test_tiny.py` — unit tests for the helper (fast, CPU).
- `tests/education/test_stage2c_induction.py` — unit test for 2c's induction-head identification logic.
- `notebooks/education/stage2_a_single_block_reference.ipynb` / `_experiment.ipynb`
- `notebooks/education/stage2_b_multi_head_reference.ipynb` / `_experiment.ipynb`
- `notebooks/education/stage2_c_depth_induction_reference.ipynb` / `_experiment.ipynb`
- `notebooks/education/stage2_d_mlp_reference.ipynb` / `_experiment.ipynb`

---

## Task 1: Shared helper `tiny.py` + tests + harness infra

**Branch:** `stage2-tiny-helper` (off `main`). **Merges to `main` first.**

**Files:**
- Create: `notebooks/education/tiny.py`
- Create: `tests/education/test_tiny.py`
- Modify: `notebooks/education/verify_notebooks.py` (harness infra for Stage 2)

**Interfaces — Produces (every later task consumes these exact signatures):**
- `device() -> str`
- `make_tiny_model(n_layers: int, n_heads: int, d_vocab: int, n_ctx: int, d_model: int = 128, attn_only: bool = True, seed: int = 42) -> HookedTransformer`
- `make_natural_batches(token_ids, n_ctx: int, batch_size: int | None = None) -> torch.Tensor`  # `[N, n_ctx]` long
- `make_induction_data(batch: int, seq_len: int, d_vocab: int, seed: int = 42) -> torch.Tensor`  # `[batch, seq_len]` long, even `seq_len`, second half repeats first
- `train(model, batches: torch.Tensor, n_epochs: int = 10, lr: float = 1e-3, seed: int = 42) -> list[float]`  # per-epoch mean loss

- [ ] **Step 1: Write the failing test** — `tests/education/test_tiny.py`

```python
"""Fast CPU tests for the Stage 2 shared helper (notebooks/education/tiny.py)."""
import sys
from pathlib import Path

import torch

# tiny.py lives beside the notebooks, not in the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "education"))
import tiny  # noqa: E402


def test_device_returns_known_string():
    assert tiny.device() in ("cuda", "cpu")


def test_make_tiny_model_attn_only_shapes():
    m = tiny.make_tiny_model(n_layers=1, n_heads=1, d_vocab=50, n_ctx=16, d_model=32)
    assert m.cfg.n_layers == 1 and m.cfg.n_heads == 1 and m.cfg.attn_only is True
    logits = m(torch.randint(0, 50, (2, 16)), return_type="logits")
    assert tuple(logits.shape) == (2, 16, 50)


def test_make_tiny_model_with_mlp():
    m = tiny.make_tiny_model(n_layers=2, n_heads=2, d_vocab=40, n_ctx=16,
                             d_model=32, attn_only=False)
    assert m.cfg.attn_only is False
    # forward still works with the MLP on
    assert m(torch.randint(0, 40, (1, 16)), return_type="loss").ndim == 0


def test_make_induction_data_second_half_repeats_first():
    data = tiny.make_induction_data(batch=4, seq_len=16, d_vocab=20, seed=0)
    assert tuple(data.shape) == (4, 16)
    half = 16 // 2
    assert torch.equal(data[:, :half], data[:, half:])
    assert int(data.min()) >= 1 and int(data.max()) < 20


def test_make_natural_batches_chunks_and_drops_remainder():
    ids = list(range(35))
    b = tiny.make_natural_batches(ids, n_ctx=16)
    assert tuple(b.shape) == (2, 16)  # 35 // 16 == 2, remainder dropped


def test_train_reduces_loss():
    m = tiny.make_tiny_model(n_layers=1, n_heads=2, d_vocab=40, n_ctx=16, d_model=32)
    batches = tiny.make_induction_data(batch=8, seq_len=16, d_vocab=40, seed=0)
    losses = tiny.train(m, batches, n_epochs=40, lr=1e-3, seed=0)
    assert len(losses) == 40 and losses[-1] < losses[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/education/test_tiny.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tiny'`.

- [ ] **Step 3: Write minimal implementation** — `notebooks/education/tiny.py`

```python
"""tiny.py — shared helper for the Stage 2 architecture-ladder notebooks.

Self-contained: depends only on torch, transformer_lens, numpy. Delivered to
Colab via wget of this single file; imported locally as a sibling module.
"""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

DEFAULT_SEED = 42


def device() -> str:
    """Generic device pick so identical code runs on Colab (CUDA) and Strix Halo."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_tiny_model(n_layers, n_heads, d_vocab, n_ctx, d_model=128,
                    attn_only=True, seed=DEFAULT_SEED):
    """Build a tiny HookedTransformer. Rungs 2a-2c keep attn_only=True; 2d flips it."""
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        d_mlp=(4 * d_model if not attn_only else None),
        attn_only=attn_only,
        act_fn=(None if attn_only else "gelu"),
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        normalization_type="LN",
        seed=seed,
        device=device(),
    )
    return HookedTransformer(cfg)


def make_natural_batches(token_ids, n_ctx, batch_size=None):
    """Chunk a 1D stream of ids into a [N, n_ctx] long tensor (drops the remainder)."""
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    n = ids.shape[0] // n_ctx
    ids = ids[: n * n_ctx].reshape(n, n_ctx)
    if batch_size is not None:
        ids = ids[:batch_size]
    return ids


def make_induction_data(batch, seq_len, d_vocab, seed=DEFAULT_SEED):
    """Sequences whose second half repeats their first half -> rewards induction.

    Returns a [batch, seq_len] long tensor; seq_len must be even; ids in [1, d_vocab).
    """
    assert seq_len % 2 == 0, "seq_len must be even"
    g = torch.Generator().manual_seed(seed)
    half = seq_len // 2
    first = torch.randint(1, d_vocab, (batch, half), generator=g)
    return torch.cat([first, first], dim=1)


def train(model, batches, n_epochs=10, lr=1e-3, seed=DEFAULT_SEED):
    """Full-batch train on a [N, n_ctx] long tensor. Returns per-epoch loss list."""
    torch.manual_seed(seed)
    batches = batches.to(model.cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = model(batches, return_type="loss")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/education/test_tiny.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add notebooks/education/tiny.py tests/education/test_tiny.py
git commit -m "feat(stage2): tiny.py helper for from-scratch toy transformers + tests"
```

- [ ] **Step 6: Harness infra — neutralize `!wget`, add notebook dir to sys.path**

In `notebooks/education/verify_notebooks.py`, inside `run_notebook`, find:

```python
        cell_source = cell_source.replace("!pip install", "pass # !pip install")
```

Add directly after it:

```python
        cell_source = cell_source.replace("!wget", "pass # !wget")
```

And at the top of `run_notebook`, before building `source_lines`, ensure `tiny` is importable when a notebook does `import tiny`:

```python
    import sys, os
    edu_dir = os.path.dirname(os.path.abspath(__file__))
    if edu_dir not in sys.path:
        sys.path.insert(0, edu_dir)
```

- [ ] **Step 7: Verify existing notebooks still pass the harness**

Run: `.venv/bin/python notebooks/education/verify_notebooks.py`
Expected: existing stage1 notebooks still report `Result: SUCCESS` (no regression).

- [ ] **Step 8: Commit**

```bash
git add notebooks/education/verify_notebooks.py
git commit -m "test(stage2): harness neutralizes !wget and resolves local tiny import"
```

**Merge gate:** `ruff format . && ruff check .`; both pytest and the harness green; user sign-off → merge `stage2-tiny-helper` to `main`.

---

## Notebook task convention (applies to Tasks 2–5)

Each notebook task builds the `_reference` to fully-working-and-tested, gets user review, THEN derives the `_experiment`. One branch per pair. The computational cells below are given verbatim (API verified against TransformerLens 3.1.0). The **markdown/prose cells are authored during the build** following the mandatory scaffolding (Global Constraints) — their required content is specified per cell; this is a content spec, not a placeholder.

**Every reference notebook's cell 0 is the standard setup cell:**

```python
# 🛠️ Setup: install + fetch the shared helper on Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformer_lens datasets plotly
    !wget -q https://raw.githubusercontent.com/barmag/fanous-llm-lens/main/notebooks/education/tiny.py
import tiny
import torch, numpy as np
import plotly.graph_objects as go
torch.manual_seed(42)
print("device:", tiny.device())
```

**Shared RTL attention-heatmap helper** (paste into the notebook where first used; keep identical across rungs):

```python
def show_attention(model, prompt_ids, str_tokens, layer=0, head=0, title=""):
    _, cache = model.run_with_cache(prompt_ids)
    pattern = cache["pattern", layer][0, head].detach().cpu().numpy()  # ← (seq, seq)
    fig = go.Figure(go.Heatmap(z=pattern, x=str_tokens, y=str_tokens, colorscale="Blues"))
    fig.update_layout(title=title, xaxis=dict(side="top"))
    fig.update_xaxes(autorange="reversed")   # RTL: first Arabic token on the right
    fig.update_yaxes(autorange="reversed")
    return fig
```

**Verify-harness mock pattern** (add one per reference notebook in `verify_notebooks.py`, then register it). The mock shrinks data + epochs and no-ops Plotly:

```python
def mock_stage2_x(ctx):
    import plotly.graph_objects as go
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")
    # Notebook reads optional knobs from globals with these defaults, so the mock
    # can shrink them: VOCAB, N_CTX, N_EPOCHS, BATCH. (See cell specs.)
    ctx["N_EPOCHS"] = 2
    ctx["BATCH"] = 4
```

Each reference notebook reads `N_EPOCHS`, `BATCH`, `VOCAB`, `N_CTX` from module globals if present, else uses its own defaults — this is what lets the harness run it in seconds. Specified in each task's cell list.

---

## Task 2: Rung 2a — single block, single head

**Branch:** `stage2a-single-block`. **Files:** `notebooks/education/stage2_a_single_block_reference.ipynb`, then `..._experiment.ipynb`; MODIFY `verify_notebooks.py`.

**Interfaces — Consumes:** `tiny.make_tiny_model`, `tiny.make_natural_batches`, `tiny.train`, `tiny.device`.

**Capability taught:** a token can look at other tokens (context). **Aha:** the heatmap *is* the QK matrix.

- [ ] **Step 1: Reference notebook — ordered cells.** Build these cells in order:

  0. **Setup cell** (standard, above).
  1. **md — Title + intro** (RTL bilingual): what 1c left us with (context-blind embeddings); what one attention block adds; upfront limit ("tiny model on little data — outputs are weak; the point is *seeing the head point*").
  2. **md — `## 1. البيانات · Data`**: we reuse the same MSA+Masri sources as 1c, train a BPE vocab, encode to ids.
  3. **code — corpus + BPE + encode.** Reuse 1c's fetch/BPE approach; expose `VOCAB = globals().get("VOCAB", 1000)`; print `← (n_ids,)`. (Copy the proven fetch/clean/BPE cell from `stage1_c_subword_reference.ipynb`; do not re-derive.)
  4. **code — build + train the 1-layer 1-head model:**
     ```python
     N_CTX = globals().get("N_CTX", 64)
     N_EPOCHS = globals().get("N_EPOCHS", 200)
     batches = tiny.make_natural_batches(all_ids, n_ctx=N_CTX)
     model = tiny.make_tiny_model(n_layers=1, n_heads=1, d_vocab=VOCAB, n_ctx=N_CTX)
     losses = tiny.train(model, batches, n_epochs=N_EPOCHS)
     print("loss:", round(losses[0], 3), "->", round(losses[-1], 3))  # ← scalar
     ```
  5. **md — `## 2. اليد · By hand`**: we will compute attention for one sentence by hand, then prove it matches the model's cache.
  6. **code — by-hand QK then prove-it-matches:**
     ```python
     prompt = "القطة بتاكل السمك"        # a short Masri sentence
     ids = torch.tensor([encode(prompt)]).to(tiny.device())   # encode from cell 3
     str_toks = [decode([i]) for i in ids[0].tolist()]
     _, cache = model.run_with_cache(ids)
     q = cache["q", 0][0, :, 0, :]      # ← (seq, d_head)
     k = cache["k", 0][0, :, 0, :]      # ← (seq, d_head)
     scores = (q @ k.T) / (q.shape[-1] ** 0.5)
     mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
     by_hand = torch.softmax(scores.masked_fill(mask, -1e9), dim=-1)
     from_cache = cache["pattern", 0][0, 0]
     print("max abs diff:", float((by_hand - from_cache).abs().max()))  # ← ~0
     ```
  7. **md — `## 3. الخريطة الحرارية · The heatmap`**: the number above is ~0, so the picture below *is* that matrix.
  8. **code — RTL heatmap** via `show_attention(model, ids, str_toks, 0, 0, "Head 0").show()`.
  9. **md — `## 4. قبل وبعد · Before / after`**: context-blind vs context-aware next token.
  10. **code — before/after top-k:** print top-5 next-token for the prompt from this model; contrast in prose with 1c's context-blind behaviour.
  11. **md — recap + handoff**: one head learns one relation; next notebook adds *several heads* for several relations at once.

- [ ] **Step 2: Add harness mock + register.** In `verify_notebooks.py` add `mock_stage2_a` (sets `VOCAB=200, N_CTX=16, N_EPOCHS=2`, no-ops Plotly) and register `run_notebook(".../stage2_a_single_block_reference.ipynb", mock_stage2_a)`.

- [ ] **Step 3: Run the harness — verify reference executes**

Run: `.venv/bin/python notebooks/education/verify_notebooks.py`
Expected: `stage2_a_single_block_reference.ipynb … Result: SUCCESS`.

- [ ] **Step 4: Commit (reference)**

```bash
jupyter nbconvert --clear-output --inplace notebooks/education/stage2_a_single_block_reference.ipynb
git add notebooks/education/stage2_a_single_block_reference.ipynb notebooks/education/verify_notebooks.py
git commit -m "feat(stage2a): single-block reference notebook + harness mock"
```

- [ ] **Step 5: USER REVIEW of the reference** (render it, converse, iterate; each fix that re-greens the harness is its own commit).

- [ ] **Step 6: Derive the experiment notebook.** Copy the reference; hollow out the *insight* cells into `# TODO` student tasks (cell 4 model/train args, cell 6 by-hand QK math, cell 8 heatmap call), keeping setup/data/prose intact. Add `mock_stage2_a` registration for it too. Re-run harness → SUCCESS.

- [ ] **Step 7: Commit (experiment) + clear outputs**

```bash
jupyter nbconvert --clear-output --inplace notebooks/education/stage2_a_single_block_experiment.ipynb
git add notebooks/education/stage2_a_single_block_experiment.ipynb notebooks/education/verify_notebooks.py
git commit -m "feat(stage2a): scaffolded single-block experiment notebook"
```

**Merge gate:** harness green for both notebooks; ruff clean; **Colab smoke-run of the reference on a clean kernel**; user sign-off → merge to `main`.

---

## Task 3: Rung 2b — multiple heads

**Branch:** `stage2b-multi-head`. **Files:** `stage2_b_multi_head_reference.ipynb` / `_experiment.ipynb`; MODIFY `verify_notebooks.py`.

**Consumes:** `tiny.*` as Task 2. **Capability:** several relations at once. **Aha:** different heads specialise.

- [ ] **Step 1: Reference notebook — ordered cells.**
  0. Setup. 1. md intro (recap 2a; "one head = one relation; what if we want several?"; upfront limit). 2–3. Data (same pattern as 2a; reuse cell). 4. **build + train `n_heads=4`:**
     ```python
     model = tiny.make_tiny_model(n_layers=1, n_heads=globals().get("N_HEADS", 4),
                                  d_vocab=VOCAB, n_ctx=N_CTX)
     losses = tiny.train(model, batches, n_epochs=N_EPOCHS)
     ```
  5. md "## 2. الرؤوس · The heads". 6. **code — small-multiples grid of per-head heatmaps:**
     ```python
     ids = torch.tensor([encode(prompt)]).to(tiny.device())
     str_toks = [decode([i]) for i in ids[0].tolist()]
     _, cache = model.run_with_cache(ids)
     from plotly.subplots import make_subplots
     H = model.cfg.n_heads
     fig = make_subplots(rows=1, cols=H, subplot_titles=[f"head {h}" for h in range(H)])
     for h in range(H):
         p = cache["pattern", 0][0, h].detach().cpu().numpy()
         fig.add_trace(go.Heatmap(z=p, x=str_toks, y=str_toks, colorscale="Blues",
                                  showscale=False), row=1, col=h + 1)
     fig.update_xaxes(autorange="reversed"); fig.update_yaxes(autorange="reversed")
     fig.show()
     ```
  7. md — interpret which head attends to what (adjacency / article–noun / agreement), in prose. 8. **code — per-head OV geometry** (optional vector view). 9. md recap + handoff ("heads work in parallel within one layer; next we stack a *second layer* and something new appears — composition").

- [ ] **Step 2: Harness mock + register** (`mock_stage2_b`: `VOCAB=200,N_CTX=16,N_EPOCHS=2,N_HEADS=4`).
- [ ] **Step 3: Run harness.** Expected `… Result: SUCCESS`.
- [ ] **Step 4: Commit (reference)** — clear outputs; message `feat(stage2b): multi-head reference notebook + harness mock`.
- [ ] **Step 5: USER REVIEW + iterate** (commits per re-green).
- [ ] **Step 6: Derive experiment** (hollow cell 4 `n_heads` + cell 6 grid loop into `# TODO`). Register mock; harness green.
- [ ] **Step 7: Commit (experiment)** — clear outputs; message `feat(stage2b): scaffolded multi-head experiment notebook`.

**Merge gate:** harness green; ruff clean; Colab smoke-run; user sign-off → merge.

---

## Task 4: Rung 2c — second layer → induction

**Branch:** `stage2c-depth-induction`. **Files:** `stage2_c_depth_induction_reference.ipynb` / `_experiment.ipynb`; `tests/education/test_stage2c_induction.py`; MODIFY `verify_notebooks.py`.

**Consumes:** `tiny.make_tiny_model`, `tiny.make_induction_data`, `tiny.make_natural_batches`, `tiny.train`. **Capability:** composition. **Aha:** depth lets one head use another head's output (prev-token head → induction head).

**Interfaces — Produces (consumed by `test_stage2c_induction.py`):**
- `induction_score(model, seq_len: int, d_vocab: int, seed: int = 0) -> torch.Tensor` — returns a `[n_layers, n_heads]` tensor; each entry is the mean attention a head pays from a repeated token back to the token that followed its first occurrence (offset `seq_len//2 - 1`). The induction head scores ≈1.

- [ ] **Step 1: Write the induction-analysis test FIRST** — `tests/education/test_stage2c_induction.py`. Mirrors `test_stage1c_probe.py`: load the analysis code cell out of the reference notebook by index and exec against a tiny model trained on synthetic data; assert the max induction score exceeds a chance threshold and the argmax head is in the second layer.

```python
import json
import sys
from pathlib import Path

import torch

EDU = Path(__file__).resolve().parents[2] / "notebooks" / "education"
sys.path.insert(0, str(EDU))
import tiny  # noqa: E402

NB = EDU / "stage2_c_depth_induction_reference.ipynb"


def _load_cell_defining(name):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for c in cells:
        src = "".join(c["source"])
        if f"def {name}" in src:
            ns = {"tiny": tiny, "torch": torch}
            exec(src, ns)
            return ns[name]
    raise AssertionError(f"no cell defines {name}")


def test_induction_score_peaks_in_second_layer():
    fn = _load_cell_defining("induction_score")
    model = tiny.make_tiny_model(n_layers=2, n_heads=2, d_vocab=20, n_ctx=32)
    data = tiny.make_induction_data(batch=64, seq_len=32, d_vocab=20, seed=0)
    tiny.train(model, data, n_epochs=200, lr=1e-3, seed=0)
    scores = fn(model, seq_len=32, d_vocab=20, seed=1)
    assert tuple(scores.shape) == (2, 2)
    best_layer = int(scores.flatten().argmax()) // 2
    assert best_layer == 1                       # induction head lives in layer 1
    assert float(scores.max()) > 0.5             # well above chance for seq_len 32
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/bin/python -m pytest tests/education/test_stage2c_induction.py -q` → FAIL (`no cell defines induction_score`, notebook absent).

- [ ] **Step 3: Build the reference notebook — ordered cells.**
  0. Setup. 1. md intro (recap 2b; pose: "what can two layers do that one cannot?"; upfront limit). 2. md "## 1. مهمة التكرار · The repeat task" — explain `[A][B]…[A]→[B]` and why a single layer cannot do it. 3. **code — synthetic data + 2-layer model + train:**
     ```python
     VOCAB = globals().get("VOCAB", 50); N_CTX = globals().get("N_CTX", 64)
     data = tiny.make_induction_data(batch=globals().get("BATCH", 256),
                                     seq_len=N_CTX, d_vocab=VOCAB, seed=0)
     model = tiny.make_tiny_model(n_layers=2, n_heads=2, d_vocab=VOCAB, n_ctx=N_CTX)
     losses = tiny.train(model, data, n_epochs=globals().get("N_EPOCHS", 400))
     print("loss:", round(losses[0], 3), "->", round(losses[-1], 3))
     ```
  4. **code — define `induction_score` (the tested function):**
     ```python
     def induction_score(model, seq_len, d_vocab, seed=0):
         data = tiny.make_induction_data(1, seq_len, d_vocab, seed=seed).to(model.cfg.device)
         _, cache = model.run_with_cache(data)
         half = seq_len // 2
         scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
         for L in range(model.cfg.n_layers):
             patt = cache["pattern", L][0]          # ← (heads, seq, seq)
             for h in range(model.cfg.n_heads):
                 # at each repeated position, attention back to first-occurrence + 1
                 idx = torch.arange(half, seq_len)
                 src = idx - half + 1
                 scores[L, h] = patt[h, idx, src].mean()
         return scores
     scores = induction_score(model, N_CTX, VOCAB, seed=1)
     print(scores)  # ← (n_layers, n_heads)
     ```
  5. md "## 2. الرأس الاستقرائي · The induction head" — read off the peak; identify prev-token head (L0) + induction head (L1). 6. **code — two RTL heatmaps** (L0 head = the prev-token diagonal-shift; L1 head = the induction stripe) via `show_attention`. 7. md — name K-composition: L1's keys read L0's output, which is how the induction head finds "the token after the previous A". 8. md "## 3. في البرية · In the wild" — handoff to natural text. 9. **code — run the *same model* on a natural Masri batch** and report whether the induction pattern persists; honest verdict either way. 10. md recap + handoff ("depth unlocks composition — the capstone of the attention ladder; next, 2d adds the MLP to complete the block").

- [ ] **Step 4: Run the unit test to verify it passes** — `.venv/bin/python -m pytest tests/education/test_stage2c_induction.py -q` → PASS.

- [ ] **Step 5: Harness mock + register** (`mock_stage2_c`: `VOCAB=20,N_CTX=16,BATCH=16,N_EPOCHS=3`). Run harness → SUCCESS.

- [ ] **Step 6: Commit (reference + test)**

```bash
jupyter nbconvert --clear-output --inplace notebooks/education/stage2_c_depth_induction_reference.ipynb
git add notebooks/education/stage2_c_depth_induction_reference.ipynb \
        tests/education/test_stage2c_induction.py notebooks/education/verify_notebooks.py
git commit -m "feat(stage2c): induction reference notebook + induction_score test"
```

- [ ] **Step 7: USER REVIEW + iterate** (commits per re-green).
- [ ] **Step 8: Derive experiment** (hollow cell 3 model args + cell 4 `induction_score` body into `# TODO`). Register mock; harness green; commit `feat(stage2c): scaffolded induction experiment notebook` (clear outputs first).

**Merge gate:** unit test + harness green; ruff clean; Colab smoke-run; user sign-off → merge.

---

## Task 5: Rung 2d — + MLP

**Branch:** `stage2d-mlp`. **Files:** `stage2_d_mlp_reference.ipynb` / `_experiment.ipynb`; MODIFY `verify_notebooks.py`.

**Consumes:** `tiny.make_tiny_model(..., attn_only=False)`, `tiny.make_natural_batches`, `tiny.train`. **Capability:** per-token computation (no token-mixing). **Aha:** the MLP transforms a token in place; it moves no information between positions.

- [ ] **Step 1: Reference notebook — ordered cells.**
  0. Setup. 1. md intro (recap: 2a-2c were attention-only and faithful to the framework; a real block also has an MLP; upfront limit + the honest "this is where the clean circuit story ends"). 2–3. Data (same pattern). 4. **build + train with MLP on:**
     ```python
     model = tiny.make_tiny_model(n_layers=2, n_heads=2, d_vocab=VOCAB, n_ctx=N_CTX,
                                  attn_only=False)
     losses = tiny.train(model, batches, n_epochs=N_EPOCHS)
     ```
  5. md "## 1. لا تحريك للمعلومات · No information moves" — the key contrast. 6. **code — show the MLP is per-position:** run with cache, take `cache["mlp_out", 0]` (`← (batch, seq, d_model)`); demonstrate that recomputing the MLP on a single position in isolation gives the same vector as in-context (because the MLP has no cross-token term), contrasting with attention which does not have this property. 7. **code — before/after next-token top-k** vs the attention-only 2c-style model (same data/seed, `attn_only=True`), showing what the MLP buys. 8. md "## 2. الجسر · The bridge" — MLP-as-features is harder; this is the lead-in to later feature/SAE work. 9. md recap + handoff (Stage 2 complete: embeddings → block → heads → depth → MLP; what Stage 3 will open).

- [ ] **Step 2: Harness mock + register** (`mock_stage2_d`: `VOCAB=200,N_CTX=16,N_EPOCHS=2`). Run harness → SUCCESS.
- [ ] **Step 3: Commit (reference)** — clear outputs; `feat(stage2d): MLP reference notebook + harness mock`.
- [ ] **Step 4: USER REVIEW + iterate** (commits per re-green).
- [ ] **Step 5: Derive experiment** (hollow cell 4 `attn_only` arg + cell 6 per-position demonstration into `# TODO`). Register mock; harness green; commit `feat(stage2d): scaffolded MLP experiment notebook` (clear outputs).

**Merge gate:** harness green; ruff clean; Colab smoke-run; user sign-off → merge. **Final:** update README roadmap + Notebooks table to list Stage 2 (separate small commit on its own branch).

---

## Self-Review

**Spec coverage:**
- Helper `tiny.py` (all 5 functions) + attention-only/MLP toggle → Task 1. ✓
- Colab wget delivery + harness neutralization + sys.path → Task 1 (Steps 6–8) + setup cell convention. ✓
- Rungs 2a/2b/2c/2d, each reference→experiment, capability + aha + visual spine → Tasks 2–5. ✓
- By-hand→prove-match (2a), per-head grid (2b), synthetic→natural + induction id (2c), MLP per-position contrast (2d) → present in each task's cells. ✓
- Tests: `test_tiny.py` (Task 1), `test_stage2c_induction.py` (Task 4), harness extension per rung. ✓
- Build order helper-first; per-notebook branch; reference-before-experiment; tests-as-commits; merge on sign-off → task headers + merge gates. ✓
- Scaffolding conventions, RTL, shape spine, recap/handoff → Global Constraints + referenced in every notebook cell list. ✓
- Out-of-scope (fracture, MLP circuit analysis) honored — no task introduces them. ✓

**Placeholder scan:** `# TODO` appears only as the *deliverable* of the experiment-derivation steps (scaffolding for students), never as unfinished plan content. Computational cells are complete and API-verified. Prose cells carry explicit per-cell content specs + the mandatory scaffolding reference — a content spec, not a gap.

**Type consistency:** `make_tiny_model(n_layers, n_heads, d_vocab, n_ctx, d_model=128, attn_only=True, seed=42)`, `make_induction_data(batch, seq_len, d_vocab, seed=42)`, `make_natural_batches(token_ids, n_ctx, batch_size=None)`, `train(model, batches, n_epochs=10, lr=1e-3, seed=42)`, `induction_score(model, seq_len, d_vocab, seed=0)` — used consistently across Task 1 producer, the tests, and Tasks 2–5 consumers. ✓
