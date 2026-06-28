# Stage 2dash² — Two-layer composition & induction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the faithful-scale two-layer attention-only Arabic artifact that reproduces the framework paper's composition + induction-head result: a shared induction-score helper, an offline training script with a verification gate, and a dense reference notebook that loads the checkpoint and proves the circuit by hand.

**Architecture:** Mirror the existing Stage 2dash pipeline. Pure, unit-tested induction-score helpers live in `tiny.py` so both the training gate and the notebook reuse them (DRY). `train_stage2dash2.py` (a near-copy of `train_stage2dash.py`) trains a 2-layer model offline, asserts an induction head emerged before saving, and writes per-head scores into `metrics.json`. The reference notebook loads that checkpoint (local → HF fallback, with a `FORCE_TINY` network-free path for CI) and walks the 10-section decomposition.

**Tech Stack:** Python, PyTorch (ROCm 2.5.1+rocm6.2 via `uv run --extra rocm`), TransformerLens (`HookedTransformer` via `tiny.make_tiny_model`), `tokenizers` BPE, plotly, pytest, Jupyter notebooks (verified by `notebooks/education/verify_notebooks.py`).

## Global Constraints

- **Model config (verbatim):** `n_layers=2`, `d_model=512`, `n_heads=8`, `attn_only=True`, `normalization_type=None`, `positional_embedding_type="shortformer"`. Reuse the existing 2dash `tokenizer.json` (12k unicode BPE) and `tokens.npy`.
- **GPU is headless-only.** The Strix Halo iGPU drives the display; never run GPU-saturating jobs on the live desktop. CI/tests are CPU-only (`tiny.device()` returns `cpu` when no runnable GPU). GPU runs use `HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm python ...`.
- **Default batch 32**; larger batch is headless-only (halves gradient updates → needs LR re-tuning). `--bf16` recommended on for the real run (params stay fp32 under AdamW).
- **Faithfulness caveat (must appear in the notebook, verbatim idea):** "We reproduce the paper's *results* on a model deliberately configured (LN-free + shortformer) for *exact decomposition* — not the paper's literal architecture. Faithful-scale ≠ faithful architecture."
- **Attribution (must appear in §8):** induction is specifically the **K-composition** path; Q- and V-composition are shown but are not what builds the induction head.
- **Pedagogical conventions (`feedback_pedagogical_scaffolding`):** bilingual RTL (`<div dir="rtl">`, headers `## N. عربي · English`, Arabic prose first), shape-spine `← (d1, d2, ...)` annotations on tensors, upfront limitations in prose, name-then-show, closing recap → handoff.
- **Naming:** script `train_stage2dash2.py`; checkpoint dir `checkpoints/stage2dash2/`; notebook `stage2_dash2_composition_induction_reference.ipynb`; HF repo default `yassermakram/fanous-stage2dash2-attn-only-2l`; verify stage arg `2dash2`; mock `mock_stage2_dash2`.
- **Commit style:** end commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Work on branch `stage2dash2-two-layer-composition-induction` (already created).

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `notebooks/education/tiny.py` | Add shared induction-score helpers | Modify |
| `tests/education/test_induction_score.py` | Unit tests for the helpers | Create |
| `notebooks/education/corpus.py` | Shared corpus/tokenizer helpers (extracted from train_stage2dash.py) | Create |
| `notebooks/education/train_stage2dash.py` | Import shared helpers from `corpus.py` (refactor, no behaviour change) | Modify |
| `tests/education/test_corpus.py` | Unit test for `corpus.clean` + import-smoke for both trainers | Create |
| `notebooks/education/train_stage2dash2.py` | Offline 2-layer training + verification gate | Create |
| `notebooks/education/stage2_dash2_composition_induction_reference.ipynb` | The reference notebook (10 sections) | Create |
| `notebooks/education/verify_notebooks.py` | CI mock + registration for the new notebook | Modify |
| `README.md` | Ladder-table row + roadmap checkbox | Modify |

`.gitignore` already ignores `notebooks/education/checkpoints/` — no change needed (verify in Task 6).

---

## Task 1: Shared induction-score helpers in `tiny.py`

**Files:**
- Modify: `notebooks/education/tiny.py` (append two functions near the other model helpers)
- Test: `tests/education/test_induction_score.py`

**Interfaces:**
- Produces:
  - `induction_score_from_pattern(pattern, seq_len) -> torch.Tensor` — pure. `pattern` is `(batch, n_heads, n_pos, n_pos)` post-softmax attention with `n_pos == 2*seq_len`; returns `(n_heads,)` mean induction-stripe score in `[0, 1]`.
  - `induction_scores(model, seq_len=50, n_seqs=8, seed=0) -> torch.Tensor` — runs a repeated-random-tokens forward pass through a `HookedTransformer` and returns a CPU tensor of shape `(n_layers, n_heads)`. Internally clamps `seq_len` to `model.cfg.n_ctx // 2`.

- [ ] **Step 1: Write the failing test**

Create `tests/education/test_induction_score.py`:

```python
"""Unit tests for the shared induction-score helpers in tiny.py.

The pure helper is tested against a hand-built attention pattern where the
answer is known; the model-level helper is tested for shape/range on a tiny
CPU 2-layer model (no GPU, no network, no training)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "education"))
import tiny


def test_pattern_helper_scores_perfect_induction_high_and_uniform_low():
    seq_len = 4
    n_pos = 2 * seq_len
    pattern = torch.zeros(1, 2, n_pos, n_pos)
    # Head 0: perfect induction stripe — query (seq_len+i) attends to key (i+1),
    # i.e. pattern[k, k - (seq_len - 1)] == 1 for the second-half queries.
    for k in range(seq_len - 1, n_pos):
        pattern[0, 0, k, k - (seq_len - 1)] = 1.0
    # Head 1: uniform attention (no induction).
    pattern[0, 1] = 1.0 / n_pos

    scores = tiny.induction_score_from_pattern(pattern, seq_len)

    assert scores.shape == (2,)
    assert scores[0] > 0.99
    assert scores[1] < 0.30


def test_model_helper_shape_and_range():
    model = tiny.make_tiny_model(
        n_layers=2, n_heads=4, d_vocab=64, n_ctx=32, d_model=64,
        attn_only=True, normalization_type=None,
        positional_embedding_type="shortformer",
    )
    scores = tiny.induction_scores(model, seq_len=8, n_seqs=2, seed=0)
    assert scores.shape == (2, 4)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu pytest tests/education/test_induction_score.py -v`
Expected: FAIL — `AttributeError: module 'tiny' has no attribute 'induction_score_from_pattern'`.

- [ ] **Step 3: Write minimal implementation**

Append to `notebooks/education/tiny.py`:

```python
def induction_score_from_pattern(pattern, seq_len):
    """Mean induction-stripe score per head from an attention pattern.

    `pattern` is (batch, n_heads, n_pos, n_pos) post-softmax attention over a
    repeated sequence [x_0..x_{L-1}][x_0..x_{L-1}] with n_pos == 2*L. An
    induction head at query position L+i attends to key i+1 (the token after
    the previous occurrence of the current token); that is the diagonal at
    offset (1 - L). Returns a (n_heads,) tensor in [0, 1]."""
    stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)  # (batch, n_heads, L+1)
    return stripe.mean(dim=(0, -1))


def induction_scores(model, seq_len=50, n_seqs=8, seed=0):
    """Per-(layer, head) induction score for a HookedTransformer.

    Feeds n_seqs sequences of repeated random in-vocab tokens and measures how
    strongly each head attends along the induction stripe. Returns a CPU tensor
    of shape (n_layers, n_heads). seq_len is clamped so 2*seq_len <= n_ctx."""
    import torch

    seq_len = min(seq_len, model.cfg.n_ctx // 2)
    g = torch.Generator().manual_seed(seed)
    half = torch.randint(0, model.cfg.d_vocab, (n_seqs, seq_len), generator=g)
    toks = torch.cat([half, half], dim=1).to(model.cfg.device)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, return_type=None)
    per_layer = [
        induction_score_from_pattern(cache["pattern", layer], seq_len)
        for layer in range(model.cfg.n_layers)
    ]
    return torch.stack(per_layer).detach().cpu()  # (n_layers, n_heads)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra cpu pytest tests/education/test_induction_score.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add notebooks/education/tiny.py tests/education/test_induction_score.py
git commit -m "feat(tiny): shared induction-score helpers for the gate + notebook

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Extract shared corpus/tokenizer module `corpus.py`

**Files:**
- Create: `notebooks/education/corpus.py`
- Modify: `notebooks/education/train_stage2dash.py` (import the helpers instead of defining them)
- Test: `tests/education/test_corpus.py`

**Interfaces:**
- Consumes: nothing new (pure extraction).
- Produces: module `corpus` exposing `clean(text) -> str`, `build_corpus(char_budget, cache_path) -> str`, `train_tokenizer(text, vocab_size, out_path) -> Tokenizer`, `tokenize(text, tok, cache_path) -> np.ndarray`. Both `train_stage2dash.py` and (Task 3) `train_stage2dash2.py` import these.

This is a behaviour-preserving refactor: move the four helpers verbatim into `corpus.py`, then have `train_stage2dash.py` import them. The `_AR`/`_NOISE`/`_WS` module-level regexes move with `clean`.

- [ ] **Step 1: Write the failing test**

Create `tests/education/test_corpus.py`:

```python
"""Unit test for the shared corpus helpers + import-smoke for both trainers.

clean() is pure (regex-only, no network); we test it directly. We also assert
both training scripts import the shared module so the refactor is wired up."""
import sys
from pathlib import Path

EDU = Path(__file__).resolve().parents[2] / "notebooks" / "education"
sys.path.insert(0, str(EDU))


def test_clean_strips_latin_digits_and_collapses_whitespace():
    import corpus
    # keeps Arabic letters, drops latin/digits/underscore/@, collapses whitespace
    assert corpus.clean("مرحبا   world123 @user يا") == "مرحبا  يا"


def test_both_trainers_import_shared_corpus():
    import importlib
    for mod in ("train_stage2dash", "train_stage2dash2"):
        m = importlib.import_module(mod)
        assert hasattr(m, "corpus") or hasattr(m, "build_corpus"), mod
```

Note: `test_both_trainers_import_shared_corpus` will not fully pass until Task 3 creates `train_stage2dash2.py`; in this task, run only the `clean` test (next step). The import-smoke test is included now so it is not forgotten — it goes green in Task 3.

- [ ] **Step 2: Run the clean() test to verify it fails**

Run: `uv run --extra cpu pytest tests/education/test_corpus.py::test_clean_strips_latin_digits_and_collapses_whitespace -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus'`.

- [ ] **Step 3: Create `corpus.py` and refactor `train_stage2dash.py`**

Create `notebooks/education/corpus.py` with the module docstring and the four helpers (`clean`, `build_corpus`, `train_tokenizer`, `tokenize`) plus the `_AR`/`_NOISE`/`_WS` regexes, moved **verbatim** from `train_stage2dash.py` (lines 46–145), with the needed imports (`os`, `re`, `numpy as np`; `datasets`/`tokenizers` stay as the existing lazy in-function imports):

```python
"""Shared corpus + tokenizer helpers for the offline Stage 2dash / 2dash² trainers.

Streaming/cleaning Arabic text, training a small unicode BPE, and caching the
tokenised ids. Extracted from train_stage2dash.py so both trainers reuse one copy."""
from __future__ import annotations

import os
import re

import numpy as np

_AR = re.compile(r"[^\sء-ي]")
_NOISE = re.compile(r"[a-zA-Z0-9_@]+")
_WS = re.compile(r"\s+")

# (clean, build_corpus, train_tokenizer, tokenize — moved verbatim from
#  train_stage2dash.py lines 54-145)
```

Then in `train_stage2dash.py`: delete the four moved helpers and the three regexes, and add `import corpus` near the other imports, plus `from corpus import build_corpus, train_tokenizer, tokenize` (so the existing call sites in `train()` are unchanged). Keep everything else identical.

- [ ] **Step 4: Run the clean() test + confirm 2dash trainer still imports**

Run: `uv run --extra cpu pytest tests/education/test_corpus.py::test_clean_strips_latin_digits_and_collapses_whitespace -v`
Expected: PASS.
Run: `uv run --extra cpu python -c "import sys; sys.path.insert(0,'notebooks/education'); import train_stage2dash; print('2dash import OK')"`
Expected: `2dash import OK`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/education/corpus.py notebooks/education/train_stage2dash.py tests/education/test_corpus.py
git commit -m "refactor(stage2dash): extract shared corpus/tokenizer helpers into corpus.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Offline training script `train_stage2dash2.py`

**Files:**
- Create: `notebooks/education/train_stage2dash2.py` (structure mirrors `train_stage2dash.py`)
- Test: extend `tests/education/test_induction_score.py` with a gate-logic test

**Interfaces:**
- Consumes: `tiny.make_tiny_model`, `tiny.induction_scores`, `tiny.device`, `tiny.DEFAULT_SEED`; the shared `corpus.build_corpus`, `corpus.train_tokenizer`, `corpus.tokenize`.
- Produces: `checkpoints/stage2dash2/model.pt` (state_dict + `config` with `n_layers=2`), `metrics.json` (now with `induction_scores` and `best_induction_score`). CLI flags `--bf16`, `--induction-threshold` (default 0.4), default `--hf-repo yassermakram/fanous-stage2dash2-attn-only-2l`.

- [ ] **Step 1: Write the failing test (gate logic)**

Append to `tests/education/test_induction_score.py`:

```python
def test_gate_passes_when_a_head_crosses_threshold():
    # A trained model would have a real induction head; here we just assert the
    # gate predicate the training script uses: best score >= threshold.
    scores = torch.tensor([[0.05, 0.02, 0.61, 0.10], [0.08, 0.03, 0.04, 0.07]])
    best = float(scores.max())
    threshold = 0.4
    assert best >= threshold  # gate would PASS
    # locate (layer, head) of the best — the notebook's §8 selection logic
    layer, head = [int(x) for x in divmod(int(scores.argmax()), scores.shape[1])]
    assert (layer, head) == (0, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra cpu pytest tests/education/test_induction_score.py::test_gate_passes_when_a_head_crosses_threshold -v`
Expected: PASS already (pure tensor logic) — this test pins the gate predicate so Step 3 implements it identically. If it errors on import, that is the failure to fix.

- [ ] **Step 3: Create the training script**

Create `notebooks/education/train_stage2dash2.py`. Model its structure on `train_stage2dash.py` but **import the corpus/tokenizer helpers from `corpus.py`** (do not redefine them) — i.e. start from the imports + `train()` + `main()` and add `from corpus import build_corpus, train_tokenizer, tokenize`. Then apply exactly these edits:

1. Replace the module docstring's first paragraph and "Why this config" block to describe the 2-layer model:

```python
"""Train the Stage 2dash² model: a faithful-scale, TWO-layer attention-only Arabic
transformer, reusing the Stage 2dash tokenizer + token cache.

Run-once heavy step (headless — the iGPU drives the display). The reference notebook
stage2_dash2_composition_induction_reference.ipynb loads the checkpoint and does the
fast interpretability (composition algebra + induction head).

Why this config (A Mathematical Framework for Transformer Circuits, two-layer section):
  - 2 layers, attention-only, d_model=512, n_heads=8 — paper-class scale where
    head composition forms a legible induction head.
  - normalization_type=None + positional_embedding_type="shortformer" so the two-layer
    path expansion is exact and induction is purely content-based (a principled
    deviation from the paper's LN + learned-positional attn-only-2l).
  - Reuses the Stage 2dash 12k unicode BPE tokenizer.json + tokens.npy (identical vocab
    and corpus) so the notebook's 1-layer-vs-2-layer comparison is on identical tokens.

A verification gate asserts an induction head emerged (induction score >= threshold)
before the checkpoint is saved; per-head scores are written to metrics.json.

Run (headless, gfx1151 masquerade):
  HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm python \
      notebooks/education/train_stage2dash2.py --bf16
  ... --push-hub --hf-repo <user>/fanous-stage2dash2-attn-only-2l
  ... --calibrate            # throughput projection then stop
"""
```

2. Add `from contextlib import nullcontext` to the imports.

3. Point the default corpus/token cache at the existing 2dash checkpoint dir so they are reused (do NOT re-stream the corpus). Change the `--out` default to the 2dash² dir, and read corpus/tokens/tokenizer from the 2dash dir. Replace the head of `train()` (the corpus/tokenizer/tokenize block) with:

```python
def train(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    device = tiny.device()
    print(f"[train] device={device}")

    # Reuse the Stage 2dash tokenizer + corpus + token cache (identical vocab/corpus).
    src = args.reuse_from
    char_budget = args.corpus_chars or int(args.tokens * 4.2)
    text = build_corpus(char_budget, os.path.join(src, "corpus.txt"))
    tok = train_tokenizer(text, args.vocab, os.path.join(src, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    ids = tokenize(text, tok, os.path.join(src, "tokens.npy"))
    if len(ids) > args.tokens:
        ids = ids[: args.tokens]
    print(f"[train] training on {len(ids):,} tokens, vocab={vocab} (reused from {src})")
    # copy the tokenizer into the 2dash² out dir so the checkpoint is self-contained
    import shutil
    shutil.copy(os.path.join(src, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
```

4. Change `make_tiny_model(n_layers=1, ...)` to `n_layers=2`.

5. Wrap the forward pass in a bf16 autocast context:

```python
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if (args.bf16 and device == "cuda") else nullcontext())
    ...
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (args.batch,), generator=g)
        batch = data[idx].to(device)
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_at(step)
        opt.zero_grad()
        with amp:
            loss = model(batch, return_type="loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ...
```

6. Set `"n_layers": 2` in the saved `config` dict.

7. Add the verification gate + per-head scores immediately before `torch.save(...)`:

```python
    model.eval()
    ind = tiny.induction_scores(model)          # (n_layers, n_heads)
    best = float(ind.max())
    layer, head = [int(x) for x in divmod(int(ind.argmax()), ind.shape[1])]
    print(f"[gate] best induction score {best:.3f} at layer {layer} head {head}")
    if best < args.induction_threshold:
        raise SystemExit(
            f"[gate] FAILED: best induction score {best:.3f} < {args.induction_threshold}. "
            f"No induction head formed — checkpoint NOT saved.")
    model.train()
```

8. Add `induction_scores` + `best_induction_score` to the `metrics` dict:

```python
    metrics = {
        ...
        "induction_scores": ind.tolist(),
        "best_induction_score": best,
        "induction_head": [layer, head],
    }
```

9. Add CLI flags in `main()`:

```python
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast for the forward (params stay fp32); recommended on GPU")
    p.add_argument("--induction-threshold", type=float, default=0.4,
                   help="min best induction score required to save the checkpoint")
    p.add_argument("--reuse-from", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints", "stage2dash"),
        help="dir holding the 2dash corpus.txt / tokenizer.json / tokens.npy to reuse")
```

10. Change the `--out` default to the 2dash² dir and `--hf-repo` default:

```python
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints", "stage2dash2"))
    p.add_argument("--hf-repo", default="yassermakram/fanous-stage2dash2-attn-only-2l",
                   help="HF repo id for --push-hub (must match HF_REPO in the notebook)")
```

- [ ] **Step 4: Verify the script imports and the gate test passes**

Run: `uv run --extra cpu python -c "import sys; sys.path.insert(0,'notebooks/education'); import train_stage2dash2; print('import OK')"`
Expected: `import OK`
Run: `uv run --extra cpu pytest tests/education/test_induction_score.py -v`
Expected: PASS (3 passed).

Note: a full training run is headless GPU work (`--bf16`), run by the user separately; `--calibrate` projects throughput without finishing. Do NOT run training in CI.

- [ ] **Step 5: Commit**

```bash
git add notebooks/education/train_stage2dash2.py tests/education/test_induction_score.py
git commit -m "feat(stage2dash2): offline 2-layer trainer with induction verification gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Notebook skeleton + CI registration (§1–3, runnable end-to-end)

**Files:**
- Create: `notebooks/education/stage2_dash2_composition_induction_reference.ipynb`
- Modify: `notebooks/education/verify_notebooks.py`

**Interfaces:**
- Consumes: `tiny.make_tiny_model`, `tiny.induction_scores`, `tiny.device`; the 2dash 1-layer checkpoint (for §1) and the 2dash² 2-layer checkpoint (for §3), both with a `FORCE_TINY` network-free fallback.
- Produces: globals `model` (2-layer), `model1` (2dash 1-layer), `tok`, `encode`, `id_to_str`, `VOCAB`, `EVAL_TEXT`, `FORCE_TINY` consumed by later tasks' cells.

Build the notebook as JSON with the standard nbformat 4 structure (`"nbformat": 4, "nbformat_minor": 5`, a `python3` kernelspec). Each cell below is one notebook cell, in order. Markdown cells wrap Arabic in `<div dir="rtl">`. After creating it, register it in the harness and run verify.

- [ ] **Step 1: Create the notebook with cells 0–9 below**

**Cell 0 (markdown)** — Colab badge:
```markdown
[![Open In Colab](https://colab.research.google.com/github/barmag/fanous-llm-lens/blob/main/notebooks/education/stage2_dash2_composition_induction_reference.ipynb)](https://colab.research.google.com/github/barmag/fanous-llm-lens/blob/main/notebooks/education/stage2_dash2_composition_induction_reference.ipynb)
```

**Cell 1 (markdown)** — title + prereq + faithfulness caveat (bilingual, RTL):
```markdown
<div dir="rtl">

# المرحلة ٢داش²: طبقتين انتباه — التركيب ورأس الاستقراء (على عربي حقيقي)

ده أكتف نوتبوك في المنهج. بيفترض إنك خلصت **٢ج** (حدس رأس الاستقراء على نموذج صغير) و**٢داش** (طريقة تفكيك الدائرة بإيدك).

**ملاحظة أمانة:** النموذج ده **من غير LayerNorm + shortformer** عشان التفكيك يطلع **بالظبط**؛ ده اختيار مقصود، مش نفس معمار الورقة بالحرف. بنعيد إنتاج **نتايج** الورقة (الاستقراء عن طريق التركيب)، مش المعمار الحرفي. "نفس المقياس" ≠ "نفس المعمار".

</div>

# Stage 2dash²: two attention layers — composition & the induction head (on real Arabic)

The densest notebook in the curriculum. Assumes you have done **2c** (induction intuition on a toy model) and **2dash** (the by-hand circuit-decomposition method).

**Faithfulness note:** this model is **LN-free + shortformer** so the decomposition is *exact* — a deliberate choice, **not the paper's literal architecture**. We reproduce the paper's *results* (induction via composition), not its literal model. "Faithful-scale" ≠ "faithful architecture".
```

**Cell 2 (code)** — Colab setup (mirror 2dash cell 1 exactly):
```python
# Setup: install deps + fetch the shared helper on Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformer_lens tokenizers huggingface_hub plotly
    !pip uninstall -y -q torchaudio
    !wget -q https://raw.githubusercontent.com/barmag/fanous-llm-lens/main/notebooks/education/tiny.py
import tiny
import torch
import plotly.graph_objects as go

torch.manual_seed(42)
print("device:", tiny.device())
```

**Cell 3 (markdown)** — `## ١. مقدمة: ليه طبقة واحدة مش كفاية · 1. Why one layer isn't enough` (bilingual): explain skip-trigrams can't copy a *novel* token, so a repeated novel token defeats the 1-layer model — motivating a second layer. Arabic first, English mirror.

**Cell 4 (code)** — shared loader for BOTH checkpoints + FORCE_TINY (adapt 2dash cell 5):
```python
import os

CKPT1_DIR = os.path.join(os.path.dirname(os.path.abspath(tiny.__file__)), "checkpoints", "stage2dash")
CKPT2_DIR = os.path.join(os.path.dirname(os.path.abspath(tiny.__file__)), "checkpoints", "stage2dash2")
HF_REPO1 = "yassermakram/fanous-stage2dash-attn-only-1l"
HF_REPO2 = "yassermakram/fanous-stage2dash2-attn-only-2l"

EVAL_TEXT = (
    "القطة بتاكل السمك والولد بيشرب اللبن في البيت. "
    "الجو حلو النهارده واحنا رايحين نتمشى في الشارع. "
    "المدينة كبيرة وفيها ناس كتير بتروح وتيجي كل يوم. "
    "هو قال إنه هيسافر بكرة بدري عشان الشغل المهم."
)

def _model_from_ckpt(ckpt):
    c = ckpt["config"]
    model = tiny.make_tiny_model(
        n_layers=c["n_layers"], n_heads=c["n_heads"], d_vocab=c["d_vocab"],
        n_ctx=c["n_ctx"], d_model=c["d_model"], attn_only=c["attn_only"],
        normalization_type=c["normalization_type"],
        positional_embedding_type=c["positional_embedding_type"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def _load_real(ckpt_dir, hf_repo):
    from tokenizers import Tokenizer
    tpath, mpath = os.path.join(ckpt_dir, "tokenizer.json"), os.path.join(ckpt_dir, "model.pt")
    if not (os.path.exists(tpath) and os.path.exists(mpath)):
        from huggingface_hub import hf_hub_download
        tpath = hf_hub_download(hf_repo, "tokenizer.json")
        mpath = hf_hub_download(hf_repo, "model.pt")
    ckpt = torch.load(mpath, map_location=tiny.device(), weights_only=False)
    return _model_from_ckpt(ckpt), Tokenizer.from_file(tpath)

def _build_tiny(n_layers):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([EVAL_TEXT] * 20,
                            trainers.BpeTrainer(vocab_size=200, special_tokens=["[UNK]"]))
    model = tiny.make_tiny_model(n_layers=n_layers, n_heads=4, d_vocab=tok.get_vocab_size(),
        n_ctx=64, d_model=64, attn_only=True,
        normalization_type=None, positional_embedding_type="shortformer")
    model.eval()
    return model, tok

if globals().get("FORCE_TINY"):
    model, tok = _build_tiny(n_layers=2)
    model1, _ = _build_tiny(n_layers=1)
else:
    model, tok = _load_real(CKPT2_DIR, HF_REPO2)
    model1, _ = _load_real(CKPT1_DIR, HF_REPO1)

VOCAB = tok.get_vocab_size()
id_to_str = {i: (tok.id_to_token(i) or "[?]") for i in range(VOCAB)}
def encode(text):
    return tok.encode(text).ids

print(f"2-layer: layers={model.cfg.n_layers} heads={model.cfg.n_heads} "
      f"d_model={model.cfg.d_model} n_ctx={model.cfg.n_ctx} vocab={VOCAB}")
print(f"1-layer (2dash): layers={model1.cfg.n_layers}")
```

**Cell 5 (markdown)** — short bilingual lead-in to the repeat experiment.

**Cell 6 (code)** — the 1-layer-fails demonstration. Build a repeated sequence of in-vocab tokens and compare the 1-layer vs 2-layer next-token rank for the repeated continuation:
```python
def repeat_ids(n):
    base = encode(EVAL_TEXT)[:n]
    seq = base + base
    return torch.tensor([seq]).to(tiny.device()), len(base)

ids2, L = repeat_ids(min(20, model.cfg.n_ctx // 2, model1.cfg.n_ctx // 2))
# at the last position of the first copy's continuation, does the model predict
# the token that actually repeats next?
def rank_of_next(m, ids, L):
    logits = m(ids, return_type="logits")[0]            # ← (pos, V)
    correct = ids[0, L + 1 : 2 * L]                      # the repeated continuation
    preds = logits[L : 2 * L - 1]                        # predictions at those positions
    ranks = (preds.argsort(dim=-1, descending=True) == correct[:, None]).float().argmax(dim=-1)
    return float(ranks.float().mean())

print("mean rank of the repeated token  — 1-layer:", round(rank_of_next(model1, ids2, L), 1))
print("mean rank of the repeated token  — 2-layer:", round(rank_of_next(model,  ids2, L), 1))
# lower is better; the 2-layer model should rank the repeated token far higher.
```

**Cell 7 (markdown)** — `## ٢. التفكيك لطبقتين · 2. The two-layer path expansion` (bilingual): write the logit equation expanded into direct path + per-head terms + virtual (composed) head terms. Prose + the algebra; no heavy compute.

**Cell 8 (markdown)** — `## ٣. نحمّل الموديل · 3. Load the 2-layer model` (bilingual): one line noting the model was already loaded in the setup cell (local → HF fallback, FORCE_TINY for CI), and restate the config printed above.

**Cell 9 (code)** — sanity forward pass + shapes (the shape-spine beat):
```python
ids = torch.tensor([encode(EVAL_TEXT)[: model.cfg.n_ctx]]).to(tiny.device())
logits, cache = model.run_with_cache(ids)
print("logits:", tuple(logits.shape))                 # ← (1, ctx, V)
print("layer-0 attn pattern:", tuple(cache["pattern", 0].shape))  # ← (1, heads, ctx, ctx)
print("layer-1 attn pattern:", tuple(cache["pattern", 1].shape))  # ← (1, heads, ctx, ctx)
```

- [ ] **Step 2: Register the notebook in the verify harness**

In `notebooks/education/verify_notebooks.py`, add a mock after `mock_stage2_dash`:

```python
def mock_stage2_dash2(ctx):
    # Stage 2dash²: loads two trained checkpoints (1-layer 2dash + 2-layer) and
    # decomposes the two-layer model into composition circuits + an induction head.
    # FORCE_TINY swaps in tiny network-free models so CI runs in seconds without the
    # (gitignored) checkpoints or any HF download.
    import plotly.graph_objects as go

    ctx["FORCE_TINY"] = True
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")
```

And add a run block (after the Stage 2dash block, before Stage 2a):

```python
# Stage 2dash²
if stage_arg in ("2dash2", "all"):
    if os.path.exists("stage2_dash2_composition_induction_reference.ipynb"):
        success = run_notebook("stage2_dash2_composition_induction_reference.ipynb", mock_stage2_dash2)
        if not success:
            all_passed = False
```

- [ ] **Step 3: Run the harness to verify the skeleton runs end-to-end**

Run: `cd notebooks/education && uv run --extra cpu python verify_notebooks.py 2dash2`
Expected: `Result: SUCCESS` then `🎉 REFERENCE NOTEBOOKS (2dash2) VERIFIED SUCCESSFULLY!`

- [ ] **Step 4: Commit**

```bash
git add notebooks/education/stage2_dash2_composition_induction_reference.ipynb notebooks/education/verify_notebooks.py
git commit -m "feat(stage2dash2): notebook skeleton (1-vs-2-layer, loaders) + CI registration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Composition algebra cells (§4–7)

**Files:**
- Modify: `notebooks/education/stage2_dash2_composition_induction_reference.ipynb` (append cells)

**Interfaces:**
- Consumes: `model`, `cache`, `tiny`, `torch`, `go` from Task 4 cells.
- Produces: `q_comp`, `k_comp`, `v_comp` composition-score tensors and `ov_eig` used by §8's narrative continuity.

All composition measures are weight-space (model-agnostic to language), following the paper: composition score = Frobenius norm of the composed product divided by the product of the operand norms.

- [ ] **Step 1: Append §4 (Q/K/V composition) — markdown then code**

Markdown cell: `## ٤. التركيب: Q · K · V · 4. Composition: Q / K / V` (bilingual) — define the three composition types and the Frobenius-ratio score; state that all three are real but only one builds induction (forward reference to §8).

Code cell:
```python
import torch

def fro(x):
    return float(torch.linalg.matrix_norm(x, ord="fro"))

# Layer-0 output writes W_OV0 = W_V0 @ W_O0 into the residual; layer-1 heads read it
# through their W_Q1 / W_K1 / W_V1. Composition score = ||A B|| / (||A|| ||B||).
W_OV0 = torch.stack([model.W_V[0, h] @ model.W_O[0, h] for h in range(model.cfg.n_heads)])  # (H, d, d)

def comp_scores(W_in1):  # W_in1: (H, d_model, d_head) for layer-1 Q/K/V
    out = torch.zeros(model.cfg.n_heads, model.cfg.n_heads)
    for h0 in range(model.cfg.n_heads):
        for h1 in range(model.cfg.n_heads):
            prod = W_OV0[h0] @ W_in1[h1]
            out[h0, h1] = fro(prod) / (fro(W_OV0[h0]) * fro(W_in1[h1]) + 1e-9)
    return out.detach()

q_comp = comp_scores(model.W_Q[1])
k_comp = comp_scores(model.W_K[1])
v_comp = comp_scores(model.W_V[1])
print("Q-composition (layer0 head -> layer1 head):", tuple(q_comp.shape))  # ← (H, H)
print("max K-composition:", round(float(k_comp.max()), 3),
      "at", [int(x) for x in divmod(int(k_comp.argmax()), k_comp.shape[1])])
```

- [ ] **Step 2: Append §5 (virtual attention heads) — markdown then code**

Markdown: `## ٥. الرؤوس الافتراضية · 5. Virtual attention heads` (bilingual) — a composed (layer0 head → layer1 head) pair acts as one effective "virtual head"; the composition-score matrices above are exactly the virtual-head strengths.

Code cell — heatmap of the K-composition matrix (the virtual-head map), no-op-safe under the plotly mock:
```python
fig = go.Figure(data=go.Heatmap(
    z=k_comp.numpy(),
    x=[f"L1 h{h}" for h in range(model.cfg.n_heads)],
    y=[f"L0 h{h}" for h in range(model.cfg.n_heads)],
    colorscale="Blues"))
fig.update_layout(title="K-composition / virtual-head strength", height=380)
fig.show()
```

- [ ] **Step 3: Append §6 (term importance) — markdown then code**

Markdown: `## ٦. أهمية الحدود · 6. Term importance` (bilingual) — rank the expansion's terms (direct path, each individual head, each virtual head) by contribution magnitude.

Code cell:
```python
terms = {"direct (W_E W_U)": fro(model.W_E @ model.W_U)}
for L in range(model.cfg.n_layers):
    for h in range(model.cfg.n_heads):
        ov = model.W_V[L, h] @ model.W_O[L, h]
        terms[f"L{L} h{h}"] = fro(model.W_E @ ov @ model.W_U)
top = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:8]
for name, val in top:
    print(f"  {name:>16}: {val:,.1f}")
```

- [ ] **Step 4: Append §7 (eigenvalue copying analysis) — markdown then code**

Markdown: `## ٧. تحليل القيم الذاتية للنسخ · 7. Eigenvalue copying analysis` (bilingual) — a copying OV circuit has mostly-positive real eigenvalues of `W_E^T W_OV W_U` (square in vocab space is large; use the head's `W_O W_V` content map eigenvalues as the paper's tractable proxy). Report the positive-eigenvalue fraction per head.

Code cell:
```python
def copying_fraction(L, h):
    ov = (model.W_V[L, h] @ model.W_O[L, h]).detach()      # (d_model, d_model)
    eig = torch.linalg.eigvals(ov).real
    return float((eig > 0).float().mean())

ov_eig = {(L, h): copying_fraction(L, h)
          for L in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)}
for (L, h), frac in sorted(ov_eig.items(), key=lambda kv: kv[1], reverse=True)[:6]:
    print(f"  L{L} h{h}: positive-eigenvalue fraction = {frac:.2f}")
```

- [ ] **Step 5: Run the harness**

Run: `cd notebooks/education && uv run --extra cpu python verify_notebooks.py 2dash2`
Expected: `🎉 REFERENCE NOTEBOOKS (2dash2) VERIFIED SUCCESSFULLY!`

- [ ] **Step 6: Commit**

```bash
git add notebooks/education/stage2_dash2_composition_induction_reference.ipynb
git commit -m "feat(stage2dash2): composition algebra cells (Q/K/V, virtual heads, terms, eigenvalues)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Induction synthesis + recap (§8–10) and .gitignore check

**Files:**
- Modify: `notebooks/education/stage2_dash2_composition_induction_reference.ipynb` (append cells)
- Verify (no edit expected): `.gitignore`

**Interfaces:**
- Consumes: `model`, `tiny`, `torch`, `go`, `encode`, `id_to_str`, `k_comp` from earlier cells (Tasks 4–5).
- Produces: terminal notebook (recap/handoff). No globals consumed downstream.

- [ ] **Step 1: Append §8 (induction = K-composition) — markdown then code**

Markdown: `## ٨. رأس الاستقراء = مسار الـ K-composition · 8. Induction *is* the K-composition path` (bilingual). State **plainly**: induction is built by **K-composition** — the layer-1 key reads a layer-0 previous-token head's output from the residual stream. Q- and V-composition (shown in §4) are real but are *not* what builds the induction head.

Code cell:
```python
# 1) Locate the induction head from the live induction score (same helper the
#    training gate used; metrics.json stores the training-time scores too).
ind = tiny.induction_scores(model)                       # ← (n_layers, n_heads)
iL, iH = [int(x) for x in divmod(int(ind.argmax()), ind.shape[1])]
print(f"induction head: layer {iL} head {iH}  score={float(ind.max()):.3f}")

# 2) Find the layer-0 previous-token head it K-composes with (highest K-comp into iH).
prev_head = int(k_comp[:, iH].argmax()) if iL == 1 else None
print(f"strongest layer-0 head feeding head {iH} via K-composition: {prev_head}")

# 3) Confirm that layer-0 head is a previous-token head (attends to position i-1).
if prev_head is not None:
    ids = torch.tensor([encode(EVAL_TEXT)[: model.cfg.n_ctx]]).to(tiny.device())
    _, cache = model.run_with_cache(ids)
    patt = cache["pattern", 0][0, prev_head]             # ← (ctx, ctx)
    offdiag = patt.diagonal(offset=-1).mean()
    print(f"layer-0 head {prev_head} mean attention to previous token: {float(offdiag):.2f}")
```

(Under `FORCE_TINY` the tiny random model has no trained induction head — guard the
strong claims: wrap the interpretive asserts so CI only checks the code path runs.)

Add a guard cell or inline guard:
```python
if not globals().get("FORCE_TINY"):
    assert float(ind.max()) >= 0.4, "expected a real induction head in the trained model"
    assert prev_head is not None
```

- [ ] **Step 2: Append §9 (fires on fresh Arabic) — markdown then code**

Markdown: `## ٩. بيشتغل على عربي جديد؟ · 9. Fires on fresh Arabic` (bilingual) — show the induction head attends back to the repeated token on a held-out Arabic repeat.

Code cell:
```python
fresh = "النهارده الجو حلو والشمس طالعة"
base = encode(fresh)[: min(12, model.cfg.n_ctx // 2)]
seq = torch.tensor([base + base]).to(tiny.device())
_, cache = model.run_with_cache(seq)
patt = cache["pattern", iL][0, iH]                        # ← (2L, 2L)
L = len(base)
stripe = patt.diagonal(offset=1 - L).mean()
print(f"induction-stripe attention on a fresh Arabic repeat: {float(stripe):.2f}")
fig = go.Figure(data=go.Heatmap(z=patt.detach().cpu().numpy(), colorscale="Viridis"))
fig.update_layout(title=f"induction head L{iL}h{iH} on a fresh repeat", height=420)
fig.show()
```

- [ ] **Step 3: Append §10 (recap & handoff) — markdown**

Markdown: `## ١٠. الخلاصة والخطوة الجاية · 10. Recap & handoff` (bilingual). Recap: one attention layer = bigram + skip-trigrams (2dash) and *can't* copy novel tokens; a second layer composes (specifically K-composition) a previous-token head with a copy head to build an induction head that copies novel repeats — on real Arabic. Handoff: this closes the architecture ladder; Phase 2's probing notebooks (nb05) build on this residual-stream literacy.

- [ ] **Step 4: Confirm `.gitignore` already excludes the new checkpoint dir**

Run: `git check-ignore notebooks/education/checkpoints/stage2dash2/model.pt`
Expected: prints the path (it is ignored by the existing `notebooks/education/checkpoints/` rule). If it prints nothing, add `notebooks/education/checkpoints/` to `.gitignore`.

- [ ] **Step 5: Run the full harness**

Run: `cd notebooks/education && uv run --extra cpu python verify_notebooks.py 2dash2`
Expected: `🎉 REFERENCE NOTEBOOKS (2dash2) VERIFIED SUCCESSFULLY!`
Then run the whole suite to confirm no regression:
Run: `cd notebooks/education && uv run --extra cpu python verify_notebooks.py all`
Expected: `🎉 REFERENCE NOTEBOOKS (all) VERIFIED SUCCESSFULLY!`

- [ ] **Step 6: Commit**

```bash
git add notebooks/education/stage2_dash2_composition_induction_reference.ipynb
git commit -m "feat(stage2dash2): induction-head synthesis (K-composition) + fresh-Arabic + recap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Docs — README ladder row + roadmap

**Files:**
- Modify: `README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the ladder-table row**

In `README.md`, immediately after the `| 2c | ... |` row (the table around line 123), add:

```markdown
| 2dash² | [`stage2_dash2_composition_induction`](notebooks/education/stage2_dash2_composition_induction_reference.ipynb) | **+ a faithful-scale second layer** — the full composition algebra (Q/K/V-composition, virtual heads, term importance, eigenvalue copying) culminating in an **induction head proven to be the K-composition path**, on real Arabic. The two-layer counterpart to 2dash. |
```

- [ ] **Step 2: Add the roadmap checkbox**

Under the Phase 2 "Architecture ladder, from scratch" bullet, append a sub-line:

```markdown
- [x] **Two-layer composition & induction, faithful scale.** The framework paper's two-layer attention-only result reproduced rigorously on real Arabic — composition algebra + an induction head verified to emerge (induction-score gate). *(See [`stage2_dash2_composition_induction`](notebooks/education/stage2_dash2_composition_induction_reference.ipynb); offline trainer `train_stage2dash2.py`.)*
```

- [ ] **Step 3: Verify links resolve**

Run: `grep -n "stage2_dash2_composition_induction" README.md`
Expected: two matches (table row + roadmap line).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(stage2dash2): README ladder row + roadmap entry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage:**
- Shared corpus/tokenizer module (DRY, per user pre-flight decision) → Task 2. ✓
- Faithful-scale 2-layer Arabic model + offline trainer → Task 3. ✓
- Reuse 2dash tokenizer + corpus → Task 3 Step 3 (`--reuse-from`) + Task 2 (shared helpers). ✓
- Induction verification gate + per-head scores in metrics.json → Task 3 Step 3 (gate) + Task 1 (helper). ✓
- bf16 + batch/headless guidance → Task 3 Step 3 (`--bf16`) + Global Constraints. ✓
- One dense notebook, 10 sections → Tasks 4–6 (§1–3, §4–7, §8–10). ✓
- Faithfulness caveat + K-composition attribution in the notebook → Task 4 Cell 1 + Task 6 §8. ✓
- 1-layer-vs-2-layer opening beat → Task 4 Cell 6. ✓
- FORCE_TINY CI path + guarded assertions → Task 4 Cell 4 + Task 6 §8 guard. ✓
- CI mock + registration → Task 4 Step 2. ✓
- Docs (README row + roadmap) → Task 7. ✓
- gitignore covers checkpoint → Task 6 Step 4 (verify). ✓
- Pedagogical conventions (RTL, shape-spine, recap/handoff) → applied across Tasks 4–6. ✓

**Placeholder scan:** No TBD/TODO. The only descriptive (non-code) cells are bilingual *markdown* prose cells (§1, §2, §5–§10 lead-ins), where the required content and verbatim mandated sentences (caveat, attribution) are specified — prose is written to the documented RTL convention, not code.

**Type consistency:** `induction_score_from_pattern(pattern, seq_len) -> (n_heads,)` and `induction_scores(model, ...) -> (n_layers, n_heads)` are used identically in Task 1 (def + test), Task 3 (gate: `.max()`, `argmax`/`divmod`), and Task 6 (§8: same `argmax`/`divmod` selection). The shared `corpus.{build_corpus,train_tokenizer,tokenize}` signatures (Task 2) match the call sites in both trainers (Task 2 refactor + Task 3). Checkpoint `config` keys (`n_layers`, `n_heads`, `d_model`, `d_head`, `d_vocab`, `n_ctx`, `attn_only`, `normalization_type`, `positional_embedding_type`) match `_model_from_ckpt` in both the 2dash notebook and Task 4 Cell 4. `k_comp` (Task 5) is the exact name consumed by Task 6 §8.
