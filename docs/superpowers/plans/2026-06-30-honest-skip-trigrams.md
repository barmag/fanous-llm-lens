# Honest Skip-Trigrams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the degenerate self-copy example in the Stage 2dash skip-trigram notebook with a category-driven, held-out-verified method, and add a BPE-vs-Unigram tokenizer comparison.

**Architecture:** Reusable analysis logic lives in a new testable module `notebooks/education/skip_trigrams.py` (unit-tested against a tiny random model, mirroring `tiny.py`'s pattern). The reference notebook becomes a tokenizer-agnostic answer key that imports those helpers; two thin experiment notebooks point it at the BPE and Unigram checkpoints. A Unigram tokenizer path is added to `corpus.py` and a retrain is run via `train_stage2dash.py --tokenizer unigram`.

**Tech Stack:** Python, PyTorch, TransformerLens, HuggingFace `tokenizers`, plotly, pytest, `uv`.

## Global Constraints

- Package manager: `uv` (`uv run`, `uv pip install`). Never add deps without asking.
- Format/lint: `ruff format` + `ruff check --fix`. Type check: `basedpyright`.
- Import alias: `import fanous_lens as fl` where the package is used.
- Tests: pytest, CPU-only by default; fast. Live under `tests/education/`.
- Notebooks: clear all outputs before commit (`jupyter nbconvert --clear-output`).
- Model is **tokenizer-welded**: Unigram comparison requires a full retrain (~62 min iGPU).
- Controlled comparison: both tokenizers use vocab=12k, NFKC normalizer, **Whitespace**
  pre-tokenizer; only `models.BPE` vs `models.Unigram` differs.
- GPU runs need the masquerade: `HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm python ...`.
- Arabic: RTL display in notebooks; bilingual (Arabic + English) markdown; label MSA vs Masri.
- Honesty guardrails: no fixed example count; report real above-noise counts (don't pad);
  featured triples must move the real next-token distribution; empty categories are findings.
- Commit message trailer (every commit):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility |
|---|---|
| `notebooks/education/skip_trigrams.py` | **New.** Pure analysis logic: head circuits, head diagnostic, composite candidate scoring, seeded + unsupervised pool generation, held-out verification. Imported by notebooks; unit-tested. |
| `tests/education/test_skip_trigrams.py` | **New.** Unit tests for `skip_trigrams.py` against a tiny random model. |
| `notebooks/education/corpus.py` | **Modify.** Add `kind="bpe"\|"unigram"` to `train_tokenizer`. |
| `tests/education/test_corpus.py` | **Modify.** Add Unigram-path test. |
| `notebooks/education/train_stage2dash.py` | **Modify.** Add `--tokenizer {bpe,unigram}`; route Unigram to `checkpoints/stage2dash_unigram/`. |
| `notebooks/education/stage2_dash_skip_trigram_reference.ipynb` | **Modify.** Replace degenerate cells with the honest method; read checkpoint dir from a variable. |
| `notebooks/education/stage2_dash_skip_trigram_bpe_experiment.ipynb` | **New.** Scaffold on the BPE checkpoint. |
| `notebooks/education/stage2_dash_skip_trigram_unigram_experiment.ipynb` | **New.** Scaffold on the Unigram checkpoint. |
| `checkpoints/stage2dash_unigram/` | **New (gitignored output).** Unigram tokenizer + model from the retrain. |

---

### Task 1: `skip_trigrams.py` — head circuits + head diagnostic

**Files:**
- Create: `notebooks/education/skip_trigrams.py`
- Test: `tests/education/test_skip_trigrams.py`

**Interfaces:**
- Consumes: a TransformerLens `HookedTransformer` (1 layer, attn-only) as built by
  `tiny.make_tiny_model(...)`; `model.W_E, W_U, W_Q, W_K, W_V, W_O` (cpu tensors ok).
- Produces:
  - `head_circuits(model, h) -> tuple[Tensor, Tensor]` returns `(QK, OV)` where
    `QK` is `(V, V)` dst×src and `OV` is `(V, V)` src×out, on CPU.
  - `head_attention_kind(pattern, *, prev_bias=0.5, bos_bias=0.5) -> str` returns one of
    `"prev_token"`, `"bos"`, `"content"` given a single head's `(seq, seq)` attention matrix.

- [ ] **Step 1: Write the failing test**

```python
# tests/education/test_skip_trigrams.py
"""Unit tests for the honest skip-trigram analysis helpers.

All tests run against a tiny random attn-only model (no checkpoint, no network),
mirroring the FORCE_TINY path in the reference notebook.
"""
import sys
from pathlib import Path

import torch

EDU = Path(__file__).resolve().parents[2] / "notebooks" / "education"
sys.path.insert(0, str(EDU))

import skip_trigrams as st  # noqa: E402
import tiny  # noqa: E402


def _tiny_model(d_vocab=40):
    return tiny.make_tiny_model(
        n_layers=1, n_heads=2, d_vocab=d_vocab, n_ctx=32, d_model=64,
        attn_only=True, normalization_type=None,
        positional_embedding_type="shortformer")


def test_head_circuits_shapes():
    model = _tiny_model(d_vocab=40)
    QK, OV = st.head_circuits(model, 0)
    assert QK.shape == (40, 40)
    assert OV.shape == (40, 40)


def test_head_attention_kind_detects_bos():
    # all attention on column 0 -> "bos"
    pattern = torch.zeros(5, 5)
    pattern[:, 0] = 1.0
    assert st.head_attention_kind(pattern) == "bos"


def test_head_attention_kind_detects_prev_token():
    # each row attends to the immediately preceding token -> "prev_token"
    pattern = torch.zeros(5, 5)
    for i in range(1, 5):
        pattern[i, i - 1] = 1.0
    pattern[0, 0] = 1.0
    assert st.head_attention_kind(pattern) == "prev_token"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/education/test_skip_trigrams.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'skip_trigrams'`

- [ ] **Step 3: Write minimal implementation**

```python
# notebooks/education/skip_trigrams.py
"""Honest skip-trigram analysis for one-layer attention-only models.

Pure, testable helpers used by the Stage 2dash skip-trigram notebooks. Mirrors the
decomposition in *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021):
each head is a skip-trigram table built from a QK circuit (which source to attend to) and an
OV circuit (what the attended source promotes). Nothing here touches the network or disk.
"""
from __future__ import annotations

import torch


def head_circuits(model, h: int):
    """(QK, OV) for head h. QK is (V,V) dst x src; OV is (V,V) src x out. CPU tensors."""
    W_E = model.W_E.detach().cpu()
    W_U = model.W_U.detach().cpu()
    W_Q = model.W_Q[0, h].detach().cpu()
    W_K = model.W_K[0, h].detach().cpu()
    W_V = model.W_V[0, h].detach().cpu()
    W_O = model.W_O[0, h].detach().cpu()
    QK = W_E @ W_Q @ W_K.T @ W_E.T          # dst x src
    OV = W_E @ W_V @ W_O @ W_U              # src x out
    return QK, OV


def head_attention_kind(pattern, *, prev_bias: float = 0.5, bos_bias: float = 0.5) -> str:
    """Classify a single head's (seq, seq) attention matrix.

    Returns "bos" (mass on position 0), "prev_token" (mass on the diagonal-1 band), or
    "content" (neither dominates -> candidate for content-based long-range skip-trigrams).
    Rows are destinations; only the causal lower triangle carries mass.
    """
    p = torch.as_tensor(pattern, dtype=torch.float32)
    n = p.shape[0]
    if n < 2:
        return "content"
    rows = range(1, n)  # row 0 can only attend to itself; skip it
    bos = sum(float(p[i, 0]) for i in rows) / len(rows)
    prev = sum(float(p[i, i - 1]) for i in rows) / len(rows)
    if bos >= bos_bias and bos >= prev:
        return "bos"
    if prev >= prev_bias:
        return "prev_token"
    return "content"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/education/test_skip_trigrams.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
ruff format notebooks/education/skip_trigrams.py tests/education/test_skip_trigrams.py
git add notebooks/education/skip_trigrams.py tests/education/test_skip_trigrams.py
git commit -m "feat(skip-trigrams): head circuits + attention-kind diagnostic

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `skip_trigrams.py` — composite scoring + candidate pools

**Files:**
- Modify: `notebooks/education/skip_trigrams.py`
- Test: `tests/education/test_skip_trigrams.py`

**Interfaces:**
- Consumes: `head_circuits` (Task 1); an `id_to_str: dict[int,str]`; optional `encode` callable
  `str -> list[int]` for seed expansion.
- Produces:
  - `candidate_pool(model, *, head, freq=2500, include_self_copy=False, top_n=100, sources=None) -> list[dict]`
    where each dict is `{"source": int, "dest": int, "output": int, "ov": float, "qk": float, "score": float}`,
    sorted by `score` desc. When `sources` is given, only triples whose source is in that set are
    considered (seeded pool); otherwise all frequent sources (unsupervised pool).
  - `seed_ids(encode, id_to_str, seed_words, freq=2500) -> list[int]` resolves seed words/tokens
    to in-vocab frequent token ids.

- [ ] **Step 1: Write the failing test**

```python
def test_candidate_pool_returns_sorted_scored_triples():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=10)
    assert 0 < len(pool) <= 10
    keys = {"source", "dest", "output", "ov", "qk", "score"}
    assert keys <= set(pool[0])
    scores = [c["score"] for c in pool]
    assert scores == sorted(scores, reverse=True)


def test_candidate_pool_excludes_self_copy_by_default():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=40)
    assert all(c["source"] != c["output"] for c in pool)


def test_seeded_pool_restricts_sources():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=40, sources={3, 7})
    assert {c["source"] for c in pool} <= {3, 7}


def test_seed_ids_resolves_in_vocab_tokens():
    model = _tiny_model(d_vocab=40)  # noqa: F841
    id_to_str = {i: f"t{i}" for i in range(40)}
    encode = lambda s: [int(s[1:])] if s.startswith("t") else []
    ids = st.seed_ids(encode, id_to_str, ["t3", "t7", "zzz"], freq=40)
    assert set(ids) == {3, 7}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/education/test_skip_trigrams.py -k "pool or seed" -v`
Expected: FAIL with `AttributeError: module 'skip_trigrams' has no attribute 'candidate_pool'`

- [ ] **Step 3: Write minimal implementation** (append to `skip_trigrams.py`)

```python
def seed_ids(encode, id_to_str, seed_words, freq: int = 2500) -> list[int]:
    """Resolve seed words/tokens to in-vocab, frequent token ids (deduped, order-stable)."""
    out, seen = [], set()
    for w in seed_words:
        for tid in encode(w):
            if tid < freq and tid not in seen:
                seen.add(tid)
                out.append(int(tid))
    return out


def candidate_pool(model, *, head: int, freq: int = 2500, include_self_copy: bool = False,
                   top_n: int = 100, sources=None) -> list[dict]:
    """Rank skip-trigram triples (source, dest, output) for one head by a composite score.

    score = OV[source, output] * QK[dest, source], over frequent tokens. For each source we
    take its single best output (off-diagonal unless include_self_copy) and the destination
    that most strongly routes attention to it. `sources` (a set of ids) makes this a seeded
    pool; None scans all frequent sources (unsupervised).
    """
    QK, OV = head_circuits(model, head)
    QK, OV = QK[:freq, :freq], OV[:freq, :freq]
    src_iter = range(freq) if sources is None else sorted(s for s in sources if s < freq)
    out = []
    for s in src_iter:
        ov_row = OV[s].clone()
        if not include_self_copy:
            ov_row[s] = float("-inf")  # forbid self-copy
        o = int(torch.argmax(ov_row))
        ov = float(ov_row[o])
        if ov == float("-inf"):
            continue
        d = int(torch.argmax(QK[:, s]))   # destination routing attention to s
        qk = float(QK[d, s])
        out.append({"source": s, "dest": d, "output": o, "ov": ov, "qk": qk,
                    "score": ov * qk})
    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:top_n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/education/test_skip_trigrams.py -k "pool or seed" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
ruff format notebooks/education/skip_trigrams.py tests/education/test_skip_trigrams.py
git add notebooks/education/skip_trigrams.py tests/education/test_skip_trigrams.py
git commit -m "feat(skip-trigrams): composite scoring + seeded/unsupervised candidate pools

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `skip_trigrams.py` — held-out verification

**Files:**
- Modify: `notebooks/education/skip_trigrams.py`
- Test: `tests/education/test_skip_trigrams.py`

**Interfaces:**
- Consumes: the model; a triple dict from `candidate_pool`.
- Produces:
  - `verify_triple(model, triple, *, n_ctx=None) -> dict` extends the triple with
    `{"p_full": float, "p_bigram": float, "lift": float, "verified": bool}`. It builds a minimal
    causal sequence `[source, <pad...>, dest]`, runs the forward pass, and at the dest position
    compares full-model `P(output)` against the bigram-only (direct-path) `P(output)`.
    `verified = lift > 0` (full model raises the output above the context-blind baseline).
  - `verify_pool(model, pool, *, top_k=20) -> list[dict]` verifies the top_k of a pool.

- [ ] **Step 1: Write the failing test**

```python
def test_verify_triple_reports_lift_and_probabilities():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=5)
    v = st.verify_triple(model, pool[0])
    for key in ("p_full", "p_bigram", "lift", "verified"):
        assert key in v
    assert 0.0 <= v["p_full"] <= 1.0
    assert 0.0 <= v["p_bigram"] <= 1.0
    assert v["lift"] == v["p_full"] - v["p_bigram"]
    assert isinstance(v["verified"], bool)


def test_verify_pool_runs_topk():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=10)
    verified = st.verify_pool(model, pool, top_k=3)
    assert len(verified) == 3
    assert all("lift" in v for v in verified)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/education/test_skip_trigrams.py -k verify -v`
Expected: FAIL with `AttributeError: module 'skip_trigrams' has no attribute 'verify_triple'`

- [ ] **Step 3: Write minimal implementation** (append to `skip_trigrams.py`)

```python
def verify_triple(model, triple: dict, *, n_ctx: int | None = None) -> dict:
    """Does the full model raise P(output) at the dest position above the bigram baseline?

    Builds [source, source, ..., dest] (source repeated to fill context so attention has a
    real earlier token to find), runs the forward pass with cache, and at the final (dest)
    position compares softmax(full logits)[output] vs softmax(direct-path logits)[output].
    The direct path is the context-blind bigram: resid_pre @ W_U + b_U.
    """
    import torch as _t

    device = next(model.parameters()).device
    ctx = n_ctx or min(8, model.cfg.n_ctx)
    s, d, o = triple["source"], triple["dest"], triple["output"]
    seq = [s] * (ctx - 1) + [d]
    ids = _t.tensor([seq], device=device)
    logits, cache = model.run_with_cache(ids)
    direct = cache["resid_pre", 0] @ model.W_U + model.b_U
    p_full = float(_t.softmax(logits[0, -1], -1)[o])
    p_bigram = float(_t.softmax(direct[0, -1], -1)[o])
    lift = p_full - p_bigram
    return {**triple, "p_full": p_full, "p_bigram": p_bigram, "lift": lift,
            "verified": lift > 0}


def verify_pool(model, pool: list[dict], *, top_k: int = 20) -> list[dict]:
    """Verify the top_k candidates of a pool on held-out forward passes."""
    return [verify_triple(model, c) for c in pool[:top_k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/education/test_skip_trigrams.py -k verify -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
ruff format notebooks/education/skip_trigrams.py tests/education/test_skip_trigrams.py
git add notebooks/education/skip_trigrams.py tests/education/test_skip_trigrams.py
git commit -m "feat(skip-trigrams): held-out verification of candidate triples

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Unigram tokenizer path in `corpus.py`

**Files:**
- Modify: `notebooks/education/corpus.py` (`train_tokenizer`, ~line 72)
- Test: `tests/education/test_corpus.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train_tokenizer(text, vocab_size, out_path, kind="bpe")` — `kind="unigram"` trains
  `models.Unigram` via `UnigramTrainer`, same NFKC normalizer + Whitespace pre-tokenizer. Default
  `kind="bpe"` preserves the existing signature/behaviour for `train_stage2dash2.py`.

- [ ] **Step 1: Write the failing test** (append to `tests/education/test_corpus.py`)

```python
def test_train_tokenizer_unigram_kind(tmp_path):
    import corpus
    text = "القطة بتاكل السمك والولد بيشرب اللبن في البيت " * 50
    out = tmp_path / "uni.json"
    tok = corpus.train_tokenizer(text, vocab_size=120, out_path=str(out), kind="unigram")
    assert out.exists()
    assert tok.get_vocab_size() <= 120
    # round-trips Arabic text to non-empty ids
    assert tok.encode("القطة في البيت").ids


def test_train_tokenizer_defaults_to_bpe(tmp_path):
    import corpus
    text = "القطة بتاكل السمك " * 50
    out = tmp_path / "bpe.json"
    tok = corpus.train_tokenizer(text, vocab_size=120, out_path=str(out))
    assert out.exists() and tok.get_vocab_size() <= 120
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/education/test_corpus.py -k tokenizer -v`
Expected: FAIL — `train_tokenizer() got an unexpected keyword argument 'kind'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `train_tokenizer` in `notebooks/education/corpus.py` with:

```python
def train_tokenizer(text: str, vocab_size: int, out_path: str, kind: str = "bpe"):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if os.path.exists(out_path):
        print(f"[tok] cache hit: {out_path}")
        return Tokenizer.from_file(out_path)

    print(f"[tok] training {vocab_size}-vocab {kind}...")
    if kind == "unigram":
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size, special_tokens=["[UNK]"], unk_token="[UNK]")
        decoder = decoders.WordPiece(prefix="")  # tokens are whitespace-pretokenized pieces
    elif kind == "bpe":
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, special_tokens=["[UNK]"])
        decoder = decoders.BPEDecoder()
    else:
        raise ValueError(f"unknown tokenizer kind: {kind!r} (expected 'bpe' or 'unigram')")

    # Controlled comparison: identical normalizer + pre-tokenizer; only the model differs.
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoder
    chunk = 1_000_000
    tok.train_from_iterator(
        (text[i : i + chunk] for i in range(0, len(text), chunk)), trainer=trainer
    )
    tok.save(out_path)
    print(f"[tok] saved {tok.get_vocab_size()} tokens -> {out_path}")
    return tok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/education/test_corpus.py -k tokenizer -v`
Expected: PASS (2 tests). Also run the full file to confirm no regression:
`uv run pytest tests/education/test_corpus.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
ruff format notebooks/education/corpus.py tests/education/test_corpus.py
git add notebooks/education/corpus.py tests/education/test_corpus.py
git commit -m "feat(corpus): add Unigram tokenizer path (controlled vs BPE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `--tokenizer` flag in `train_stage2dash.py`

**Files:**
- Modify: `notebooks/education/train_stage2dash.py` (argparse ~line 180; `train()` tokenizer call; default `--out`)
- Test: `tests/education/test_corpus.py` (import/wiring smoke — keeps tokenizer tests together)

**Interfaces:**
- Consumes: `corpus.train_tokenizer(..., kind=...)` (Task 4).
- Produces: CLI `--tokenizer {bpe,unigram}` (default `bpe`); when `unigram` and `--out` is left at
  default, output dir becomes `checkpoints/stage2dash_unigram/`. The `kind` is threaded into
  `train_tokenizer`.

- [ ] **Step 1: Write the failing test** (append to `tests/education/test_corpus.py`)

```python
def test_train_stage2dash_exposes_tokenizer_flag():
    import importlib
    m = importlib.import_module("train_stage2dash")
    parser = m.build_parser()  # parser factored out for testability
    args = parser.parse_args(["--tokenizer", "unigram"])
    assert args.tokenizer == "unigram"
    assert parser.parse_args([]).tokenizer == "bpe"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/education/test_corpus.py -k tokenizer_flag -v`
Expected: FAIL — `module 'train_stage2dash' has no attribute 'build_parser'`

- [ ] **Step 3: Write minimal implementation**

In `train_stage2dash.py`: (a) factor the argparse block into `def build_parser() -> argparse.ArgumentParser:` returning the parser (move the existing `add_argument` calls there), and have `main()`/`__main__` call `build_parser().parse_args()`. (b) Add inside `build_parser`:

```python
    p.add_argument("--tokenizer", choices=["bpe", "unigram"], default="bpe",
                   help="tokenizer algorithm; 'unigram' defaults --out to stage2dash_unigram/")
```

(c) After parsing, before `train(args)`, default the out dir for unigram:

```python
    if args.tokenizer == "unigram" and args.out == _DEFAULT_OUT:
        args.out = os.path.join(os.path.dirname(_DEFAULT_OUT), "stage2dash_unigram")
```

where `_DEFAULT_OUT` is the existing default path expression hoisted to a module constant and used as the `--out` default. (d) In `train()`, pass the kind through:

```python
    tok = train_tokenizer(text, args.vocab, os.path.join(out, "tokenizer.json"),
                          kind=args.tokenizer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/education/test_corpus.py -k tokenizer_flag -v`
Expected: PASS. Confirm wiring smoke still green:
`uv run pytest tests/education/test_corpus.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
ruff format notebooks/education/train_stage2dash.py tests/education/test_corpus.py
git add notebooks/education/train_stage2dash.py tests/education/test_corpus.py
git commit -m "feat(train): --tokenizer {bpe,unigram} flag + unigram out dir

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Upgrade the reference notebook to the honest method

**Files:**
- Modify: `notebooks/education/stage2_dash_skip_trigram_reference.ipynb`

**Interfaces:**
- Consumes: `skip_trigrams` (Tasks 1-3); existing `load_model_and_tokenizer`, `encode`,
  `id_to_str`, `EVAL_TEXT` cells.
- Produces: a tokenizer-agnostic notebook whose checkpoint dir is set by one variable
  `CKPT_DIR` (already present) so experiment notebooks override it.

This is notebook assembly; verification is via `verify_notebooks.py` (FORCE_TINY). Keep the
existing setup/load/decomposition/bigram/quantitative/recap cells. **Replace** the single `skip`
cell and **add** new cells. Each new code cell's source:

- [ ] **Step 1: Add the seed-lexicon cell** (markdown + code) after the `load` cell

```python
# Category seed lexicons (Arabic-reader curated; edit freely). Self-copy is the control.
SEED_LEXICONS = {
    "MSA fixed expressions": ["الرغم", "بالإضافة", "حين", "بسبب", "أجل"],
    "Religious / formulaic": ["الله", "صلى", "رضي", "شاء", "عليه"],
    "Definite-article / morphology": ["ال", "الذي", "التي"],
    "MSA↔Masri contrast": ["اللي", "عايز", "دلوقتي", "الذي", "يريد", "الآن"],
}
```

- [ ] **Step 2: Add the all-heads attention diagnostic cell**

```python
import skip_trigrams as st

sent = "هو قال إنه هيسافر بكرة بدري عشان الشغل المهم"
sids = encode(sent)[: model.cfg.n_ctx]
_, hcache = model.run_with_cache(torch.tensor([sids]).to(tiny.device()))
print("Per-head attention kind (positional heads can't host content skip-trigrams):")
for h in range(model.cfg.n_heads):
    patt = hcache["pattern", 0][0, h].detach().cpu()
    print(f"  head {h}: {st.head_attention_kind(patt)}")
```

- [ ] **Step 3: Add the candidate-pool + verification cell** (the core; replaces `skip`)

```python
import skip_trigrams as st

FREQ = min(2500, VOCAB)
CONTENT_HEADS = [h for h in range(model.cfg.n_heads)
                 if st.head_attention_kind(
                     hcache["pattern", 0][0, h].detach().cpu()) == "content"] or [0]

def pool_for(seed_words, head, top_n=100):
    src = set(st.seed_ids(encode, id_to_str, seed_words, FREQ)) if seed_words else None
    pool = st.candidate_pool(model, head=head, freq=FREQ, top_n=top_n, sources=src)
    return st.verify_pool(model, pool, top_k=min(20, len(pool)))

def show(label, verified, k=8):
    real = [v for v in verified if v["verified"]]
    print(f"\n{label}: {len(real)} verified of {len(verified)} checked (showing ≤{k})")
    for v in real[:k]:
        s, d, o = id_to_str.get(v['source']), id_to_str.get(v['dest']), id_to_str.get(v['output'])
        print(f"   [{s} … {d} → {o}]   lift={v['lift']:+.3f}  score={v['score']:.2f}")

h = CONTENT_HEADS[0]
for label, seeds in SEED_LEXICONS.items():
    show(label, pool_for(seeds, h))
show("Self-copy baseline (control)",
     st.verify_pool(model, st.candidate_pool(model, head=h, freq=FREQ,
                    include_self_copy=True, top_n=20)))
show("Unsupervised — what else is in here", pool_for(None, h))
```

- [ ] **Step 4: Add the skip-trigram bug cell**

```python
# The bug: a head can't jointly condition on source AND destination. Pick a strong source;
# its top-2 promoted outputs are BOTH raised regardless of which destination attends.
QK, OV = st.head_circuits(model, CONTENT_HEADS[0])
QK, OV = QK[:FREQ, :FREQ], OV[:FREQ, :FREQ]
s = int(torch.argmax(OV.max(dim=1).values))
o1, o2 = [int(i) for i in torch.topk(OV[s], 2).indices]
print(f"source {id_to_str.get(s)} promotes BOTH {id_to_str.get(o1)} and {id_to_str.get(o2)}")
print("=> any destination that attends to it raises both — no conditioning possible.")
print("This is the structural skip-trigram bug (keep…in→mind forces keep…in→bay).")
```

- [ ] **Step 5: Update the title/intro and recap markdown** to describe the honest method
      (category pools, held-out verification, the bug), and **add upfront limits** per the
      pedagogical-scaffolding convention: "we report verified counts, not a fixed number; empty
      categories are findings; positional-only heads ⇒ the bug is the headline." Keep the
      existing bilingual RTL structure.

- [ ] **Step 6: Run the notebook end-to-end on the real BPE checkpoint**

Run: `cd notebooks/education && uv run jupyter nbconvert --to notebook --execute --inplace stage2_dash_skip_trigram_reference.ipynb`
Expected: executes without error; verified triples print with lifts; freeze the observed picks
in the markdown narration.

- [ ] **Step 7: Verify the FORCE_TINY/CI path**

Run: `cd notebooks/education && uv run python -c "import verify_notebooks as v; assert v.run_notebook('stage2_dash_skip_trigram_reference.ipynb', getattr(v,'mock_force_tiny',None)) "`
If `verify_notebooks.py` lacks a FORCE_TINY mock for this notebook, add one mirroring the existing
mocks (sets `FORCE_TINY=True` in the exec context). Expected: `Result: SUCCESS`.

- [ ] **Step 8: Clear outputs and commit**

```bash
cd notebooks/education && uv run jupyter nbconvert --clear-output --inplace stage2_dash_skip_trigram_reference.ipynb
cd ../.. && git add notebooks/education/stage2_dash_skip_trigram_reference.ipynb notebooks/education/verify_notebooks.py
git commit -m "feat(notebook): honest skip-trigram method in Stage 2dash reference

Replace the degenerate OV-diagonal self-copy example with an all-heads diagnostic,
category-seeded + unsupervised candidate pools, held-out verification, and the
structural skip-trigram bug demo.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: BPE experiment scaffold notebook

**Files:**
- Create: `notebooks/education/stage2_dash_skip_trigram_bpe_experiment.ipynb`

**Interfaces:**
- Consumes: the reference notebook's cells/`skip_trigrams`; BPE checkpoint `checkpoints/stage2dash/`.
- Produces: a learner scaffold (name-then-experiment): worked setup/load cells, but the
  candidate-pool/verification/bug cells have the key lines blanked with `# TODO(you):` guidance
  and a "reveal" pointer to the reference.

- [ ] **Step 1: Create the scaffold**

Copy the reference notebook structure. Keep setup, load (`CKPT_DIR` → `checkpoints/stage2dash`),
decomposition, diagnostic. For the candidate-pool and bug cells, replace the load-bearing call
lines with `# TODO(you): call st.candidate_pool(...) then st.verify_pool(...)` plus a one-line
hint, and a markdown cell pointing to the reference as the answer key. Intro markdown states the
hypothesis ("which heads host content skip-trigrams on BPE Arabic?") and upfront limits.

- [ ] **Step 2: Verify it executes (FORCE_TINY)**

Run: `cd notebooks/education && uv run python -c "import verify_notebooks as v; assert v.run_notebook('stage2_dash_skip_trigram_bpe_experiment.ipynb', getattr(v,'mock_force_tiny',None))"`
Expected: `Result: SUCCESS` (TODO cells must be valid Python — use `pass`/placeholder values so the
scaffold runs even before the learner fills it).

- [ ] **Step 3: Clear outputs and commit**

```bash
cd notebooks/education && uv run jupyter nbconvert --clear-output --inplace stage2_dash_skip_trigram_bpe_experiment.ipynb
cd ../.. && git add notebooks/education/stage2_dash_skip_trigram_bpe_experiment.ipynb
git commit -m "feat(notebook): BPE skip-trigram experiment scaffold

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Run the Unigram retrain (offline procedure)

**Files:**
- Output: `checkpoints/stage2dash_unigram/` (gitignored).

This is an operational task (no TDD; ~62 min on the iGPU). It depends on Tasks 4-5.

- [ ] **Step 1: Calibrate smoke (short run) to confirm wiring before the full retrain**

Run:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm python \
  notebooks/education/train_stage2dash.py --tokenizer unigram --calibrate
```
Expected: builds a Unigram `tokenizer.json`, runs a short calibration, prints throughput — no error.

- [ ] **Step 2: Full retrain (same budget/seed as the BPE model)**

Run:
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm python \
  notebooks/education/train_stage2dash.py --tokenizer unigram --tokens 500_000_000
```
Expected: writes `checkpoints/stage2dash_unigram/{tokenizer.json,model.pt,metrics.json}`;
`metrics.json` shows vocab=12000 and a final loss in a sane range (compare to BPE's 4.96).

- [ ] **Step 3: Sanity-check the checkpoint loads and decomposes**

Run:
```bash
cd notebooks/education && uv run python -c "
import torch, tiny, skip_trigrams as st
from tokenizers import Tokenizer
ck = torch.load('checkpoints/stage2dash_unigram/model.pt', map_location='cpu', weights_only=False)
c = ck['config']; print('vocab', c['d_vocab'], 'heads', c['n_heads'])
"`
```
Expected: prints `vocab 12000 heads 8` (or the trained config) with no error.

- [ ] **Step 4: Record the metrics** (no code commit; checkpoint is gitignored)

Note the Unigram `metrics.json` loss/vocab in the Task 9 notebook narration so the comparison is
honest about any train-loss gap between tokenizers.

---

### Task 9: Unigram experiment scaffold + end-to-end verification

**Files:**
- Create: `notebooks/education/stage2_dash_skip_trigram_unigram_experiment.ipynb`

**Interfaces:**
- Consumes: Unigram checkpoint (Task 8); `skip_trigrams`; reference cells.
- Produces: the Unigram scaffold (same shape as Task 7) pointing `CKPT_DIR` at
  `checkpoints/stage2dash_unigram` and `HF_REPO` left unset/local-first.

- [ ] **Step 1: Create the scaffold** mirroring Task 7 but with `CKPT_DIR = ".../stage2dash_unigram"`.
      Intro markdown states the comparison hypothesis ("does Unigram segmentation make Arabic
      skip-trigrams more legible than BPE?") and the honest caveat that the two models have
      different train losses (cite Task 8 metrics).

- [ ] **Step 2: Run end-to-end on the Unigram checkpoint**

Run: `cd notebooks/education && uv run jupyter nbconvert --to notebook --execute --inplace stage2_dash_skip_trigram_unigram_experiment.ipynb`
Expected: executes; verified triples print. Eyeball BPE vs Unigram legibility.

- [ ] **Step 3: Verify FORCE_TINY path + full education test suite**

Run:
```bash
cd notebooks/education && uv run python -c "import verify_notebooks as v; assert v.run_notebook('stage2_dash_skip_trigram_unigram_experiment.ipynb', getattr(v,'mock_force_tiny',None))"
cd ../.. && uv run pytest tests/education/ -v
```
Expected: `Result: SUCCESS`; all education tests PASS.

- [ ] **Step 4: Clear outputs and commit**

```bash
cd notebooks/education && uv run jupyter nbconvert --clear-output --inplace stage2_dash_skip_trigram_unigram_experiment.ipynb
cd ../.. && git add notebooks/education/stage2_dash_skip_trigram_unigram_experiment.ipynb
git commit -m "feat(notebook): Unigram skip-trigram experiment + BPE/Unigram comparison

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Honest method replacing degenerate example → Tasks 1-3 (logic) + Task 6 (notebook). ✓
- Arabic categories + hybrid bucketing (seeded + unsupervised) → Task 2 (`candidate_pool` with
  `sources`) + Task 6 (SEED_LEXICONS + unsupervised cell). ✓
- ~100/category pool with noise cutoff → `top_n=100` + verified-count reporting in `show()`. ✓
  (Score-distribution plot is folded into Task 6 Step 3's reporting; if a visual cutoff plot is
  wanted it is an additive cell, not a new task.)
- Held-out verification → Task 3 + used in Task 6. ✓
- Skip-trigram bug (structural) → Task 6 Step 4. ✓
- BPE vs Unigram, controlled (12k, NFKC, Whitespace) → Tasks 4-5, 8-9. ✓
- Reference + 2 experiment notebooks → Tasks 6, 7, 9. ✓
- Local-first, no HF push → Task 9 Step 1. ✓
- Honesty guardrails → Task 6 Step 5 (limits) + `show()` verified-count reporting. ✓

**Placeholder scan:** notebook tasks (6,7,9) describe cell sources with real code; the `# TODO(you)`
markers in Task 7/9 are intentional *learner* scaffolding, not plan placeholders. No "TBD".

**Type consistency:** triple dict keys (`source,dest,output,ov,qk,score`) are produced in Task 2 and
extended (`p_full,p_bigram,lift,verified`) in Task 3; consumed with those exact keys in Task 6.
`head_circuits`/`head_attention_kind`/`candidate_pool`/`seed_ids`/`verify_triple`/`verify_pool`
names match across tasks. ✓

**Note on the linguistic (pretty) bug pair:** the spec lists it as *attempted*; it is intentionally
not a checkbox step (can't guarantee it exists). If found during Task 6 Step 6, add it as an extra
markdown+code cell — additive, not gating.
