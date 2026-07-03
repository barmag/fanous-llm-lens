# Stage 2c Induction Visualization Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken node-link visualization in `notebooks/education/stage2c_induction_tinystories.ipynb` with a 5-panel stepwise explanation of the induction circuit, mapped cell-by-cell to Elhage et al. 2021 and Olsson et al. 2022, plus an interactive sandbox.

**Architecture:** Testable pure helpers (`attn_heatmap`, `pick_example`, `diag_score`, `prefix_matching_score`) live in a new module `notebooks/education/induction_viz.py` (same pattern as `tiny.py`/`corpus.py`); the notebook imports them. The notebook restructure is done in one deterministic `nbformat` script so cell ordering is exact. All figures use real Plotly traces (never annotation-only figures — that was the out-of-bounds bug).

**Tech Stack:** Python, numpy, plotly, nbformat, pytest. No new dependencies (all already in the env).

**Spec:** `docs/superpowers/specs/2026-07-03-stage2c-induction-viz-design.md`

## Global Constraints

- No new external dependencies.
- `ruff format` + `ruff check --fix` on any `.py` file before committing.
- Notebook outputs cleared before commit (`jupyter nbconvert --clear-output --inplace`).
- Tests are CPU-only and fast (no model, no GPU) — injected fakes instead.
- Commit messages name the result, not the change.
- Work happens on branch `stage2c-induction-viz` (already created, spec committed).
- The checkpoint dir `notebooks/education/checkpoints/induction_tiny/` already exists with tokenizer + model; executing the notebook continues training 2000 steps (cell-5 always trains) — this is expected and acceptable.
- Every code cell that produces a tensor prints its shape with a `← (dim1, dim2, ...)` annotation (project "shape spine" convention).

---

### Task 1: `induction_viz.py` helpers (TDD)

**Files:**
- Create: `notebooks/education/induction_viz.py`
- Test: `tests/education/test_induction_viz.py`

**Interfaces:**
- Produces (used by Task 2's notebook cells):
  - `attn_heatmap(pattern, tokens, highlight=None, title="") -> plotly.graph_objects.Figure`
  - `find_induction_anchor(ids: list[int]) -> tuple[int, int] | None`
  - `pick_example(candidates, encode, decode_token, topk5, top_n=3) -> Example` (dataclass, fields below)
  - `diag_score(pattern) -> float`
  - `prefix_matching_score(pattern, ids) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/education/test_induction_viz.py`:

```python
"""Unit tests for the stage2c induction visualization helpers.

Everything is tested with numpy arrays and injected fakes — no model,
no tokenizer, no GPU."""

import sys
from pathlib import Path

import numpy as np
import pytest

EDU = Path(__file__).resolve().parents[2] / "notebooks" / "education"
sys.path.insert(0, str(EDU))

from induction_viz import (  # noqa: E402
    attn_heatmap,
    diag_score,
    find_induction_anchor,
    pick_example,
    prefix_matching_score,
)


# ---------------------------------------------------------------- attn_heatmap
def test_attn_heatmap_has_one_real_trace_and_token_ticks():
    pat = np.eye(3)
    tokens = ["Tom", "and", "Tom"]  # duplicates must survive as distinct ticks
    fig = attn_heatmap(pat, tokens)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert list(fig.layout.xaxis.tickvals) == [0, 1, 2]
    assert list(fig.layout.xaxis.ticktext) == tokens
    assert list(fig.layout.yaxis.ticktext) == tokens


def test_attn_heatmap_highlight_draws_one_shape_per_cell():
    fig = attn_heatmap(np.eye(4), list("abcd"), highlight=[(1, 0), (3, 2)])
    assert len(fig.layout.shapes) == 2


def test_attn_heatmap_rejects_token_length_mismatch():
    with pytest.raises(ValueError):
        attn_heatmap(np.eye(3), ["a", "b"])


# ------------------------------------------------------- find_induction_anchor
def test_anchor_finds_first_earlier_occurrence():
    # ids: A B C A  -> last token A at pos 3, first A at pos 0, B1 at pos 1
    assert find_induction_anchor([5, 6, 7, 5]) == (0, 1)


def test_anchor_none_when_last_token_is_new():
    assert find_induction_anchor([5, 6, 7, 8]) is None


def test_anchor_none_when_follower_would_be_the_query_itself():
    # ids: A A -> the token after the first A IS the last position; no target
    assert find_induction_anchor([5, 5]) is None


# ----------------------------------------------------------------- pick_example
def _fake_encode(vocab):
    def encode(text):
        return [vocab[w] for w in text.split()]

    return encode


def _fake_decode(rev):
    def decode_token(tid):
        return rev[tid]

    return decode_token


def test_pick_example_returns_first_passing_candidate():
    vocab = {"Tom": 1, "and": 2, "Lily": 3, "ran": 4}
    rev = {v: k for k, v in vocab.items()}

    def topk5(ids):
        return [(3, 0.9), (4, 0.05), (2, 0.02), (1, 0.01), (0, 0.01)]

    ex = pick_example(
        ["ran ran ran ran", "Tom and Lily ran Tom and"],
        _fake_encode(vocab),
        _fake_decode(rev),
        topk5,
    )
    # candidate 1: anchor (0,1), target 'ran' (id 4) IS in top-3 -> accepted
    # first, so the picker must return it (first-match semantics).
    assert ex.prompt == "ran ran ran ran"
    assert ex.query_pos == len(ex.ids) - 1
    assert ex.ids[ex.key_pos] == ex.target_id
    assert ex.target_str == rev[ex.target_id]
    assert len(ex.topk) == 5


def test_pick_example_skips_candidate_whose_target_misses_top3():
    vocab = {"Tom": 1, "and": 2, "Lily": 3, "ran": 4, "sat": 5, "dog": 6}
    rev = {v: k for k, v in vocab.items()}

    def topk5(ids):
        # model always predicts Lily strongly; everything else is noise
        return [(3, 0.9), (6, 0.04), (5, 0.03), (2, 0.02), (1, 0.01)]

    ex = pick_example(
        ["Tom ran sat Tom", "Tom and Lily ran Tom and"],
        _fake_encode(vocab),
        _fake_decode(rev),
        topk5,
    )
    # candidate 1 anchor -> target 'ran' (id 4), not in top-3 -> rejected
    # candidate 2 anchor -> target 'Lily' (id 3), top-1 -> accepted
    assert ex.prompt == "Tom and Lily ran Tom and"
    assert ex.target_str == "Lily"


def test_pick_example_raises_with_all_rejections_named():
    vocab = {"a": 1, "b": 2, "c": 3}
    rev = {v: k for k, v in vocab.items()}

    def topk5(ids):
        return [(3, 0.9), (3, 0.05), (3, 0.02), (3, 0.01), (3, 0.01)]

    with pytest.raises(ValueError, match="a b c"):
        pick_example(["a b c"], _fake_encode(vocab), _fake_decode(rev), topk5)


# ------------------------------------------------------------------ diag_score
def test_diag_score_is_one_for_perfect_prev_token_head():
    S = 5
    pat = np.zeros((S, S))
    for i in range(1, S):
        pat[i, i - 1] = 1.0
    assert diag_score(pat) == pytest.approx(1.0)


# ------------------------------------------------------ prefix_matching_score
def test_prefix_matching_score_reads_the_stripe():
    # ids: A B A B  -> at t=2 (2nd A), most recent earlier A is s=0, expect
    # attention on s+1=1; at t=3 (2nd B), earlier B at s=1, expect s+1=2.
    ids = [7, 8, 7, 8]
    pat = np.zeros((4, 4))
    pat[2, 1] = 1.0
    pat[3, 2] = 1.0
    assert prefix_matching_score(pat, ids) == pytest.approx(1.0)


def test_prefix_matching_score_zero_when_no_repeats():
    assert prefix_matching_score(np.eye(4), [1, 2, 3, 4]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/education/test_induction_viz.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'induction_viz'`

- [ ] **Step 3: Write the implementation**

Create `notebooks/education/induction_viz.py`:

```python
"""Visualization + example-picking helpers for stage2c_induction_tinystories.

Pure functions over numpy arrays and injected callables, so everything here
is testable on CPU with fakes — no model, no tokenizer, no GPU.

Paper anchors:
- attn_heatmap renders the attention pattern A = softmax(q·k/sqrt(d)) that
  Elhage et al. 2021 call the QK circuit's output ("where to look").
- prefix_matching_score is the Olsson et al. 2022 prefix-matching criterion:
  a head is induction-like when, at a repeated token, it attends to the token
  AFTER the previous occurrence.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

HIGHLIGHT = "#e65100"


def attn_heatmap(pattern, tokens, highlight=None, title=""):
    """One attention pattern as a heatmap with decoded-token axis labels.

    pattern: [S, S] array-like, rows = query (destination), cols = key (source).
    tokens: S decoded token strings (duplicates fine — ticks are positional).
    highlight: optional list of (query, key) cells to outline.
    """
    pattern = np.asarray(pattern)
    S = pattern.shape[0]
    if pattern.shape != (S, S) or len(tokens) != S:
        raise ValueError(
            f"pattern {pattern.shape} needs matching tokens, got {len(tokens)}"
        )
    hover = [
        [
            f"query {tokens[q]!r} (pos {q}) ← key {tokens[k]!r} (pos {k}): "
            f"{pattern[q, k]:.2f}"
            for k in range(S)
        ]
        for q in range(S)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=pattern,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            showscale=False,
            text=hover,
            hoverinfo="text",
        )
    )
    idx = list(range(S))
    tickfont = dict(size=9, family="monospace")
    fig.update_xaxes(
        tickvals=idx, ticktext=list(tokens), tickangle=45, tickfont=tickfont,
        title_text="key (source) — the token being looked AT",
    )
    fig.update_yaxes(
        tickvals=idx, ticktext=list(tokens), autorange="reversed",
        tickfont=tickfont,
        title_text="query (destination) — the token doing the looking",
    )
    for q, k in highlight or []:
        fig.add_shape(
            type="rect",
            x0=k - 0.5, x1=k + 0.5, y0=q - 0.5, y1=q + 0.5,
            line=dict(color=HIGHLIGHT, width=2),
        )
    side = max(420, 24 * S + 170)
    fig.update_layout(
        title=title, width=side, height=side,
        margin=dict(l=90, r=30, t=60, b=90),
    )
    return fig


@dataclass
class Example:
    """The running example carried through panels 1–4."""

    prompt: str
    ids: list
    tokens: list
    query_pos: int  # position of the last token (2nd occurrence of A)
    key_pos: int  # position of B1 = the token after the 1st occurrence of A
    target_id: int
    target_str: str
    topk: list  # [(token_str, prob)] top-5 at query_pos


def find_induction_anchor(ids):
    """Find the induction anchor for a prompt ending in a repeated token.

    Returns (first_pos, follower_pos) for the FIRST earlier occurrence of the
    final token, or None if the final token never appeared before (or its
    follower would be the final position itself, leaving nothing to predict).
    """
    last = ids[-1]
    for s in range(len(ids) - 1):
        if ids[s] == last and s + 1 < len(ids) - 1:
            return s, s + 1
    return None


def pick_example(candidates, encode, decode_token, topk5, top_n=3):
    """Pick the first candidate prompt the model actually completes.

    candidates: prompt strings ending on the 2nd occurrence of some token A.
    encode: str -> list[int].  decode_token: int -> str.
    topk5: list[int] -> [(token_id, prob)] descending, at least 5 entries.
    Accepts the first prompt whose induction target lands in the model's
    top-`top_n`. Raises ValueError naming every rejected prompt otherwise —
    never silently falls back.
    """
    rejected = []
    for prompt in candidates:
        ids = list(encode(prompt))
        anchor = find_induction_anchor(ids)
        if anchor is None:
            rejected.append((prompt, "last token never appears earlier"))
            continue
        s, follower = anchor
        target_id = ids[follower]
        preds = topk5(ids)
        if target_id not in [tid for tid, _ in preds[:top_n]]:
            rejected.append(
                (prompt, f"target {decode_token(target_id)!r} not in top-{top_n}")
            )
            continue
        return Example(
            prompt=prompt,
            ids=ids,
            tokens=[decode_token(t) for t in ids],
            query_pos=len(ids) - 1,
            key_pos=follower,
            target_id=target_id,
            target_str=decode_token(target_id),
            topk=[(decode_token(t), p) for t, p in preds],
        )
    lines = "\n".join(f"  {p!r}: {why}" for p, why in rejected)
    raise ValueError(f"no candidate prompt passed the picker:\n{lines}")


def diag_score(pattern):
    """Prev-token strength: mean attention from position i to i-1."""
    p = np.asarray(pattern)
    S = p.shape[0]
    return float(sum(p[i, i - 1] for i in range(1, S)) / max(S - 1, 1))


def prefix_matching_score(pattern, ids):
    """Olsson et al. prefix-matching score on a real token sequence.

    For every position t whose token appeared earlier (most recent earlier
    occurrence s, with s+1 < t), average the attention pattern[t, s+1].
    Returns 0.0 when the sequence has no usable repeats.
    """
    p = np.asarray(pattern)
    tot, cnt = 0.0, 0
    for t in range(len(ids)):
        prev = [s for s in range(t) if ids[s] == ids[t] and s + 1 < t]
        if not prev:
            continue
        s = prev[-1]
        tot += float(p[t, s + 1])
        cnt += 1
    return tot / cnt if cnt else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/education/test_induction_viz.py -v`
Expected: all 11 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens
uv run ruff format notebooks/education/induction_viz.py tests/education/test_induction_viz.py
uv run ruff check --fix notebooks/education/induction_viz.py tests/education/test_induction_viz.py
uv run pytest tests/education/test_induction_viz.py -q
git add notebooks/education/induction_viz.py tests/education/test_induction_viz.py
git commit -m "induction_viz helpers: trace-based attn heatmap + Olsson prefix-matching score, 11 CPU tests green"
```

---

### Task 2: Notebook restructure (delete broken viz, insert panels)

**Files:**
- Modify: `notebooks/education/stage2c_induction_tinystories.ipynb`
- Create (scratch, not committed): `<scratchpad>/restructure_nb.py`

**Interfaces:**
- Consumes from Task 1: `attn_heatmap`, `pick_example`, `diag_score`, `prefix_matching_score`, `Example` (fields `prompt, ids, tokens, query_pos, key_pos, target_id, target_str, topk`).
- Consumes from existing notebook globals: `tokenizer`, `model`, `device`, `df` (score table, columns `pattern_idx/type/length/layer/head/score`), `eval_patterns`, `all_patterns_pats`, `all_scores`, `N_CTX`.
- Produces notebook globals used across new cells: `ex` (Example), `ex_pats` (`[n_layers, 1, n_heads, S, S]` tensor), `L0_head`, `L1_head`, `encode`, `decode_token`, `S`.

The restructure is one deterministic script (many inserts via `NotebookEdit` would require re-reading ids after every insert). The script locates cells by content markers, never by index, so it is robust to the notebook's current state.

- [ ] **Step 1: Write the restructure script**

Write `<scratchpad>/restructure_nb.py` with the following content. The `SOURCES` dict holds every new cell verbatim.

```python
"""Restructure stage2c_induction_tinystories.ipynb per the 2026-07-03 spec.

- rewrites the intro markdown (adds route map + paper map table)
- deletes the broken node-link cell and the old heatmap-pair cell
- inserts: viz-setup, panels 1-5 (markdown+code pairs), sandbox, closing
Idempotent-ish: refuses to run twice (checks for a marker string).
"""

import nbformat

NB = "/home/yassermakram/code/fanous-llm-lens/notebooks/education/stage2c_induction_tinystories.ipynb"

INTRO_MD = '''\
# Stage 2c: Induction heads on TinyStories — from zero

This notebook trains a 2-layer, 2-head transformer on **TinyStories** and then
builds up the **induction circuit** one idea at a time. It is a personal
learning notebook: the goal is that by the end you can look at an attention
pattern and *see* the circuit.

**Limits up front:** the model is tiny (500-token BPE vocab, d_model=256,
context 64) and trained only on TinyStories. Its predictions are toy-grade.
That is fine — the induction circuit is the same algorithm frontier models use,
and small is what makes it visible.

## The route (5 panels, one idea each)

1. **The behavior** — the model completes a repeated pair. No mechanism yet.
2. **L0 prev-token head** — every token learns who is directly behind it.
3. **L1 induction head** — a repeated token looks at what *followed* its first occurrence.
4. **Composition** — how the two heads chain (K-composition). Why one layer can't do this.
5. **The punchline** — the same circuit fires on *random* tokens: it's an algorithm, not memorization. That is in-context learning.

Then a **sandbox**: feed the model your own text and watch both heads.

## Paper map

| Notebook section | Concept | Paper |
|---|---|---|
| Training + eval patterns (below) | repeated-random-sequence evaluation | [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), "Argument 1" |
| `induction_from_patterns` score | prefix-matching score | Olsson et al. 2022, "Definition of induction heads" |
| Panel 1 | behavioral definition of in-context learning | Olsson et al. 2022, "Defining in-context learning" |
| Panel 2 | previous-token heads, QK vs OV circuits | [Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html), "Two-Layer Attention-Only Models" |
| Panel 3 | induction head = prefix matching + copying | Olsson et al. 2022 |
| Panel 4 | K-composition / virtual heads; why depth ≥ 2 | Elhage et al. 2021, "Composition" + term expansion |
| Panel 5 | induction heads drive in-context learning | Olsson et al. 2022, main claim |
'''

VIZ_SETUP = '''\
# Viz setup: shared helpers + pick the running example for panels 1-4
import importlib

import induction_viz

importlib.reload(induction_viz)
from induction_viz import attn_heatmap, diag_score, pick_example, prefix_matching_score


def encode(text):
    return tokenizer.encode(text).ids


def decode_token(tid):
    return tokenizer.decode([int(tid)]).strip() or '·'


def topk5(ids):
    with torch.no_grad():
        logits = model(torch.tensor([ids]).to(device))[0, -1]
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, 5)
    return [(int(i), float(v)) for i, v in zip(idx, vals)]


# Prompts end on the SECOND occurrence of a token; the picker keeps the first
# one whose continuation the model actually predicts (top-3). It fails loudly
# rather than fall back to random tokens — panels 1-4 must be meaningful text.
CANDIDATES = [
    'Tom and Lily went to the park . Tom and',
    'One day Tim found a big red ball . Tim',
    'The little bird flew up to the tree . The little bird',
    'Sara had a small cat . The cat was soft . Sara had',
    'Ben and Tom played all day . Ben and',
]
ex = pick_example(CANDIDATES, encode, decode_token, topk5)
print(f'[example] prompt: {ex.prompt!r}')
print(f'[example] tokens: {ex.tokens}')
print(f'[example] 2nd occurrence at pos {ex.query_pos} should attend pos {ex.key_pos} ({ex.target_str!r})')
print(f'[example] model top-5: {[(t, round(p, 3)) for t, p in ex.topk]}')

with torch.no_grad():
    _, ex_cache = model.run_with_cache(torch.tensor([ex.ids]).to(device))
ex_pats = torch.stack(
    [ex_cache['pattern', L].detach().cpu() for L in range(model.cfg.n_layers)]
)
S = len(ex.ids)
print(f'[example] ex_pats shape: {tuple(ex_pats.shape)}  ← (layer, batch, head, query, key)')

# Head selection: L0 by prev-token diagonal on THIS prompt, L1 by mean
# prefix-matching score over the 20 random eval patterns (the score table).
L0_head = max(range(model.cfg.n_heads), key=lambda h: diag_score(ex_pats[0, 0, h].numpy()))
L1_head = int(df[df['layer'] == 1].groupby('head')['score'].mean().idxmax())
print(f'[heads] L0 prev-token head: {L0_head}   L1 induction head: {L1_head}')
'''

P1_MD = '''\
## Panel 1 — the behavior, before any mechanism

The prompt ends on the **second** occurrence of a token pair. If the model has
an induction circuit, it should predict the token that followed the **first**
occurrence — even though this tiny model has no idea what the sentence means.

**Paper hook — Olsson et al. 2022, "Defining in-context learning":** in-context
learning is *behavior*: the model gets better at predicting tokens later in the
context because it can reuse what already appeared. The bar chart below is that
behavior at a single position. The rest of the notebook opens the box to find
the mechanism.
'''

P1_CODE = '''\
from plotly.subplots import make_subplots

first_a = ex.key_pos - 1
colors = ['#d9d9d9'] * S
colors[first_a] = '#ffcc80'      # first occurrence of A
colors[ex.key_pos] = '#a5d6a7'   # B1: the token that followed it
colors[ex.query_pos] = '#ffcc80' # second occurrence of A (the query)

fig = make_subplots(
    rows=2, cols=1, row_heights=[0.35, 0.65], vertical_spacing=0.18,
    subplot_titles=(
        'the input, token by token (orange = repeated token, green = what followed it)',
        f'model top-5 predictions after the second {ex.tokens[ex.query_pos]!r}',
    ),
)
fig.add_trace(
    go.Scatter(
        x=list(range(S)), y=[0] * S, mode='markers+text',
        text=ex.tokens, textposition='top center',
        textfont=dict(size=11, family='monospace'),
        marker=dict(size=26, color=colors, symbol='square'),
        hoverinfo='skip',
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=[S], y=[0], mode='markers+text', text=['?'], textposition='top center',
        textfont=dict(size=11, family='monospace'),
        marker=dict(size=26, color='#ffffff', symbol='square', line=dict(color='#333', width=1)),
        hoverinfo='skip',
    ),
    row=1, col=1,
)
top_labels = [t for t, _ in ex.topk]
top_probs = [p for _, p in ex.topk]
bar_colors = ['#2e7d32' if t == ex.target_str else '#90a4ae' for t in top_labels]
fig.add_trace(go.Bar(x=top_labels, y=top_probs, marker_color=bar_colors), row=2, col=1)
fig.update_xaxes(visible=False, range=[-0.6, S + 0.6], row=1, col=1)
fig.update_yaxes(visible=False, range=[-0.6, 0.9], row=1, col=1)
fig.update_yaxes(title_text='probability', range=[0, 1], row=2, col=1)
fig.update_layout(width=840, height=480, showlegend=False,
                  title_text='Panel 1 — the model completes the repeated pair')
fig.show()
print(f'[P1] target {ex.target_str!r} probability: {dict(ex.topk).get(ex.target_str, 0.0):.3f}')
'''

P2_MD = '''\
## Panel 2 — the L0 prev-token head: "who is behind me?"

Layer 0 can only look at raw token embeddings — no token knows anything about
its neighbours yet. The prev-token head fixes that: **every position attends to
the position directly behind it** and copies that token's identity into its own
residual stream. On the heatmap this is the sub-diagonal (outlined). Read one
row: "this query token looks at → that key token."

**Paper hook — Elhage et al. 2021:** attention heads factor into a **QK
circuit** (*where* to look — this diagonal) and an **OV circuit** (*what* to
move — here, the identity of the previous token). Prev-token heads are the
layer-0 ingredient their two-layer analysis needs for induction. The stamp this
head writes ("Tom is behind me") is what layer 1 will search for.
'''

P2_CODE = '''\
L0_pat = ex_pats[0, 0, L0_head].numpy()
print(f'[P2] L0_pat shape: {L0_pat.shape}  ← (query, key)')
fig = attn_heatmap(
    L0_pat, ex.tokens,
    highlight=[(q, q - 1) for q in range(1, S)],
    title=f'Panel 2 — L0 head {L0_head}: every token looks one step back',
)
fig.show()
print(f'[P2] prev-token (diagonal) strength: {diag_score(L0_pat):.2f}  (1.0 = perfect)')
'''

P3_MD = '''\
## Panel 3 — the L1 induction head: "what followed me last time?"

Same axes, same tokens, layer 1. The circled cell is the whole story: the
query is the **second** occurrence of the repeated token, and it attends to the
token **after** the **first** occurrence — exactly the token the model then
predicts in Panel 1.

**Paper hook — Olsson et al. 2022, definition of induction heads:** a head is
an induction head if it does **prefix matching** (at a repeated token, attend
to the token after the previous occurrence — this circled cell) and **copying**
(its OV circuit raises the probability of the attended token). The
`induction_from_patterns` score computed earlier in this notebook is a
prefix-matching score with known source positions; `prefix_matching_score`
printed below is the same measurement on this real sentence.
'''

P3_CODE = '''\
L1_pat = ex_pats[1, 0, L1_head].numpy()
print(f'[P3] L1_pat shape: {L1_pat.shape}  ← (query, key)')
fig = attn_heatmap(
    L1_pat, ex.tokens,
    highlight=[(ex.query_pos, ex.key_pos)],
    title=(
        f'Panel 3 — L1 head {L1_head}: the second {ex.tokens[ex.query_pos]!r} '
        f'looks at what followed the first'
    ),
)
fig.show()
print(f'[P3] attention on the circled cell: {L1_pat[ex.query_pos, ex.key_pos]:.2f}')
print(f'[P3] prefix-matching score on this prompt: {prefix_matching_score(L1_pat, ex.ids):.2f}')
'''

P4_MD = '''\
## Panel 4 — how the two heads compose (schematic, not model data)

Neither head alone can do induction. The prev-token head only ever looks one
step back; the induction head needs to know *which key has the repeated token
behind it* — information that does not exist in the raw embeddings. The chain:

1. **L0** stamps "«A» is behind me" into the residual stream at B₁'s position.
2. **L1**'s query at the second «A» asks "who has «A» behind them?" — its **key**
   at B₁ is built from the L0 stamp. Query matches key → attention lands on B₁.
3. **L1**'s OV circuit copies B into the output → the prediction in Panel 1.

**Paper hook — Elhage et al. 2021, "Composition":** step 2 is
**K-composition** — the L1 head's key reads from residual content *written by
an L0 head*, forming a "virtual head" that neither layer contains alone. Their
term-expansion argument shows a **one-layer** model has no such term: this is
why induction requires depth ≥ 2. (The sibling notebook
`stage2_dash2_composition_induction_reference.ipynb` measures this same
K-composition directly from the weight matrices.)
'''

P4_CODE = '''\
A, B = ex.tokens[ex.query_pos], ex.target_str
names = [f'{A}\\u2081', f'{B}\\u2081', '…', f'{A}\\u2082', f'{B}?']
node_colors = ['#ffcc80', '#a5d6a7', '#eeeeee', '#ffcc80', '#ffffff']

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[0, 1, 2, 3, 4], y=[0] * 5, mode='markers+text', text=names,
        textposition='middle center', textfont=dict(size=13, family='monospace'),
        marker=dict(size=58, color=node_colors, symbol='square', line=dict(color='#555', width=1)),
        hoverinfo='skip',
    )
)


def arrow(x0, x1, y, color, label):
    # lane arrows: risers connect the lane to the two nodes it links
    for xr in (x0, x1):
        fig.add_shape(type='line', x0=xr, x1=xr, y0=0.12, y1=y,
                      line=dict(color=color, width=1, dash='dot'))
    fig.add_annotation(x=x1, y=y, ax=x0, ay=y, xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color)
    fig.add_annotation(x=(x0 + x1) / 2, y=y + 0.13, text=label, showarrow=False,
                       font=dict(size=10, color=color))


arrow(0, 1, 0.45, '#2b6cb0', f'1· L0 prev-token head stamps "{A} is behind me" into {B}\\u2081')
arrow(3, 1, 0.85, '#e65100', f'2· L1 query "who has {A} behind them?" matches the stamp — K-composition')
arrow(1, 4, 1.25, '#2e7d32', f'3· L1 OV circuit copies "{B}" into the prediction')

fig.update_xaxes(visible=False, range=[-0.6, 4.6])
fig.update_yaxes(visible=False, range=[-0.5, 1.6])
fig.update_layout(width=900, height=420,
                  title='Panel 4 — the induction circuit as a chain of two heads')
fig.show()
'''

P5_MD = '''\
## Panel 5 — the punchline: it works on gibberish

Everything so far used one meaningful sentence. But the 20 evaluation patterns
scored earlier are **uniform-random token IDs** — deliberate nonsense. Below is
the same L1 head on one of them: the tick labels are gibberish, and the stripe
is still there (outlined cells = where prefix matching says it should be).

The circuit never cared about meaning. It is an **algorithm over repetition**:
*find the previous occurrence of the current token, look at what followed,
predict that*. That is why it fires on text it has never seen — and that
generalization is what Olsson et al. mean by in-context learning.

**Paper hook — Olsson et al. 2022, "Argument 1":** they evaluate induction
heads on repeated **random** sequences for exactly this reason — random tokens
rule out memorized bigrams, so anything that scores high must implement the
abstract algorithm. Their main claim: these heads are the dominant mechanism of
in-context learning in small transformers. The fresh-seed patterns in the next
cell are this notebook's held-out version of that argument.
'''

P5_CODE = '''\
# best-scoring SHORT random pattern (≤ 24 tokens keeps tick labels readable)
short = df[(df['layer'] == 1) & (df['head'] == L1_head) & (df['length'] <= 24)]
p_idx = int(short.loc[short['score'].idxmax(), 'pattern_idx'])
p = eval_patterns[p_idx]
rand_tokens = [decode_token(t) for t in p['pattern']]
rand_L1 = all_patterns_pats[p_idx][1, 0, L1_head].numpy()
print(f'[P5] rand_L1 shape: {rand_L1.shape}  ← (query, key)')

stripe = [
    (t, int(p['src'][t]) + 1)
    for t in range(len(p['src']))
    if int(p['src'][t]) >= 0 and int(p['src'][t]) + 1 < t
]
fig = attn_heatmap(
    rand_L1, rand_tokens, highlight=stripe,
    title=(
        f'Panel 5 — same L1 head {L1_head}, RANDOM tokens '
        f'(pattern {p_idx}, type={p["type"]}): the stripe survives'
    ),
)
fig.show()
print(f'[P5] prefix-matching score on this random pattern: {float(all_scores[p_idx][1][L1_head]):.2f}')
print(f'[P5] mean over all 20 random patterns: '
      f'{df[(df["layer"] == 1) & (df["head"] == L1_head)]["score"].mean():.2f}')
'''

SANDBOX_MD = '''\
## Sandbox — poke the circuit yourself

`explore(text)` runs your text through the model and shows both heads through
the same heatmap lens as panels 2–3, plus the prefix-matching score.

Things to try:
- repeat a name: `"Ben saw a dog . Ben"` — does the stripe appear?
- break the repetition: change the second `Ben` to `Sam` — stripe gone?
- widen the gap between occurrences — does the score hold? (Panel 3's cell
  should move, not fade: prefix matching is position-relative, not distance-fixed)
'''

SANDBOX_CODE = '''\
def explore(text: str):
    ids = encode(text)
    if len(ids) < 3:
        print('[sandbox] need at least 3 tokens')
        return
    if len(ids) > N_CTX:
        ids = ids[:N_CTX]
        print(f'[sandbox] truncated to {N_CTX} tokens')
    toks = [decode_token(t) for t in ids]
    with torch.no_grad():
        _, cache = model.run_with_cache(torch.tensor([ids]).to(device))
    for L, head, name in [(0, L0_head, 'prev-token'), (1, L1_head, 'induction')]:
        pat = cache['pattern', L][0, head].detach().cpu().numpy()
        attn_heatmap(pat, toks, title=f'L{L} head {head} — {name}').show()
    l1 = cache['pattern', 1][0, L1_head].detach().cpu().numpy()
    print(f'[sandbox] prefix-matching score (L1 h{L1_head}): {prefix_matching_score(l1, ids):.2f}')


explore('The dog saw a cat . The dog')
'''

CLOSING_MD = '''\
## Recap

1. The model **completes repeated pairs** it has never seen (Panel 1) —
   in-context learning as behavior.
2. An **L0 prev-token head** gives every token its left neighbour's identity
   (Panel 2, the diagonal).
3. An **L1 induction head** uses that stamp to attend to *what followed the
   first occurrence* (Panel 3, the circled cell / the stripe).
4. The two chain via **K-composition** — the reason this needs two layers
   (Panel 4).
5. The circuit fires on **random tokens** (Panel 5): an algorithm over
   repetition, not memorized text — Olsson et al.'s argument that induction
   heads drive in-context learning.

**Where this connects:** `stage2_dash2_composition_induction_reference.ipynb`
measures the same K-composition from the weight matrices (a 0.43 virtual-head
strength); this notebook observed it behaviorally in the attention patterns.
Two views of one circuit.
'''

nb = nbformat.read(NB, as_version=4)
joined = "\n".join("".join(c.source) for c in nb.cells)
assert "Panel 5 — the punchline" not in joined, "restructure already applied"


def find(marker):
    for i, c in enumerate(nb.cells):
        if marker in "".join(c.source):
            return i
    raise SystemExit(f"marker not found: {marker!r}")


md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

# 1. rewrite intro
nb.cells[find("Stage 2c (revised)")].source = INTRO_MD

# 2. drop the broken node-link cell and the old heatmap-pair cell
i_nodelink = find("def node_link_diagram")
del nb.cells[i_nodelink]
i_heatpair = find("best_L1_idx = df[df['layer'] == 1]")
del nb.cells[i_heatpair]

# 3. insert viz-setup + panels where the node-link cell used to be
new_cells = [
    code(VIZ_SETUP),
    md(P1_MD), code(P1_CODE),
    md(P2_MD), code(P2_CODE),
    md(P3_MD), code(P3_CODE),
    md(P4_MD), code(P4_CODE),
    md(P5_MD), code(P5_CODE),
]
nb.cells[i_nodelink:i_nodelink] = new_cells

# 4. sandbox + closing go after the fresh-pattern verification cell (the end)
i_fresh = find("fresh_patterns = make_paper_patterns")
nb.cells[i_fresh + 1 : i_fresh + 1] = [md(SANDBOX_MD), code(SANDBOX_CODE), md(CLOSING_MD)]

nbformat.write(nb, NB)
print(f"[restructure] done: {len(nb.cells)} cells")
for i, c in enumerate(nb.cells):
    head = "".join(c.source).strip().split("\n")[0][:72]
    print(f"  {i:2d} [{c.cell_type[:4]}] {head}")
```

- [ ] **Step 2: Run the script**

Run: `cd /home/yassermakram/code/fanous-llm-lens && uv run python <scratchpad>/restructure_nb.py`
Expected output: `[restructure] done: 25 cells` followed by the cell listing, in this order: intro-md, setup, dataset, tokenizer-train, tokenize, train, load, patterns, inference, score-table, viz-setup, (P1..P5 md+code ×5), fresh-patterns, sandbox-md, sandbox-code, closing-md.

- [ ] **Step 3: Verify the notebook is valid and ordered**

Run: `cd /home/yassermakram/code/fanous-llm-lens && uv run python -c "import nbformat; nb = nbformat.read('notebooks/education/stage2c_induction_tinystories.ipynb', as_version=4); nbformat.validate(nb); print('valid,', len(nb.cells), 'cells')"`
Expected: `valid, 25 cells`

Also confirm the broken code is gone:
Run: `grep -c "node_link_diagram\|orange=forward" notebooks/education/stage2c_induction_tinystories.ipynb`
Expected: `0` (grep exits 1)

- [ ] **Step 4: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens
git add notebooks/education/stage2c_induction_tinystories.ipynb
git commit -m "stage2c notebook: 5-panel stepwise induction walkthrough replaces broken node-link viz; every panel mapped to Elhage 2021 / Olsson 2022"
```

---

### Task 3: End-to-end execution + human-reviewable render

**Files:**
- Modify: `notebooks/education/stage2c_induction_tinystories.ipynb` (executed outputs, temporary)
- Create (scratch): `<scratchpad>/stage2c_executed.html`

**Interfaces:**
- Consumes: the restructured notebook from Task 2, the existing checkpoint in `notebooks/education/checkpoints/induction_tiny/`.
- Produces: an HTML render for the user to eyeball, and confidence the notebook runs clean.

Note: cell-5 always trains 2000 steps (it loads the checkpoint first, then continues). Expect several minutes on the iGPU; this is within the project's <10 min budget and only refines the existing checkpoint.

- [ ] **Step 1: Execute the notebook in place**

Run:
```bash
cd /home/yassermakram/code/fanous-llm-lens/notebooks/education
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 stage2c_induction_tinystories.ipynb
```
Expected: exits 0. If it fails, the traceback names the cell — fix forward (most likely candidate: no prompt in `CANDIDATES` passes the picker; if so, add more TinyStories-style candidates ending on a second occurrence, or relax `top_n` to 5 in the `pick_example` call, and note which prompt won).

- [ ] **Step 2: Programmatic sanity check of the executed outputs**

Run:
```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run python - <<'EOF'
import nbformat

nb = nbformat.read("notebooks/education/stage2c_induction_tinystories.ipynb", as_version=4)
errors = [
    o for c in nb.cells if c.cell_type == "code"
    for o in c.get("outputs", []) if o.get("output_type") == "error"
]
assert not errors, errors
figs = sum(
    1 for c in nb.cells for o in c.get("outputs", [])
    if "application/vnd.plotly.v1+json" in o.get("data", {})
)
print(f"0 errors, {figs} plotly figures (expect ≥ 8: P1-P5, 2 sandbox, ≥1 more)")
EOF
```
Expected: `0 errors,` and a figure count ≥ 8.

- [ ] **Step 3: Render HTML and hand to the user for eyeballing**

```bash
cd /home/yassermakram/code/fanous-llm-lens/notebooks/education
uv run jupyter nbconvert --to html stage2c_induction_tinystories.ipynb \
  --output-dir <scratchpad> --output stage2c_executed.html
```
Then send `<scratchpad>/stage2c_executed.html` to the user (SendUserFile, display render) with a caption asking them to check: labels inside bounds, Panel 2 diagonal outlined, Panel 3 circled cell on the right token, Panel 5 stripe on gibberish. **Wait for user confirmation before Task 4** — the spec's "eyeball every figure" step is the user's gate.

---

### Task 4: Clear outputs and finalize

**Files:**
- Modify: `notebooks/education/stage2c_induction_tinystories.ipynb` (strip outputs)

- [ ] **Step 1: Clear outputs (project convention)**

```bash
cd /home/yassermakram/code/fanous-llm-lens/notebooks/education
uv run jupyter nbconvert --clear-output --inplace stage2c_induction_tinystories.ipynb
```

- [ ] **Step 2: Full test suite + lint pass**

```bash
cd /home/yassermakram/code/fanous-llm-lens
uv run pytest tests/education -q
uv run ruff check notebooks/education/induction_viz.py tests/education/test_induction_viz.py
```
Expected: all tests pass, ruff clean.

- [ ] **Step 3: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens
git add notebooks/education/stage2c_induction_tinystories.ipynb
git commit -m "stage2c: executed end-to-end on checkpoint — picker chose a real prompt, L1 stripe confirmed on random patterns (outputs cleared)"
```

Then confirm with the user before any merge (project git workflow: never merge without confirmation).

---

## Self-Review Notes

- **Spec coverage:** intro route-map + paper table (Task 2 INTRO_MD); helper with real traces, token ticks, hover, highlight (Task 1); example picker with loud failure (Task 1 + VIZ_SETUP); Panels 1–5 incl. paper hooks in every markdown cell (Task 2); sandbox (Task 2); closing recap with stage2dash² cross-link (Task 2); deletion of node-link cell + old heatmap pair, cells 1–9 and fresh-pattern cell untouched (Task 2 script deletes exactly two cells); verification incl. human eyeball gate (Task 3); clear-outputs convention (Task 4).
- **Type consistency:** `Example` fields used in cells (`ex.prompt/ids/tokens/query_pos/key_pos/target_str/topk`) match the dataclass; `attn_heatmap(pattern, tokens, highlight, title)` signature identical across Panels 2/3/5 and sandbox; `df` column names (`layer/head/score/length/pattern_idx`) match the existing cell-9 dataframe.
- **Known judgment calls:** L1 head chosen by mean score over eval patterns (stable) rather than per-prompt argmax; Panel 5 restricted to length ≤ 24 patterns for tick readability (the score table still covers all 20).
