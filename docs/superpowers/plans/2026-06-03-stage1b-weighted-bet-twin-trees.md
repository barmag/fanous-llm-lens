# Stage 1b Weighted-Bet Twin Trees — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confusing spring-layout transition graph in the Stage 1b word-level notebook with two stacked directed "weighted-bet" trees (MSA vs Masri) where arrow direction, thickness, and a printed label all encode transition probability.

**Architecture:** All logic lives inside the notebook (cells must stay self-contained for Colab). Reference cell 6 is rewritten with three small pure helpers — `build_transition_tree`, `tree_layout`, `plot_weighted_tree` — plus a driver that renders both dialects into a single `plotly` `make_subplots(rows=2)` figure. A new pytest module exercises the helpers by extracting and `exec`-ing the notebook cell against tiny synthetic probability dicts (no network, no GPU).

**Tech Stack:** Python, Plotly 6.x (`graph_objects` + `subplots`), pytest. No new plotting/graph dependencies. Notebook cells loaded via stdlib `json` (no `nbformat`/`jupyter` dependency).

**Design spec:** `docs/superpowers/specs/2026-06-03-stage1b-weighted-bet-twin-trees-design.md`

**Branch:** `stage1b-weighted-bet-trees` (already created; the design spec is already committed here).

---

## File Structure

- **Modify** `notebooks/education/stage1_b_word_reference.ipynb`
  - cell 1 (markdown): minor wording update to "What we are showing"
  - cell 5 (markdown): rewrite the intro to describe weighted-bet twin trees
  - cell 6 (code): full rewrite — the three helpers + driver
  - cells 3–4: **unchanged** (tokenization fix + bigram counts already in place)
- **Modify** `notebooks/education/stage1_b_word_experiment.ipynb`
  - cell 6 (code): replace the threshold hint with weighted-tree guidance
- **Create** `tests/education/test_stage1b_graph.py` — pytest module for the helpers + figure encoding
- **Reference (do not modify)** `notebooks/education/verify_notebooks.py` — must still pass

---

## Task 1: Commit the existing bug fixes as the "make it work" baseline

The working tree already contains the tokenization NameError fix (cell 3, both notebooks) and the threshold floor change (reference cell 6 + experiment hint) from earlier in this session. Commit these first as one logical chunk so the redesign is a clean separate commit.

**Files:**
- Modify (already edited, uncommitted): `notebooks/education/stage1_b_word_reference.ipynb`, `notebooks/education/stage1_b_word_experiment.ipynb`

- [ ] **Step 1: Confirm the current state runs end-to-end on real data**

Run:
```bash
cd /home/yassermakram/code/fanous-llm-lens
python - <<'PY'
import json
nb=json.load(open('notebooks/education/stage1_b_word_reference.ipynb'))
code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
import plotly.graph_objects as go
go.Figure.show=lambda self: None
ctx={}
exec(code[1],ctx); exec(code[2],ctx); exec(code[3],ctx)
assert ctx['masri_G'].number_of_edges()>0 and ctx['msa_G'].number_of_edges()>0
print('baseline OK: MSA edges', ctx['msa_G'].number_of_edges(), '| Masri edges', ctx['masri_G'].number_of_edges())
PY
```
Expected: `baseline OK: MSA edges 48 | Masri edges 33` (counts may vary slightly with dataset revision; both must be > 0).

- [ ] **Step 2: Commit the bug fixes**

```bash
git add notebooks/education/stage1_b_word_reference.ipynb notebooks/education/stage1_b_word_experiment.ipynb
git commit -m "fix(stage1b): tokenize text into words and lower edge threshold

Cell 3 produced only character strings but cell 4 used msa_words/masri_words,
crashing with NameError; add the whitespace tokenization step. Also lower the
graph edge threshold from 0.1 to 0.05 so the flat word-level Masri tweet
distributions are no longer filtered to an empty graph.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

Expected: one commit created; `git status` shows the two notebooks no longer modified.

---

## Task 2: Write the failing test module

**Files:**
- Create: `tests/education/test_stage1b_graph.py`

- [ ] **Step 1: Ensure pytest is available**

Run:
```bash
python -m pytest --version 2>/dev/null || uv pip install pytest
python -m pytest --version
```
Expected: a pytest version prints (e.g. `pytest 8.x`). `pytest` is a dev tool already named in `CLAUDE.md` conventions; installing it is expected.

- [ ] **Step 2: Write the test file**

Create `tests/education/test_stage1b_graph.py` with exactly this content:

```python
"""Tests for the Stage 1b weighted-bet twin-tree graph cell.

We load the plotting cell (cell index 6) straight out of the reference
notebook and exec it against tiny synthetic probability dicts, so the test
needs no network and no GPU. The cell defines build_transition_tree,
tree_layout, _edge_width, plot_weighted_tree and a driver that builds
`fig`, `msa_edges`, `masri_edges`.
"""
import json
from pathlib import Path

import plotly.graph_objects as go

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "education" / "stage1_b_word_reference.ipynb"
)

# Synthetic transitions. Seeds match the notebook's MSA_SEED / MASRI_SEED.
MSA_P = {
    ("الذي", "كان"): 0.5, ("الذي", "في"): 0.3, ("الذي", "من"): 0.1,
    ("الذي", "عن"): 0.05, ("كان", "في"): 0.6, ("كان", "له"): 0.4,
    ("في", "مصر"): 0.7, ("في", "كل"): 0.3,
}
MASRI_P = {
    ("اللي", "كان"): 0.5, ("اللي", "مش"): 0.3, ("اللي", "حصل"): 0.1,
    ("اللي", "بقى"): 0.05, ("كان", "فيه"): 0.6, ("كان", "له"): 0.4,
    ("مش", "عايز"): 0.8, ("مش", "ممكن"): 0.2,
}


def _load_cell6(msa_probs=MSA_P, masri_probs=MASRI_P):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # code cells: 0=colab setup, 1=data, 2=bigrams, 3=graph
    src = "".join(code_cells[3]["source"])
    ns = {"msa_probs": msa_probs, "masri_probs": masri_probs}
    go.Figure.show = lambda self: None  # headless no-op
    exec(compile(src, f"{NB}:graphcell", "exec"), ns)
    return ns


def test_build_tree_keeps_top_k_likeliest_new_children():
    ns = _load_cell6()
    edges = ns["build_transition_tree"](MSA_P, "الذي", max_depth=1, top_k=3)
    children = [child for _parent, child, _p, _depth in edges]
    # top-3 of الذي by probability, in descending order; عن (0.05) dropped
    assert children == ["كان", "في", "من"]


def test_build_tree_is_a_strict_tree_no_revisits():
    ns = _load_cell6()
    edges = ns["build_transition_tree"](MSA_P, "الذي", max_depth=2, top_k=3)
    children = [child for _parent, child, _p, _depth in edges]
    assert len(children) == len(set(children))  # each word placed once
    # depth recorded on the edge equals the parent's depth
    seed_edges = [e for e in edges if e[0] == "الذي"]
    assert all(depth == 0 for *_x, depth in seed_edges)


def test_layout_is_deterministic_left_to_right():
    ns = _load_cell6()
    edges = ns["build_transition_tree"](MSA_P, "الذي", max_depth=2, top_k=3)
    pos = ns["tree_layout"](edges, "الذي")
    assert pos["الذي"][0] == 0          # seed at depth 0
    # any direct child of the seed sits at x == 1
    child = edges[0][1]
    assert pos[child][0] == 1
    # calling twice gives identical positions
    assert ns["tree_layout"](edges, "الذي") == pos


def test_edge_width_increases_with_probability():
    ns = _load_cell6()
    w = ns["_edge_width"]
    assert w(0.9) > w(0.5) > w(0.1)
    assert w(0.0) < w(1.0)


def test_figure_has_two_subplots_and_directed_labelled_edges():
    ns = _load_cell6()
    fig = ns["fig"]
    assert isinstance(fig, go.Figure)
    # one markers+text node trace per dialect
    node_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
    assert len(node_traces) == 2
    # one arrow annotation per edge across both trees
    arrow_anns = [a for a in fig.layout.annotations if a.showarrow and a.ax is not None]
    assert len(arrow_anns) == len(ns["msa_edges"]) + len(ns["masri_edges"])
    # arrow thickness varies with probability (not all identical)
    widths = {round(float(a.arrowwidth), 3) for a in arrow_anns}
    assert len(widths) > 1
    # at least one probability label like "0.70" is present
    label_texts = [a.text for a in fig.layout.annotations if not a.showarrow and a.text]
    assert any(t[:1].isdigit() and "." in t for t in label_texts)
```

- [ ] **Step 3: Run the tests and confirm they FAIL**

Run:
```bash
python -m pytest tests/education/test_stage1b_graph.py -v
```
Expected: FAIL — the current cell 6 has no `build_transition_tree`/`tree_layout`/`_edge_width`/`fig`, so the tests raise `KeyError`/`NameError` during `_load_cell6`.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/education/test_stage1b_graph.py
git commit -m "test(stage1b): add tests for weighted-bet twin-tree graph cell

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Rewrite reference cell 6 to pass the tests

**Files:**
- Modify: `notebooks/education/stage1_b_word_reference.ipynb` (cell index 6, `cell_id` = `cell-6`)

- [ ] **Step 1: Replace cell 6 with the weighted-tree implementation**

Use the NotebookEdit tool to replace `cell-6` (cell_type `code`) of
`notebooks/education/stage1_b_word_reference.ipynb` with exactly this source:

```python
# 🕸️ Plot the Weighted-Bet Transition Trees (MSA vs Masri)
# Each arrow points from a word to a likely NEXT word, and shows probability 3 ways:
#   • direction  — current word ──▶ next word
#   • THICKNESS  — thicker arrow = higher transition probability
#   • the number — the probability itself, printed on the arrow
# We grow a small tree from one seed word per dialect: at each step keep the
# top_k likeliest NEW next words, down to max_depth. No spring layout, and no
# magic probability threshold — top_k does the selecting.
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_transition_tree(probs, seed, max_depth=2, top_k=3):
    """Grow a strict tree from `seed`.

    At each node keep the `top_k` likeliest next words we have not placed yet,
    expanding breadth-first to `max_depth`.
    Returns a list of edges as (parent, child, prob, depth) tuples.
    """
    edges = []
    visited = {seed}
    frontier = [(seed, 0)]
    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        outgoing = sorted(
            ((w2, p) for (w1, w2), p in probs.items() if w1 == node),
            key=lambda wp: wp[1], reverse=True,
        )
        kept = 0
        for child, p in outgoing:
            if child in visited:
                continue
            edges.append((node, child, p, depth))
            visited.add(child)
            frontier.append((child, depth + 1))
            kept += 1
            if kept >= top_k:
                break
    return edges


def tree_layout(edges, seed):
    """Deterministic left->right layout.

    x = depth (so the tree flows left to right); y = evenly spaced, vertically
    centred slot among the words at that depth. Returns {word: (x, y)}.
    """
    depth_of = {seed: 0}
    for _parent, child, _p, depth in edges:
        depth_of[child] = depth + 1
    levels = defaultdict(list)
    for word, d in depth_of.items():
        levels[d].append(word)
    pos = {}
    for d, words in levels.items():
        n = len(words)
        for i, word in enumerate(words):
            pos[word] = (d, (n - 1) / 2 - i)
    return pos


W_MIN, W_MAX = 1.5, 9.0


def _edge_width(p):
    """Map a probability in [0, 1] to a line width in [W_MIN, W_MAX]."""
    return W_MIN + p * (W_MAX - W_MIN)


def plot_weighted_tree(fig, row, edges, pos, color, seed):
    """Draw one weighted, directed tree into subplot `row` (1-indexed)."""
    suffix = "" if row == 1 else str(row)
    xref, yref = f"x{suffix}", f"y{suffix}"
    for parent, child, p, _depth in edges:
        x0, y0 = pos[parent]
        x1, y1 = pos[child]
        # the arrow itself carries direction + thickness
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref=xref, yref=yref, axref=xref, ayref=yref,
            showarrow=True, arrowhead=3, arrowsize=1,
            arrowwidth=_edge_width(p), arrowcolor=color,
            text="", standoff=16, startstandoff=22,
        )
        # the probability, printed at the arrow's midpoint
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            xref=xref, yref=yref, showarrow=False,
            text=f"{p:.2f}", font=dict(size=10, color="#333"),
            bgcolor="rgba(255,255,255,0.75)",
        )
    words = list(pos.keys())
    fig.add_trace(
        go.Scatter(
            x=[pos[w][0] for w in words],
            y=[pos[w][1] for w in words],
            mode="markers+text",
            text=words, textposition="middle center",
            textfont=dict(size=13, color="#111"),
            marker=dict(
                color=[color if w == seed else "#d9d9d9" for w in words],
                size=[42 if w == seed else 28 for w in words],
                line=dict(width=1.5, color="#333"),
            ),
            hoverinfo="text", showlegend=False,
        ),
        row=row, col=1,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=1)


# --- Build & draw both dialects -------------------------------------------
MSA_SEED, MASRI_SEED = "الذي", "اللي"
MAX_DEPTH, TOP_K = 2, 3

msa_edges = build_transition_tree(msa_probs, MSA_SEED, MAX_DEPTH, TOP_K)
masri_edges = build_transition_tree(masri_probs, MASRI_SEED, MAX_DEPTH, TOP_K)
msa_pos = tree_layout(msa_edges, MSA_SEED)
masri_pos = tree_layout(masri_edges, MASRI_SEED)

fig = make_subplots(
    rows=2, cols=1, vertical_spacing=0.12,
    subplot_titles=(
        "MSA · seed الذي (relative pronoun)",
        "Masri · seed اللي (relative pronoun)",
    ),
)
plot_weighted_tree(fig, 1, msa_edges, msa_pos, "#2b6cb0", MSA_SEED)
plot_weighted_tree(fig, 2, masri_edges, masri_pos, "#dd6b20", MASRI_SEED)
fig.update_layout(
    height=680,
    title_text="Next-word bets: a thicker, labelled arrow = a likelier next word",
    margin=dict(l=20, r=20, t=80, b=20),
)
fig.show()
```

- [ ] **Step 2: Run the tests and confirm they PASS**

Run:
```bash
python -m pytest tests/education/test_stage1b_graph.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 3: Confirm the cell still runs on real data end-to-end**

Run:
```bash
python - <<'PY'
import json
nb=json.load(open('notebooks/education/stage1_b_word_reference.ipynb'))
code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
import plotly.graph_objects as go
go.Figure.show=lambda self: None
ctx={}
exec(code[1],ctx); exec(code[2],ctx); exec(code[3],ctx)
print('MSA edges', len(ctx['msa_edges']), '| Masri edges', len(ctx['masri_edges']))
assert ctx['msa_edges'] and ctx['masri_edges']
assert isinstance(ctx['fig'], go.Figure)
print('real-data render OK')
PY
```
Expected: both edge counts > 0 (each is at most `top_k + top_k*top_k` = up to ~12) and `real-data render OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/education/stage1_b_word_reference.ipynb
git commit -m "feat(stage1b): redesign transition graph as weighted-bet twin trees

Replace the spring-layout NetworkX graph with two stacked directed trees
(MSA blue, Masri orange). Arrow direction, thickness, and a printed label
all encode transition probability; deterministic left-to-right layout with
no spring hairball. Retires the threshold magic number in favour of top_k.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Update the reference markdown and the experiment notebook

**Files:**
- Modify: `notebooks/education/stage1_b_word_reference.ipynb` (cell-1 and cell-5 markdown)
- Modify: `notebooks/education/stage1_b_word_experiment.ipynb` (cell-6 code/TODO)

- [ ] **Step 1: Update reference cell-5 markdown**

Use NotebookEdit to replace `cell-5` (cell_type `markdown`) of the **reference** notebook with exactly:

```markdown
## 🗺️ Reading the Next-Word "Subway Bets"
For each dialect we pick one seed word — the relative pronoun, MSA **الذي** vs
Masri **اللي** — and grow a small tree of its most likely next words.

Each arrow says three things at once:
- **which way** it points — from a word to the word that tends to follow it,
- **how thick** it is — thicker means a higher transition probability,
- **the number on it** — that probability, printed so you can read it directly.

We keep the top-3 next words at each step, two steps deep (`TOP_K` and
`MAX_DEPTH` below — dial them up if you want a bushier tree). Compare the two
trees: same grammatical job, different words and different company.
```

- [ ] **Step 2: Update reference cell-1 markdown (the "What we are showing" paragraph)**

Use NotebookEdit to replace `cell-1` (cell_type `markdown`) of the **reference** notebook with exactly:

```markdown
# Stage 1b: Zero-Layer Transformer (Word-Level) 🗺️
## target: Dialect Next-Word Bets (Bigram Transition Trees)

Now that we understand how character transition metrics work, let's step up to the **Word Level**.

In a word-level language model, the vocabulary is composed of whole words.
Because we are predicting word-to-word transitions, training a full PyTorch model can take much longer due to the large vocabulary size.

Since we already proved in Stage 1a that training a Zero-Layer network converges to the statistical bigram counts, we will **skip the training loop** and directly calculate the optimal matrices using direct counts!

### 💡 What we are showing:
From a single word, the next word is not one fixed answer — it is a **ranked set
of bets**. We grow a small tree from a core grammatical word in each dialect
('الذي' in MSA vs 'اللي' in Masri), drawing each likely next word as a directed
arrow whose **thickness and label show how probable it is**. Side by side, the
two trees reveal how the dialects branch differently.
```

- [ ] **Step 3: Update the experiment notebook cell-6**

Use NotebookEdit to replace `cell-6` (cell_type `code`) of the **experiment** notebook with exactly:

```python
# 🕸️ Plot the Weighted-Bet Transition Trees (MSA vs Masri)
# TODO: For each dialect, grow a small tree from a seed word and draw it so the
# probability of each next word is obvious at a glance.
#
# Build three helpers and a driver:
#   1. build_transition_tree(probs, seed, max_depth=2, top_k=3)
#        BFS from `seed`; at each node keep the top_k likeliest NEW next words.
#        Return edges as (parent, child, prob, depth). Use top_k to select —
#        do NOT use a probability threshold (word-level distributions are flat,
#        so a threshold like 0.1 leaves the Masri tree empty).
#   2. tree_layout(edges, seed)
#        Deterministic left->right positions: x = depth, y = evenly spaced slot.
#        (Avoid nx.spring_layout — it produces an unreadable hairball.)
#   3. plot_weighted_tree(fig, row, edges, pos, color, seed)
#        Draw into a plotly subplot. Encode probability THREE ways:
#          • a directed arrow (annotation, parent -> child),
#          • arrow thickness scaled to the probability,
#          • the probability printed as a label on the arrow.
#
# Then build both trees (MSA seed 'الذي', Masri seed 'اللي') into a
# make_subplots(rows=2, cols=1) figure and call fig.show().
```

- [ ] **Step 4: Verify both notebooks are still valid and the reference tests still pass**

Run:
```bash
python - <<'PY'
import json
for f in ['stage1_b_word_reference.ipynb','stage1_b_word_experiment.ipynb']:
    p=f'notebooks/education/{f}'
    nb=json.load(open(p))
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            src=''.join(c['source']).replace('!pip install','pass #!pip install')
            compile(src,f'{p}:cell{i}','exec')
    print('valid JSON + compiles:',f)
PY
python -m pytest tests/education/test_stage1b_graph.py -v
```
Expected: both notebooks report "valid JSON + compiles"; all 5 tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/education/stage1_b_word_reference.ipynb notebooks/education/stage1_b_word_experiment.ipynb
git commit -m "docs(stage1b): update narrative + experiment TODO for weighted-bet trees

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Full verification and notebook-output hygiene

**Files:**
- Reference (run, do not modify): `notebooks/education/verify_notebooks.py`

- [ ] **Step 1: Confirm no cell outputs were committed**

Run:
```bash
python - <<'PY'
import json
for f in ['stage1_b_word_reference.ipynb','stage1_b_word_experiment.ipynb']:
    nb=json.load(open(f'notebooks/education/{f}'))
    bad=[i for i,c in enumerate(nb['cells'])
         if c['cell_type']=='code' and (c.get('outputs') or c.get('execution_count'))]
    print(f, 'cells with outputs:', bad)
    assert not bad, f'{f} has stray outputs in cells {bad}'
print('output hygiene OK')
PY
```
Expected: both notebooks `cells with outputs: []` and `output hygiene OK`. (Our edits never executed the notebook, so this should hold; if it fails, clear the offending cells' `outputs`/`execution_count`.)

- [ ] **Step 2: Run the repo's notebook verifier against Stage 1b**

Run:
```bash
cd notebooks/education && python verify_notebooks.py b ; cd ../..
```
Expected: `🎉 REFERENCE NOTEBOOKS (b) VERIFIED SUCCESSFULLY!` and exit code 0. This downloads real data and execs every cell with the mocked `Figure.show`, confirming the new `make_subplots` figure renders without error.

- [ ] **Step 3: Final full test run**

Run:
```bash
python -m pytest tests/education/ -v
```
Expected: all tests PASS.

- [ ] **Step 4: Commit any hygiene fixes (only if Step 1 required edits)**

```bash
git add -A notebooks/education
git commit -m "chore(stage1b): clear stray notebook cell outputs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
If Step 1 already reported clean, skip this commit.

---

## Notes / Out of scope

- **Scratch files** in `notebooks/education/` (`fix_notebooks.py`, `fix_notebooks_b.py`,
  `test_char.py`, `test_madar.py`, `test_tweets.py`) are pre-existing untracked scratch
  scripts. This plan does not touch them; ask the user before deleting.
- Stage 1a (char) and Stage 1c (subword) notebooks are unchanged.
- Merging `stage1b-weighted-bet-trees` into `main` is a separate step — confirm with the
  user first (per the project's branch-commit-merge workflow).
```
