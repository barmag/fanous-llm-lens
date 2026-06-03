# Stage 1b Graph Redesign: Weighted-Bet Twin Trees

**Date:** 2026-06-03
**Status:** Design — awaiting implementation
**Scope:** `notebooks/education/stage1_b_word_reference.ipynb` (cell 6 + supporting markdown) and `notebooks/education/stage1_b_word_experiment.ipynb` (cell 6 hint)

## Problem

The Stage 1b word-level notebook visualises dialect bigram transitions as a NetworkX
spring-layout graph rendered in Plotly. It is confusing for learners:

- **No direction** — edges are undirected lines; you cannot tell which word follows which.
- **No probability signal** — every edge is the same 1.5px gray, so the "strongest
  transitions" the lesson is about are invisible.
- **Spring layout = hairball** — node positions are arbitrary; the "subway map" metaphor
  is not delivered.
- **Two separate figures** — MSA and Masri cannot be compared at a glance.
- **A magic threshold** (`threshold=0.05`, formerly `0.1`) gates edges. The `0.1` default
  produced an empty Masri graph because word-level tweet distributions are flat; `0.05`
  patched it but remains a brittle data-dependent constant.

## Intended concept (the one "aha")

From a word, the next word is a **ranked probability distribution**, not a single answer —
and that distribution **differs by dialect**. The visualisation must make probability
*visible*, and keep the MSA↔Masri contrast (the project's North Star) front and centre.

## Design

### Visual encoding

Probability is shown three redundant ways so the point cannot be missed:

1. **Arrowheads** — every edge is directed, current word → next word.
2. **Edge thickness** — line width scales with transition probability `p`.
3. **Printed label** — each edge carries its `p` (e.g. `0.09`) as text.

Seed/root nodes are styled distinctly (filled dialect colour, larger) from discovered
nodes (neutral gray).

### Structure & layout

- **Twin trees in one figure**, stacked: MSA on top (blue) rooted at `الذي`, Masri below
  (orange) rooted at `اللي`. Colour encodes dialect; both readable at once.
- **Depth 2, top-3 transitions per node** (~13 nodes per dialect). `max_depth` and `top_k`
  are parameters so a learner can dial complexity up.
- **Deterministic hierarchical layout**, computed by hand from the BFS tree:
  one axis = BFS depth, the other = evenly spaced sibling slot. No `spring_layout`.
- **Conventional left→right flow**: root on the left, next-words branching rightward.
- **No new dependencies** — layout is hand-computed; rendering stays in Plotly (already
  used in the notebook, interactive, Colab-friendly).

### Code shape (replaces reference cell 6)

Small, single-purpose, independently testable helpers:

- `build_transition_tree(probs, seed, max_depth=2, top_k=3)`
  BFS from `seed`, keeping the `top_k` highest-probability transitions at each node.
  Returns a list of edges as `(parent, child, prob, depth)` tuples.
  **This retires the `threshold` magic number entirely** — `top_k` does the selection,
  which is the mechanism that actually solved the empty-Masri problem. (A node already
  visited is not re-expanded, preventing cycles.)

- `tree_layout(edges, seed)`
  Returns `{word: (x, y)}` positions. `x = depth`, `y` = evenly spaced slot among the
  nodes at that depth. Deterministic; left→right.

- `plot_weighted_tree(fig, row, edges, pos, color, title)`
  Draws one tree into subplot `row`:
  - edges as **one Plotly Scatter line trace per edge** (Plotly cannot vary line width
    within a single trace; trees are small so per-edge traces are cheap),
  - **arrowheads + probability labels as Plotly layout annotations** (arrow annotation
    from parent to child doubles as the arrowhead; a midpoint text annotation prints `p`),
  - nodes as a `markers+text` Scatter trace.

- **Driver**: build both trees, render into a `make_subplots(rows=2, cols=1)` figure with
  per-row titles "MSA" / "Masri", then `fig.show()`.

Edge width mapping: `width = w_min + p * (w_max - w_min)` (e.g. `w_min=1`, `w_max=8`) so
the thickest edge is the most probable and ordering is visually monotonic in `p`.

### Other cells

- **Reference cell 5 (markdown)**: rewrite the "subway map" intro to describe the
  weighted-bet twin trees — "a thicker, labelled arrow means a likelier next word; compare
  how the two dialects branch."
- **Reference cell 1 (markdown, "What we are showing")**: minor wording update to match.
- **Experiment cell 6**: replace the threshold hint with guidance to build the weighted
  tree (directed edges, width ∝ probability, printed labels, depth 2 / top-3) and to use
  `top_k` rather than a probability threshold.
- **Cells 3–4 unchanged** — the whitespace tokenisation fix (`msa_words`/`masri_words`)
  and the bigram computation stay as-is.

## Verification

End-to-end on the real datasets (Wikipedia MSA + EG tweets), running the notebook cells
as written:

- both subplots contain edges (MSA and Masri non-empty);
- edge widths are monotonic in probability (thickest edge = highest `p` per node);
- arrowheads and probability labels are present for every edge;
- `verify_notebooks.py` still passes (its mocked `Figure.show` applies to the
  `make_subplots` figure too).

## Out of scope

- Stage 1a (char) and Stage 1c (subword) notebooks — unchanged.
- The shared-seed "duel" and switchable single-tree layouts (considered, not chosen).
- RTL right→left tree flow (considered, rejected in favour of conventional flow).
- Any new plotting/graph dependency.
