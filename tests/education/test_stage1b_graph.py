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
