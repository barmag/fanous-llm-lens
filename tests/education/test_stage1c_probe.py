"""Tests for the Stage 1c dialect-probe analysis cell.

We load the analysis cell (code-cell index 3) out of the reference notebook
and exec it against a tiny synthetic, linearly-separable W_E plus synthetic
id streams, so the test needs no network, no GPU, and no training. The cell
defines dialect_labels, probe_dialect, run_pca, plot_pca_vs_probe and a
driver that builds `labels`, `probe`, `fig`.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "education" / "stage1_c_subword_reference.ipynb"
)

# Synthetic vocabulary of 10 subword ids.
#   ids 0-3  -> MSA-only      (cluster near +x)
#   ids 6-9  -> Masri-only    (cluster near -x)
#   ids 4-5  -> Shared        (near origin)
_MSA_IDS = [0, 1, 2, 3] * 5 + [4, 5] * 3
_MASRI_IDS = [6, 7, 8, 9] * 5 + [4, 5] * 3

# Linearly separable embeddings: MSA at +5 on axis 0, Masri at -5, shared at 0.
_W_E = np.zeros((10, 4), dtype=float)
for _i in range(10):
    if _i <= 3:
        _W_E[_i] = [5.0 + 0.1 * _i, 0.1 * _i, 0.0, 0.0]
    elif _i >= 6:
        _W_E[_i] = [-5.0 - 0.1 * _i, 0.1 * _i, 0.0, 0.0]
    else:
        _W_E[_i] = [0.1 * _i, 0.1, 0.0, 0.0]

_ID_TO_TOKEN = {i: f"tok{i}" for i in range(10)}


def _load_analysis_cell():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # code cells: 0=install, 1=corpus+bpe, 2=train, 3=analysis
    src = "".join(code_cells[3]["source"])
    ns = {
        "W_E": _W_E,
        "msa_ids": _MSA_IDS,
        "masri_ids": _MASRI_IDS,
        "id_to_token": _ID_TO_TOKEN,
    }
    go.Figure.show = lambda self: None  # headless no-op
    exec(compile(src, f"{NB}:analysiscell", "exec"), ns)
    return ns


def test_dialect_labels_assigns_by_frequency():
    ns = _load_analysis_cell()
    labels = ns["dialect_labels"](_MSA_IDS, _MASRI_IDS, min_count=5)
    assert labels[0] == "MSA"      # appears only in MSA stream
    assert labels[9] == "Masri"    # appears only in Masri stream
    assert labels[4] == "Shared"   # 50/50 split across streams


def test_dialect_labels_drops_rare_tokens():
    ns = _load_analysis_cell()
    # token 7 appears 5 times (>= min_count 5); raise the floor above that.
    labels = ns["dialect_labels"](_MSA_IDS, _MASRI_IDS, min_count=100)
    assert labels == {}


def test_probe_separates_separable_embeddings():
    ns = _load_analysis_cell()
    labels = ns["dialect_labels"](_MSA_IDS, _MASRI_IDS, min_count=5)
    probe = ns["probe_dialect"](_W_E, labels, seed=0)
    assert probe["accuracy"] >= 0.8
    assert probe["direction"].shape == (_W_E.shape[1],)
    # every labelled token gets a scalar projection
    assert set(probe["projections"]) == set(labels)


def test_run_pca_returns_2d():
    ns = _load_analysis_cell()
    coords = ns["run_pca"](_W_E, [0, 1, 2, 6, 7, 8])
    assert coords.shape == (6, 2)


def test_figure_has_two_panels_with_accuracy():
    ns = _load_analysis_cell()
    fig = ns["fig"]
    assert isinstance(fig, go.Figure)
    # make_subplots(cols=2) creates a second x-axis
    assert fig.layout.xaxis2 is not None
    # both a PCA scatter and a probe histogram are drawn
    assert any(t.type == "scatter" for t in fig.data)
    assert any(t.type == "histogram" for t in fig.data)
    # the probe accuracy is surfaced somewhere in the figure text
    ann_texts = [a.text or "" for a in fig.layout.annotations]
    assert any("acc" in t.lower() for t in ann_texts)
