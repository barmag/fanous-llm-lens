"""Unit tests for the library cells of the Superposition Geometry notebook (book two).

Same convention as test_toy_models.py: exec the `# lib:`-marked code cells straight out
of the notebook into a fresh namespace. No GPU needed.
"""
import json
from pathlib import Path

import torch

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "superposition" / "superposition_geometry.ipynb"
)


def load_lib(nb_path=NB):
    """Exec every `# lib:`-marked code cell, in order, into one namespace."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith("# lib:"):
            exec(compile(src, f"{nb_path.name}:{cell.get('id', '')}", "exec"), ns)
    return ns


def test_lib_cells_exec_cleanly():
    ns = load_lib()
    assert ns["SEED"] == 0
    assert "torch" in ns


def test_make_batch_shape_and_sparsity():
    ns = load_lib()
    gen = torch.Generator().manual_seed(0)
    x = ns["make_batch"](10, 0.9, 4096, generator=gen)
    assert x.shape == (4096, 10)
    frac_zero = (x == 0).float().mean().item()
    assert 0.88 < frac_zero < 0.92


def test_toymodel_forward_shape():
    ns = load_lib()
    mdl = ns["ToyModel"](5, 2)
    out = mdl(torch.rand(7, 5))
    assert out.shape == (7, 5)
    assert (out >= 0).all()  # ReLU output


def test_train_reduces_loss():
    ns = load_lib()
    torch.manual_seed(0)
    mdl = ns["ToyModel"](5, 2)
    losses = ns["train"](mdl, sparsity=0.0, importance=torch.ones(5), steps=600)
    assert losses[-1] < losses[0]


def _pentagon_W():
    ang = 2 * torch.pi * torch.arange(5) / 5
    return torch.stack([torch.cos(ang), torch.sin(ang)])  # [2, 5], unit columns


def test_dimensionality_identity_is_one():
    ns = load_lib()
    D = ns["feature_dimensionality"](torch.eye(4))
    assert torch.allclose(D, torch.ones(4), atol=1e-5)


def test_dimensionality_antipodal_is_half():
    ns = load_lib()
    W = torch.tensor([[1.0, -1.0]])  # one dim, two antipodal features
    D = ns["feature_dimensionality"](W)
    assert torch.allclose(D, torch.full((2,), 0.5), atol=1e-5)


def test_dimensionality_pentagon_is_two_fifths():
    ns = load_lib()
    D = ns["feature_dimensionality"](_pentagon_W())
    assert torch.allclose(D, torch.full((5,), 0.4), atol=1e-4)


def test_dimensionality_zero_column_is_zero():
    ns = load_lib()
    W = torch.eye(3)
    W[:, 2] = 0.0
    D = ns["feature_dimensionality"](W)
    assert D[2].item() == 0.0
    assert torch.allclose(D[:2], torch.ones(2), atol=1e-5)


def test_frobenius_dims_per_feature():
    ns = load_lib()
    assert abs(ns["frobenius_dims_per_feature"](torch.eye(4)) - 1.0) < 1e-6
    # pentagon: 5 unit-norm features in 2 dims -> 2/5 dims per feature
    assert abs(ns["frobenius_dims_per_feature"](_pentagon_W()) - 0.4) < 1e-5
