"""Unit tests for the library cells of the Toy Models of Superposition notebook.

We exec the `# lib:`-marked code cells straight out of the notebook into a fresh
namespace, so the tests need no GPU and (except one tiny convergence check) no training.
"""
import json
from pathlib import Path

import torch

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "superposition" / "toy_models_of_superposition.ipynb"
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


def test_make_batch_shape_and_range():
    lib = load_lib()
    gen = torch.Generator().manual_seed(0)
    x = lib["make_batch"](n_features=6, sparsity=0.5, batch_size=100, generator=gen)
    assert x.shape == (100, 6)
    assert x.min().item() >= 0.0
    assert x.max().item() < 1.0


def test_make_batch_sparsity_fraction():
    lib = load_lib()
    gen = torch.Generator().manual_seed(0)
    x = lib["make_batch"](n_features=10, sparsity=0.9, batch_size=20_000, generator=gen)
    frac_nonzero = (x > 0).float().mean().item()
    assert abs(frac_nonzero - 0.1) < 0.02  # ~10% survive at S=0.9


def test_importance_decay():
    lib = load_lib()
    imp = lib["importance_weights"](5, decay=0.7)
    assert imp.shape == (5,)
    assert abs(imp[0].item() - 1.0) < 1e-6
    assert abs(imp[3].item() - 0.7 ** 3) < 1e-6


def test_toymodel_param_shapes():
    lib = load_lib()
    model = lib["ToyModel"](n_features=5, n_hidden=2)
    assert tuple(model.W.shape) == (2, 5)
    assert tuple(model.b.shape) == (5,)


def test_toymodel_forward_shape():
    lib = load_lib()
    model = lib["ToyModel"](n_features=5, n_hidden=2)
    x = torch.rand(7, 5)
    assert tuple(model(x).shape) == (7, 5)


def test_relu_is_nonnegative_linear_can_be_negative():
    lib = load_lib()
    torch.manual_seed(0)
    x = torch.rand(32, 5)
    relu_model = lib["ToyModel"](5, 2, use_relu=True)
    lin_model = lib["ToyModel"](5, 2, use_relu=False)
    # force a negative bias so the pre-activation has negative entries
    with torch.no_grad():
        relu_model.b.fill_(-1.0)
        lin_model.b.copy_(relu_model.b)
        lin_model.W.copy_(relu_model.W)
    assert relu_model(x).min().item() >= 0.0
    assert lin_model(x).min().item() < 0.0


def test_train_reduces_loss():
    lib = load_lib()
    torch.manual_seed(0)
    model = lib["ToyModel"](3, 2, use_relu=True)
    imp = lib["importance_weights"](3)
    losses = lib["train"](model, sparsity=0.0, importance=imp, steps=1000, seed=0)
    assert losses[-1] < losses[0]


def test_feature_norms_shape():
    lib = load_lib()
    model = lib["ToyModel"](5, 2)
    fn = lib["feature_norms"](model)
    assert fn.shape == (5,)
    assert (fn >= 0).all()
