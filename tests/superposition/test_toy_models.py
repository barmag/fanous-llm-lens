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
