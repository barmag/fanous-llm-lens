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
