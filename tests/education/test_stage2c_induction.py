"""Unit test for Stage 2c's induction metric.

Loads `induction_from_patterns` out of the reference notebook and feeds it
synthetic attention patterns where the answer is known: a single head in layer 1
does perfect induction. The metric must (a) score that head highest and (b)
locate it in layer 1 — and must use the correct offset (attention to src+1, not
src), which is the off-by-one this guards against. Fast, CPU-only, no training.
"""
import json
from pathlib import Path

import torch

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "education" / "stage2_c_depth_induction_reference.ipynb"
)


def _load_fn(name):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code" and f"def {name}" in "".join(c["source"]):
            ns = {"torch": torch}
            exec("".join(c["source"]), ns)  # trusted: our own notebook cell
            return ns[name]
    raise AssertionError(f"no cell defines {name}")


def _patterns(induction_layer, induction_head):
    """Two layers, two heads, one sequence. Token at t=4 repeats token at pos 1,
    so its copy source is pos 1 and induction should attend to src+1 = 2."""
    n_layers, n_heads, S = 2, 2, 6
    src = torch.full((1, S), -1)
    src[0, 4] = 1  # position 4 is a repeat of position 1
    pats = [torch.full((1, n_heads, S, S), 0.05) for _ in range(n_layers)]
    # the designated head attends position 4 -> position 2 (== src+1): induction
    pats[induction_layer][0, induction_head, 4, 2] = 0.95
    return pats, src


def test_induction_metric_finds_layer1_head():
    fn = _load_fn("induction_from_patterns")
    pats, src = _patterns(induction_layer=1, induction_head=0)
    scores = fn(pats, src)
    assert tuple(scores.shape) == (2, 2)
    best_layer = int(scores.flatten().argmax()) // 2
    assert best_layer == 1
    assert float(scores[1, 0]) > float(scores[0, 0])  # induction head scored highest


def test_induction_metric_offset_is_src_plus_one():
    """If the head attended to src itself (off-by-one), the score must be low."""
    fn = _load_fn("induction_from_patterns")
    n_layers, n_heads, S = 2, 2, 6
    src = torch.full((1, S), -1)
    src[0, 4] = 1
    pats = [torch.full((1, n_heads, S, S), 0.0) for _ in range(n_layers)]
    pats[1][0, 0, 4, 1] = 1.0  # attends to src (1), NOT src+1 (2)
    scores = fn(pats, src)
    assert float(scores[1, 0]) == 0.0  # correct metric ignores attention to src itself
