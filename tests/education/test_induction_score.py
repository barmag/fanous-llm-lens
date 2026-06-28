"""Unit tests for the shared induction-score helpers in tiny.py.

The pure helper is tested against a hand-built attention pattern where the
answer is known; the model-level helper is tested for shape/range on a tiny
CPU 2-layer model (no GPU, no network, no training)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "education"))
import tiny


def test_pattern_helper_scores_perfect_induction_high_and_uniform_low():
    seq_len = 4
    n_pos = 2 * seq_len
    pattern = torch.zeros(1, 2, n_pos, n_pos)
    # Head 0: perfect induction stripe — query (seq_len+i) attends to key (i+1),
    # i.e. pattern[k, k - (seq_len - 1)] == 1 for the second-half queries.
    for k in range(seq_len - 1, n_pos):
        pattern[0, 0, k, k - (seq_len - 1)] = 1.0
    # Head 1: uniform attention (no induction).
    pattern[0, 1] = 1.0 / n_pos

    scores = tiny.induction_score_from_pattern(pattern, seq_len)

    assert scores.shape == (2,)
    assert scores[0] > 0.99
    assert scores[1] < 0.30


def test_model_helper_shape_and_range():
    model = tiny.make_tiny_model(
        n_layers=2, n_heads=4, d_vocab=64, n_ctx=32, d_model=64,
        attn_only=True, normalization_type=None,
        positional_embedding_type="shortformer",
    )
    scores = tiny.induction_scores(model, seq_len=8, n_seqs=2, seed=0)
    assert scores.shape == (2, 4)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_gate_passes_when_a_head_crosses_threshold():
    import torch
    scores = torch.tensor([[0.05, 0.02, 0.61, 0.10], [0.08, 0.03, 0.04, 0.07]])
    passed, layer, head = tiny.induction_gate(scores, threshold=0.4)
    assert passed is True
    assert (layer, head) == (0, 2)


def test_gate_fails_when_no_head_crosses_threshold():
    import torch
    scores = torch.tensor([[0.05, 0.02], [0.08, 0.03]])
    passed, layer, head = tiny.induction_gate(scores, threshold=0.4)
    assert passed is False
    assert (layer, head) == (1, 0)  # strongest head is at layer 1, head 0 (score=0.08)
