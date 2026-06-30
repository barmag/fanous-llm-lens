"""Honest skip-trigram analysis for one-layer attention-only models.

Pure, testable helpers used by the Stage 2dash skip-trigram notebooks. Mirrors the
decomposition in *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021):
each head is a skip-trigram table built from a QK circuit (which source to attend to) and an
OV circuit (what the attended source promotes). Nothing here touches the network or disk.
"""

from __future__ import annotations

import torch


def head_circuits(model, h: int):
    """(QK, OV) for head h. QK is (V,V) dst x src; OV is (V,V) src x out. CPU tensors."""
    W_E = model.W_E.detach().cpu()
    W_U = model.W_U.detach().cpu()
    W_Q = model.W_Q[0, h].detach().cpu()
    W_K = model.W_K[0, h].detach().cpu()
    W_V = model.W_V[0, h].detach().cpu()
    W_O = model.W_O[0, h].detach().cpu()
    QK = W_E @ W_Q @ W_K.T @ W_E.T  # dst x src
    OV = W_E @ W_V @ W_O @ W_U  # src x out
    return QK, OV


def head_attention_kind(pattern, *, prev_bias: float = 0.5, bos_bias: float = 0.5) -> str:
    """Classify a single head's (seq, seq) attention matrix.

    Returns "bos" (mass on position 0), "prev_token" (mass on the diagonal-1 band), or
    "content" (neither dominates -> candidate for content-based long-range skip-trigrams).
    Rows are destinations; only the causal lower triangle carries mass.
    """
    p = torch.as_tensor(pattern, dtype=torch.float32)
    n = p.shape[0]
    if n < 2:
        return "content"
    rows = range(1, n)  # row 0 can only attend to itself; skip it
    bos = sum(float(p[i, 0]) for i in rows) / len(rows)
    prev = sum(float(p[i, i - 1]) for i in rows) / len(rows)
    if bos >= bos_bias and bos >= prev:
        return "bos"
    if prev >= prev_bias:
        return "prev_token"
    return "content"
