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


def seed_ids(encode, id_to_str, seed_words, freq: int = 2500) -> list[int]:
    """Resolve seed words/tokens to in-vocab, frequent token ids (deduped, order-stable)."""
    out, seen = [], set()
    for w in seed_words:
        for tid in encode(w):
            if tid < freq and tid not in seen:
                seen.add(tid)
                out.append(int(tid))
    return out


def candidate_pool(
    model,
    *,
    head: int,
    freq: int = 2500,
    include_self_copy: bool = False,
    top_n: int = 100,
    sources=None,
) -> list[dict]:
    """Rank skip-trigram triples (source, dest, output) for one head by a composite score.

    score = OV[source, output] * QK[dest, source], over frequent tokens. For each source we
    take its single best output (off-diagonal unless include_self_copy) and the destination
    that most strongly routes attention to it. `sources` (a set of ids) makes this a seeded
    pool; None scans all frequent sources (unsupervised).
    """
    QK, OV = head_circuits(model, head)
    QK, OV = QK[:freq, :freq], OV[:freq, :freq]
    src_iter = range(freq) if sources is None else sorted(s for s in sources if s < freq)
    out = []
    for s in src_iter:
        ov_row = OV[s].clone()
        if not include_self_copy:
            ov_row[s] = float("-inf")  # forbid self-copy
        o = int(torch.argmax(ov_row))
        ov = float(ov_row[o])
        if ov == float("-inf"):
            continue
        d = int(torch.argmax(QK[:, s]))  # destination routing attention to s
        qk = float(QK[d, s])
        out.append({"source": s, "dest": d, "output": o, "ov": ov, "qk": qk, "score": ov * qk})
    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:top_n]


def verify_triple(model, triple: dict, *, n_ctx: int | None = None) -> dict:
    """Does the full model raise P(output) at the dest position above the bigram baseline?

    Builds [source, source, ..., dest] (source repeated to fill context so attention has a
    real earlier token to find), runs the forward pass with cache, and at the final (dest)
    position compares softmax(full logits)[output] vs softmax(direct-path logits)[output].
    The direct path is the context-blind bigram: resid_pre @ W_U + b_U.
    """
    device = next(model.parameters()).device
    ctx = min(8, model.cfg.n_ctx) if n_ctx is None else n_ctx
    s, d, o = triple["source"], triple["dest"], triple["output"]
    seq = [s] * (ctx - 1) + [d]
    ids = torch.tensor([seq], device=device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(ids)
        direct = cache["resid_pre", 0] @ model.W_U + model.b_U
        p_full = float(torch.softmax(logits[0, -1], -1)[o])
        p_bigram = float(torch.softmax(direct[0, -1], -1)[o])
    lift = p_full - p_bigram
    return {**triple, "p_full": p_full, "p_bigram": p_bigram, "lift": lift, "verified": lift > 0}


def verify_pool(model, pool: list[dict], *, top_k: int = 20) -> list[dict]:
    """Verify the top_k candidates of a pool on held-out forward passes."""
    return [verify_triple(model, c) for c in pool[:top_k]]
