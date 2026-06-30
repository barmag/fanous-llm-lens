"""Honest skip-trigram analysis for one-layer attention-only models.

Pure, testable helpers used by the Stage 2dash skip-trigram notebooks. Mirrors the
decomposition in *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021):
each head is a skip-trigram table built from a QK circuit (which source to attend to) and an
OV circuit (what the attended source promotes). Nothing here touches the network or disk.
"""

from __future__ import annotations

import html

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


def seed_ids(encode, id_to_str, seed_words, freq: int = 2500) -> list[int]:  # noqa: ARG001
    """Resolve seed words/tokens to in-vocab, frequent token ids (deduped, order-stable).

    `id_to_str` is unused here (resolution goes through `encode`) but kept in the signature
    for symmetry with the other notebook-facing helpers, which all take (encode, id_to_str).
    """
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
    per_source_outputs: int = 1,
    per_source_dests: int = 1,
) -> list[dict]:
    """Rank skip-trigram triples (source, dest, output) for one head by a composite score.

    score = OV[source, output] * QK[dest, source], over frequent tokens. For each source we
    enumerate its `per_source_outputs` strongest promoted outputs (off-diagonal unless
    include_self_copy) crossed with the `per_source_dests` destinations that most strongly
    route attention to it — so a handful of seed sources expands into a large candidate pool
    (5 seeds * 5 outputs * 4 dests = 100). `sources` (a set of ids) makes this a seeded pool;
    None scans all frequent sources (unsupervised). Defaults (1, 1) keep the one-per-source
    behaviour. Returns the top_n by score.
    """
    QK, OV = head_circuits(model, head)
    QK, OV = QK[:freq, :freq], OV[:freq, :freq]
    src_iter = range(freq) if sources is None else sorted(s for s in sources if s < freq)
    out = []
    for s in src_iter:
        ov_row = OV[s].clone()
        if not include_self_copy:
            ov_row[s] = float("-inf")  # forbid self-copy
        k_out = min(per_source_outputs, ov_row.numel())
        o_vals, o_idx = torch.topk(ov_row, k_out)
        qk_col = QK[:, s]  # destinations routing attention to s
        k_dst = min(per_source_dests, qk_col.numel())
        d_vals, d_idx = torch.topk(qk_col, k_dst)
        for ov, o in zip(o_vals.tolist(), o_idx.tolist(), strict=True):
            if ov == float("-inf"):
                continue  # only self-copy left for this source
            for qk, d in zip(d_vals.tolist(), d_idx.tolist(), strict=True):
                out.append(
                    {
                        "source": s,
                        "dest": int(d),
                        "output": int(o),
                        "ov": float(ov),
                        "qk": float(qk),
                        "score": float(ov) * float(qk),
                    }
                )
    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:top_n]


def dedup_triples(rows: list[dict], *, keep: str = "score") -> list[dict]:
    """Drop duplicate (source, dest, output) triples, keeping the highest-`keep` instance.

    The full model is what `verify_triple` runs, so the same triple proposed by two heads has
    one true lift; dedup before verifying (or after) to avoid double-counting. Order-stable on
    first appearance.
    """
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    for r in rows:
        k = (r["source"], r["dest"], r["output"])
        if k not in best:
            best[k] = r
            order.append(k)
        elif r.get(keep, float("-inf")) > best[k].get(keep, float("-inf")):
            best[k] = r
    return [best[k] for k in order]


def top_per_group(
    rows: list[dict], *, key: str = "group", n: int = 1, by: str = "lift"
) -> list[dict]:
    """Highest-`by` n rows per distinct `key` value, groups in first-appearance order.

    Used to pick the most-representative verified triple(s) per category for the punchline.
    """
    groups: dict = {}
    order: list = []
    for r in rows:
        g = r.get(key)
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(r)
    out = []
    for g in order:
        ranked = sorted(groups[g], key=lambda r: r.get(by, float("-inf")), reverse=True)
        out.extend(ranked[:n])
    return out


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


def _triple_expr(row: dict, id_to_str) -> str:
    """`[source … dest] → output` with tokens resolved to strings (HTML-escaped)."""
    s = html.escape(str(id_to_str.get(row["source"], row["source"])))
    d = html.escape(str(id_to_str.get(row["dest"], row["dest"])))
    o = html.escape(str(id_to_str.get(row["output"], row["output"])))
    # dir=ltr keeps the [source … dest] → output ordering even for Arabic tokens (paper form).
    return f'<span dir="ltr" style="font-family:monospace">[{s} … {d}] → {o}</span>'


def triple_table_html(
    rows: list[dict],
    id_to_str,
    *,
    title: str = "",
    note: str = "",
    empty_msg: str = "لا توجد ثلاثيات اتأكّدت · nothing verified",
) -> str:
    """Render verified skip-trigram triples as a paper-style RTL HTML table.

    Each row needs source/dest/output; optional group, head, lift, gloss columns render as
    "—" when absent. Returns a string (caller wraps in IPython.display.HTML), so this stays
    importable and unit-testable with no notebook/display dependency. Empty `rows` yields a
    small panel with `empty_msg` instead of crashing — the CI/tiny-model path may verify none.
    """
    head = (
        f'<div dir="rtl" style="font-weight:600;margin:6px 0">{html.escape(title)}</div>'
        if title
        else ""
    )
    if not rows:
        return f'{head}<div dir="rtl" style="color:#888;padding:6px">{html.escape(empty_msg)}</div>'
    cols = [
        ("التصنيف · category", "group"),
        ("رأس · head", "head"),
        ("الثلاثي · skip-trigram", "_expr"),
        ("الرفع · lift", "lift"),
        ("التفسير · what it encodes", "gloss"),
    ]
    th = "".join(
        f'<th style="text-align:right;padding:4px 10px;border-bottom:1px solid #ccc">{html.escape(h)}</th>'
        for h, _ in cols
    )
    body = []
    for r in rows:
        cells = []
        for _, k in cols:
            if k == "_expr":
                v = _triple_expr(r, id_to_str)
            elif k == "lift":
                v = f"{r['lift']:+.3f}" if "lift" in r else "—"
            else:
                raw = r.get(k)
                v = html.escape(str(raw)) if raw not in (None, "") else "—"
            cells.append(f'<td style="text-align:right;padding:4px 10px">{v}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    foot = (
        f'<div dir="rtl" style="color:#888;font-size:90%;margin-top:4px">{html.escape(note)}</div>'
        if note
        else ""
    )
    return (
        f"{head}<table dir='rtl' style='border-collapse:collapse;font-size:95%'>"
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>{foot}"
    )
