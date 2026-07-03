"""Visualization + example-picking helpers for stage2c_induction_tinystories.

Pure functions over numpy arrays and injected callables, so everything here
is testable on CPU with fakes — no model, no tokenizer, no GPU.

Paper anchors:
- attn_heatmap renders the attention pattern A = softmax(q·k/sqrt(d)) that
  Elhage et al. 2021 call the QK circuit's output ("where to look").
- prefix_matching_score is the Olsson et al. 2022 prefix-matching criterion:
  a head is induction-like when, at a repeated token, it attends to the token
  AFTER the previous occurrence.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

HIGHLIGHT = "#e65100"


def attn_heatmap(pattern, tokens, highlight=None, title=""):
    """One attention pattern as a heatmap with decoded-token axis labels.

    pattern: [S, S] array-like, rows = query (destination), cols = key (source).
    tokens: S decoded token strings (duplicates fine — ticks are positional).
    highlight: optional list of (query, key) cells to outline.
    """
    pattern = np.asarray(pattern)
    S = pattern.shape[0]
    if pattern.shape != (S, S) or len(tokens) != S:
        raise ValueError(f"pattern {pattern.shape} needs matching tokens, got {len(tokens)}")
    hover = [
        [
            f"query {tokens[q]!r} (pos {q}) ← key {tokens[k]!r} (pos {k}): {pattern[q, k]:.2f}"
            for k in range(S)
        ]
        for q in range(S)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=pattern,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            showscale=False,
            text=hover,
            hoverinfo="text",
        )
    )
    idx = list(range(S))
    tickfont = dict(size=9, family="monospace")
    fig.update_xaxes(
        tickvals=idx,
        ticktext=list(tokens),
        tickangle=45,
        tickfont=tickfont,
        title_text="key (source) — the token being looked AT",
    )
    fig.update_yaxes(
        tickvals=idx,
        ticktext=list(tokens),
        autorange="reversed",
        tickfont=tickfont,
        title_text="query (destination) — the token doing the looking",
    )
    for q, k in highlight or []:
        fig.add_shape(
            type="rect",
            x0=k - 0.5,
            x1=k + 0.5,
            y0=q - 0.5,
            y1=q + 0.5,
            line=dict(color=HIGHLIGHT, width=2),
        )
    side = max(420, 24 * S + 170)
    fig.update_layout(
        title=title,
        width=side,
        height=side,
        margin=dict(l=90, r=30, t=60, b=90),
    )
    return fig


@dataclass
class Example:
    """The running example carried through panels 1–4."""

    prompt: str
    ids: list
    tokens: list
    query_pos: int  # position of the last token (2nd occurrence of A)
    key_pos: int  # position of B1 = the token after the 1st occurrence of A
    target_id: int
    target_str: str
    topk: list  # [(token_str, prob)] top-5 at query_pos


def find_induction_anchor(ids):
    """Find the induction anchor for a prompt ending in a repeated token.

    Returns (first_pos, follower_pos) for the FIRST earlier occurrence of the
    final token, or None if the final token never appeared before (or its
    follower would be the final position itself, leaving nothing to predict).
    """
    last = ids[-1]
    for s in range(len(ids) - 1):
        if ids[s] == last and s + 1 < len(ids) - 1:
            return s, s + 1
    return None


def pick_example(candidates, encode, decode_token, topk5, top_n=3):
    """Pick the first candidate prompt the model actually completes.

    candidates: prompt strings ending on the 2nd occurrence of some token A.
    encode: str -> list[int].  decode_token: int -> str.
    topk5: list[int] -> [(token_id, prob)] descending, at least 5 entries.
    Accepts the first prompt whose induction target lands in the model's
    top-`top_n`. Raises ValueError naming every rejected prompt otherwise —
    never silently falls back.
    """
    rejected = []
    for prompt in candidates:
        ids = list(encode(prompt))
        anchor = find_induction_anchor(ids)
        if anchor is None:
            rejected.append((prompt, "last token never appears earlier"))
            continue
        _s, follower = anchor
        target_id = ids[follower]
        preds = topk5(ids)
        if target_id not in [tid for tid, _ in preds[:top_n]]:
            rejected.append((prompt, f"target {decode_token(target_id)!r} not in top-{top_n}"))
            continue
        return Example(
            prompt=prompt,
            ids=ids,
            tokens=[decode_token(t) for t in ids],
            query_pos=len(ids) - 1,
            key_pos=follower,
            target_id=target_id,
            target_str=decode_token(target_id),
            topk=[(decode_token(t), p) for t, p in preds],
        )
    lines = "\n".join(f"  {p!r}: {why}" for p, why in rejected)
    raise ValueError(f"no candidate prompt passed the picker:\n{lines}")


def diag_score(pattern):
    """Prev-token strength: mean attention from position i to i-1."""
    p = np.asarray(pattern)
    S = p.shape[0]
    return float(sum(p[i, i - 1] for i in range(1, S)) / max(S - 1, 1))


def prefix_matching_score(pattern, ids):
    """Olsson et al. prefix-matching score on a real token sequence.

    For every position t whose token appeared earlier (most recent earlier
    occurrence s, with s+1 < t), average the attention pattern[t, s+1].
    Returns 0.0 when the sequence has no usable repeats.
    """
    p = np.asarray(pattern)
    tot, cnt = 0.0, 0
    for t in range(len(ids)):
        prev = [s for s in range(t) if ids[s] == ids[t] and s + 1 < t]
        if not prev:
            continue
        s = prev[-1]
        tot += float(p[t, s + 1])
        cnt += 1
    return tot / cnt if cnt else 0.0
