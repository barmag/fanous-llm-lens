"""One-time generator: the embeddings-only (context-blind) prediction baseline PNG.

A zero-layer embed->unembed model predicts the next token from the LAST token
only (no attention mixes the prefix), so two sentences ending in the same word
get IDENTICAL prediction trees. Rendered with plotly (Arabic-correct) -> PNG,
committed so the Stage 2 notebooks show the baseline without retraining.

Run from the repo root with the `assets` extra installed (for kaleido):
    uv sync --extra cpu --extra dev --extra assets   # or --extra rocm ...
    python notebooks/education/_generate_context_blind_image.py
"""
import re
import sys

import plotly.graph_objects as go
import torch
from datasets import load_dataset
from plotly.subplots import make_subplots
from transformers import AutoTokenizer

sys.path.insert(0, "notebooks/education")
import tiny

torch.manual_seed(0)

MAX_CHARS = 120_000
msa_stream = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
tweets = load_dataset("amgadhasan/arabic_tweets_dialects", split="train").filter(
    lambda x: x["dialect"] == "EG"
)


def clean(t):
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[a-zA-Z0-9_@]+", "", t)
    return re.sub(r"[^\sء-ي]", "", t)


def collect(rows, key, n):
    out = ""
    for r in rows:
        out += clean(r[key]) + " "
        if len(out) >= n:
            break
    return out[:n]


tok = AutoTokenizer.from_pretrained("ai-forever/mGPT")
encode, corpus_ids, id_to_str, V = tiny.make_compact_encoder(
    tok, [collect(msa_stream, "text", MAX_CHARS), collect(tweets, "text", MAX_CHARS)]
)
ids = corpus_ids[0] + corpus_ids[1]

# zero-layer embed -> unembed (bigram) model
D = 64
W_E = (torch.randn(V, D) * 0.1).requires_grad_(True)
W_U = (torch.randn(D, V) * 0.1).requires_grad_(True)
cur, nxt = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
opt = torch.optim.AdamW([W_E, W_U], lr=5e-2)
for _ in range(250):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(W_E[cur] @ W_U, nxt)
    loss.backward()
    opt.step()
print("blind model loss:", round(float(loss), 3))

LOW_PROB = 0.10


def blind_tree(seed_text, max_depth=2, top_k=2):
    """Context-blind rollout: each step depends only on the last token."""
    edges, nodes = [], {"root": (seed_text, True)}
    frontier, n = [("root", encode(seed_text)[-1], 0)], 0
    while frontier:
        pkey, last_id, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        probs = torch.softmax(W_E[last_id] @ W_U, dim=-1)
        vals, idx = torch.topk(probs, top_k)
        for v, i in zip(vals, idx, strict=False):
            n += 1
            tid, p = int(i), float(v)
            label = id_to_str.get(tid, "[UNK]")
            sensible = label != "[UNK]" and p >= LOW_PROB
            ckey = f"{depth}:{n}:{label}"
            nodes[ckey] = (label, sensible)
            edges.append((pkey, ckey, p, depth, sensible))
            frontier.append((ckey, tid, depth + 1))
    return edges, nodes


def layout(edges):
    depth_of = {"root": 0}
    for _pk, ck, _p, d, _s in edges:
        depth_of[ck] = d + 1
    levels = {}
    for k, d in depth_of.items():
        levels.setdefault(d, []).append(k)
    pos = {}
    for d, keys in levels.items():
        for i, k in enumerate(keys):
            pos[k] = (d, (len(keys) - 1) / 2 - i)
    return pos


def add_tree(fig, col, edges, nodes):
    suffix = "" if col == 1 else str(col)
    xref, yref = f"x{suffix}", f"y{suffix}"
    pos = layout(edges)
    for pk, ck, p, _d, s in edges:
        (x0, y0), (x1, y1) = pos[pk], pos[ck]
        col_ = "#c0392b" if not s else "#2b6cb0"
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref=xref, yref=yref, axref=xref,
                           ayref=yref, showarrow=True, arrowhead=3, arrowwidth=1.5 + p * 7.5,
                           arrowcolor=col_, standoff=18, startstandoff=24, text="")
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, xref=xref, yref=yref,
                           showarrow=False, text=f"{p:.2f}", font=dict(size=11, color="#333"),
                           bgcolor="rgba(255,255,255,0.8)")
    keys = list(pos.keys())
    fig.add_trace(go.Scatter(
        x=[pos[k][0] for k in keys], y=[pos[k][1] for k in keys], mode="markers+text",
        text=[nodes[k][0] for k in keys], textposition="middle center",
        textfont=dict(size=15, color="#111"),
        marker=dict(color=["#f6c350" if k == "root" else ("#f6c0bb" if not nodes[k][1] else "#d9d9d9")
                           for k in keys],
                    size=[44 if k == "root" else 32 for k in keys], line=dict(width=1.5, color="#333")),
        hoverinfo="text", showlegend=False), row=1, col=col)
    fig.update_xaxes(visible=False, row=1, col=col)
    fig.update_yaxes(visible=False, row=1, col=col)


CTX_A, CTX_B = "القطة بتاكل السمك", "الولد بياكل السمك"  # different prefix, same last word
ea, na = blind_tree(CTX_A)
eb, nb = blind_tree(CTX_B)
def child_labels(nodes):
    return [v[0] for k, v in nodes.items() if k != "root"]


print("prediction children identical (context-blind):", child_labels(na) == child_labels(nb))

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                    subplot_titles=(f"السياق: {CTX_A}", f"السياق: {CTX_B}"))
add_tree(fig, 1, ea, na)
add_tree(fig, 2, eb, nb)
fig.update_layout(
    height=420, width=1100, margin=dict(l=20, r=20, t=90, b=30),
    title_text="نموذج التضمينات (المرحلة ١ج): نفس التوقع للسياقين — أعمى للسياق · "
               "embeddings-only is context-blind (prefix ignored)",
    title_font=dict(size=14),
)
out = "notebooks/education/images/embeddings_context_blind.png"
fig.write_image(out, scale=2)
print("wrote", out)
