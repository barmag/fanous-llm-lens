# Induction Heads in the Wild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` — a reference notebook that hunts the blog post's hand-crafted induction circuit (shift-QK prev-token head → K-composed copying induction head) inside pre-trained GPT-2 small and Pythia-160m, and reports job-by-job whether training found the same solution.

**Architecture:** One notebook, five acts, built act-by-act. Each act's code is first validated in a throwaway dev script (mechanics assertions only — shapes, finiteness, ranges; **never** result gates), then appended to the notebook as cells via `nbformat`, then the whole notebook is re-executed in place so outputs stay baked. Cross-act state lives in notebook globals with exact names defined in each task's Interfaces block.

**Tech Stack:** transformer_lens 2.15.4 (`HookedTransformer`, `FactoredMatrix`, `utils.composition_scores`), torch 2.5.1+rocm, matplotlib, pandas, nbformat. Spec: `docs/superpowers/specs/2026-07-15-induction-heads-in-the-wild-design.md`.

## Global Constraints

- **Branch:** all work on `induction-heads-in-the-wild` (already created off `main`). Never commit to `main`. Do not merge without user confirmation.
- **Runtime bar:** the notebook runs end-to-end in <10 min on the iGPU (both models are already in the HF cache; expect ~2–4 min total).
- **No new dependencies.** Everything needed is installed. Always run Python via `uv run --no-sync python …` and jupyter via `uv run --no-sync jupyter …` (bare `uv run` would destroy the ROCm venv).
- **Honest negatives:** the notebook itself contains **no pass/fail gates** — every metric is computed, printed, and reported whatever its value. Assertions are allowed only in dev scripts and only for mechanics (shape/dtype/finiteness), never for result magnitudes.
- **Outputs baked:** the committed notebook contains executed outputs (follows `9b7685c` precedent). Re-execute in place before every commit.
- **No Arabic content** in this notebook (blog-companion anatomy; dialect track unaffected).
- **No process-talk in notebook markdown** — markdown cells are pedagogical only; never mention CLAUDE.md, repo conventions, tasks, or this plan.
- **Notebook execution discipline:** run `nbconvert --execute --inplace` synchronously in the foreground and never edit the .ipynb while an execution is running (last-writer-wins clobbering).
- **Commit messages name the result**, not the change (e.g. "GPT-2 prev-token head shows the shift stripe at X% mass", not "add act 2a").
- **Dev scripts** go in the scratchpad directory `/tmp/claude-1000/-home-yassermakram-code-fanous-llm-lens/263973c2-4255-4472-9f52-bd9fe9c0af1a/scratchpad/` — never committed.
- **GPU env:** every dev script and the notebook setup cell begin with `os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")` **before** importing torch.

## Shared mechanics (referenced by every task)

**Appending cells.** Each task appends cells with a one-off script following this exact pattern (adjust `CELLS`; `md`/`code` are the two cell kinds):

```python
# append_cells_actN.py  (run from repo root: uv run --no-sync python <scratchpad>/append_cells_actN.py)
import nbformat

PATH = "notebooks/in_context_learning/induction_heads_in_the_wild.ipynb"
nb = nbformat.read(PATH, as_version=4)

CELLS = [
    ("md", "## Section title\n\ntext..."),
    ("code", "print('hello')"),
]
for kind, src in CELLS:
    cell = nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
    nb.cells.append(cell)

nbformat.write(nb, PATH)
print(f"now {len(nb.cells)} cells")
```

**Executing the notebook (bakes outputs):**

```bash
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
```

Expected: exits 0. Then spot-check the new outputs:

```bash
uv run --no-sync python -c "
import nbformat
nb = nbformat.read('notebooks/in_context_learning/induction_heads_in_the_wild.ipynb', as_version=4)
for c in nb.cells:
    if c.cell_type == 'code':
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                raise SystemExit('CELL ERROR: ' + '\n'.join(o.get('traceback', [])))
            if 'text' in o:
                print(o['text'][:400])
print('--- no cell errors ---')
"
```

---

### Task 1: Notebook scaffold + Act 0 (the toy, re-run)

**Files:**
- Create: `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb`

**Interfaces:**
- Produces notebook globals for all later tasks: `SEED = 42`, `device` (str, `"cuda"` or `"cpu"`), plus the toy's `VOCAB`, `predict_next` (self-contained; later acts only reference the toy conceptually).

- [ ] **Step 1: Confirm branch**

Run: `git branch --show-current`
Expected: `induction-heads-in-the-wild`. If not, stop and report.

- [ ] **Step 2: Dev-test the toy standalone**

Write `<scratchpad>/dev_act0.py` containing exactly the blog's toy (below, from the `import numpy` line through the `__main__` block) and run it: `uv run --no-sync python <scratchpad>/dev_act0.py`

Expected output (mechanics check — these exact two lines):
```
<s> banana apple cherry banana -> apple (87.0%)
<s> apple banana cherry apple -> banana (87.0%)
```

- [ ] **Step 3: Create the notebook with Act 0 cells**

Write and run an append-cells script (Shared mechanics pattern) that first *creates* the notebook (`nb = nbformat.v4.new_notebook()` instead of `read`) and adds these cells in order:

Cell 1 (md):
````markdown
# Induction heads in the wild

In [*I Didn't Understand QKV, So I Hand-Crafted an Induction Head*](https://barmag.github.io/mechanistic-interpretability/machine-learning/learning-in-public/2026/07/15/i-didnt-understand-qkv-so-i-hand-crafted-an-induction-head.html) we built a two-head induction circuit by hand: a **previous-token head** whose QK was a literal `shift` matrix, feeding a **K-composed induction head** whose QK matched token identity and whose OV copied. Eight matrices, no training, 87% correct.

The post ended on a question, and this notebook answers it:

> If I take a small open-weights model like Pythia and go looking for its previous-token head and its induction head, will the weights look anything like the shift and projection I just wrote by hand? Or does training find some smeared version, spread across heads, that only approximates this behavior?

**Hypothesis.** The hand-crafted circuit is reproducible in pre-trained models:

1. GPT-2 small's previous-token head implements `shift` visibly in its **positional QK circuit** (a subdiagonal stripe).
2. Both models' induction heads show **token-identity QK matching** (through composition with the prev-token head), a **copying OV**, and a **K-composition score** that singles out the prev-token head.
3. **Falsifiable twist:** in Pythia-160m the `shift` matrix should *not exist as a weight-space object* — rotary embeddings (RoPE) never write position into the residual stream, so the same behavior must be implemented without the matrix the toy used.

**Models:** GPT-2 small (learned absolute positions — the `shift` question is answerable matrix-by-matrix) and Pythia-160m (RoPE — the contrast case).

**Papers:** Olsson et al. 2022, [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (behavioral scores); Elhage et al. 2021, [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (QK/OV circuits, composition scores).
````

Cell 2 (code):
```python
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")  # Strix Halo gfx1151 runs the gfx1100 wheels

import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"device={device}  seed={SEED}  commit={commit}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
```

Cell 3 (md):
````markdown
## Act 0 — The toy, re-run

Before opening any trained weights, here is the object of comparison, verbatim from the post: a 15-dimensional residual stream split into three blocks (`what I am`, `before me`, `where I am`), layer 1's QK a `shift` permutation that asks "who sits at position i−1?", layer 2's QK an identity match against the `before me` block, and an OV that copies. Every matrix is exact because we chose it; the question for the rest of the notebook is which of these choices training rediscovers.

The four jobs we will hunt, and where they live in the toy:

| Job | Toy matrices | What it does |
|---|---|---|
| shift-QK | `W_Q1 = shift×3`, `W_K1 = I×3` (position block) | attend to position i−1 |
| token-match-QK | `W_Q2 = I×2` (token block), `W_K2 = I×2` (before-me block) | "who has *my* token in their before-me slot?" |
| K-composition | `W_O1` writes what `W_K2` reads | layer 1's output becomes layer 2's key |
| copy-OV | `W_V2 = I×4`, `W_O2 = I` | hand over the matched position's own token |
````

Cell 4 (code): the blog's toy, **verbatim** — the full listing from `import numpy as np` through the `__main__` loop:

```python
import numpy as np

VOCAB = ["apple", "banana", "cherry", "kiwi", "lemon"]
TOK = {t: i for i, t in enumerate(VOCAB)}
D_MODEL = 15  # 5 "what I am" + 5 "before me" + 5 "where I am"

def embed(seq):
    x = np.zeros((len(seq), D_MODEL))
    for i, tok in enumerate(seq):
        if tok is not None:
            x[i, TOK[tok]] = 1.0   # what I am
        x[i, 10 + i] = 1.0         # where I am
    return x

def attention(x, w_q, w_k, w_v, w_o):
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    scores = q @ k.T
    scores[np.triu_indices(len(x), k=1)] = -np.inf   # causal mask
    scores -= scores.max(axis=1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(axis=1, keepdims=True)
    return x + (attn @ v) @ w_o.T, attn

# --- Layer 1: previous-token head ---
shift = np.zeros((5, 5))
for i in range(1, 5):
    shift[i - 1, i] = 1.0   # onehot(i) -> onehot(i-1)

W_Q1 = np.hstack([np.zeros((5, 10)), shift * 3])
W_K1 = np.hstack([np.zeros((5, 10)), np.eye(5) * 3])
W_V1 = np.hstack([np.eye(5), np.zeros((5, 10))])
W_O1 = np.vstack([np.zeros((5, 5)), np.eye(5), np.zeros((5, 5))])

# --- Layer 2: induction head ---
W_Q2 = np.hstack([np.eye(5) * 2, np.zeros((5, 10))])
W_K2 = np.hstack([np.zeros((5, 5)), np.eye(5) * 2, np.zeros((5, 5))])
W_V2 = np.hstack([np.eye(5) * 4, np.zeros((5, 10))])
W_O2 = np.vstack([np.eye(5), np.zeros((10, 5))])

def predict_next(seq):
    x = embed(seq)
    x, _ = attention(x, W_Q1, W_K1, W_V1, W_O1)
    x, attn2 = attention(x, W_Q2, W_K2, W_V2, W_O2)
    logits = x[-1, :5]
    probs = np.exp(logits) / np.exp(logits).sum()
    return dict(zip(VOCAB, probs.round(4))), attn2[-1].round(4)

for seq in [[None, "banana", "apple", "cherry", "banana"],
            [None, "apple", "banana", "cherry", "apple"]]:
    probs, attn = predict_next(seq)
    top = max(probs, key=probs.get)
    shown = " ".join(t if t else "<s>" for t in seq)
    print(f"{shown} -> {top} ({probs[top]:.1%})")
```

(Note: the `if __name__ == "__main__":` guard is dropped — the loop runs at cell level.)

- [ ] **Step 4: Execute the notebook and verify outputs**

Run the nbconvert command and the output spot-check from Shared mechanics.
Expected: no cell errors; the toy's two 87.0% lines appear; the device line prints.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
git commit -m "induction-in-the-wild Act 0: hand-crafted toy reproduces 87% in-notebook

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Act 1 — Find the heads (behavioral sweep)

**Files:**
- Modify: `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` (append cells)

**Interfaces:**
- Consumes: `SEED`, `device` (Task 1).
- Produces notebook globals used by ALL later tasks:
  - `gpt2`, `pythia` — `HookedTransformer` instances
  - `BATCH = 32`, `T = 50`
  - `repeated_tokens(model, batch=BATCH, block=T, seed=SEED) -> LongTensor [batch, 2*T+1]`
  - `head_scores(model, tokens) -> (prev, ind)` — two `FloatTensor [n_layers, n_heads]` on CPU
  - `SPECIMENS: dict[str, dict[str, tuple[int, int]]]` — e.g. `{"gpt2": {"prev": (l, h), "ind": (l, h)}, "pythia": {...}}`
  - `SCORES: dict[str, dict[str, FloatTensor]]` — same keys, full score grids
  - `MODELS = {"gpt2": gpt2, "pythia": pythia}`

- [ ] **Step 1: Write the dev script**

`<scratchpad>/dev_act1.py` — the full Act 1 code (identical to the cell code in Step 3) plus mechanics assertions at the end:

```python
# after computing prev/ind for both models:
for name in SCORES:
    for kind in ("prev", "ind"):
        s = SCORES[name][kind]
        assert s.shape == (MODELS[name].cfg.n_layers, MODELS[name].cfg.n_heads)
        assert torch.isfinite(s).all()
        assert (s >= 0).all() and (s <= 1).all()  # attention fractions
print("mechanics OK")
print({n: SPECIMENS[n] for n in SPECIMENS})
print({n: {k: round(SCORES[n][k].max().item(), 3) for k in SCORES[n]} for n in SCORES})
```

- [ ] **Step 2: Run the dev script**

Run: `uv run --no-sync python <scratchpad>/dev_act1.py`
Expected: `mechanics OK`, then the specimen coordinates and max scores. Record the printed specimens — they go into the Step 3 markdown and the commit message. Whatever the max induction scores are (high or low), they are the result; do not tune.

- [ ] **Step 3: Append Act 1 cells**

Cell (md):
````markdown
## Act 1 — Find the heads

Olsson et al. 2022 located induction heads behaviorally before anyone opened weights: feed the model a sequence of **random tokens repeated twice** and measure where each head attends. Random tokens matter — there is no grammar to help, so any head that attends to "the token after the previous occurrence of me" can only be doing induction. We use the same two diagnostics, one per toy layer:

- **prev-token score** — mean attention from position *i* to *i−1* (the toy's layer-1 stripe)
- **induction score** — for positions in the second copy, mean attention to the token *after* the previous occurrence (offset *i−T+1* for block length *T*)

Both are fractions of attention mass, 0 to 1. The toy scores ≈1.0 on both by construction (its softmax winner takes 99.96%).
````

Cell (code):
```python
from transformer_lens import FactoredMatrix, HookedTransformer, utils

gpt2 = HookedTransformer.from_pretrained("gpt2", device=device)
pythia = HookedTransformer.from_pretrained("pythia-160m", device=device)
MODELS = {"gpt2": gpt2, "pythia": pythia}
for name, m in MODELS.items():
    print(f"{name}: layers={m.cfg.n_layers} heads={m.cfg.n_heads} d_model={m.cfg.d_model} "
          f"d_head={m.cfg.d_head} positions={m.cfg.positional_embedding_type}")
```

Cell (md):
````markdown
`from_pretrained` applies TransformerLens's standard weight processing (LayerNorm folded into the weights, writing weights centered). That is what makes the weight-space analysis in Act 2 legible — the caveat is that every matrix we inspect is the *processed* one, an exact reparameterization of the original model.

The sweep: a batch of 32 sequences, each a block of 50 uniform-random tokens repeated twice, BOS in front. One forward pass per model, caching only attention patterns.
````

Cell (code):
```python
BATCH, T = 32, 50

def repeated_tokens(model, batch=BATCH, block=T, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    block_toks = torch.randint(100, 50_000, (batch, block), generator=g)  # both vocabs exceed 50k
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    return torch.cat([bos, block_toks, block_toks], dim=1).to(model.cfg.device)

def head_scores(model, tokens):
    """(prev_token, induction) score per head, each [n_layers, n_heads] on CPU."""
    _, cache = model.run_with_cache(
        tokens, return_type=None, names_filter=lambda n: n.endswith("pattern")
    )
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    prev = torch.zeros(n_layers, n_heads)
    ind = torch.zeros(n_layers, n_heads)
    n = tokens.shape[1]
    q_prev = torch.arange(1, n)               # every query with a left neighbour
    q_ind = torch.arange(T + 1, 2 * T + 1)    # queries in the second copy
    for layer in range(n_layers):
        pat = cache["pattern", layer]         # [batch, head, query, key]
        prev[layer] = pat[:, :, q_prev, q_prev - 1].mean(dim=(0, 2)).cpu()
        ind[layer] = pat[:, :, q_ind, q_ind - T + 1].mean(dim=(0, 2)).cpu()
    return prev, ind

SCORES, SPECIMENS = {}, {}
for name, model in MODELS.items():
    prev, ind = head_scores(model, repeated_tokens(model))
    SCORES[name] = {"prev": prev, "ind": ind}
    SPECIMENS[name] = {
        kind: divmod(SCORES[name][kind].argmax().item(), model.cfg.n_heads)
        for kind in ("prev", "ind")
    }
    for kind in ("prev", "ind"):
        l, h = SPECIMENS[name][kind]
        print(f"{name} top {kind:4s} head: L{l}H{h}  score={SCORES[name][kind][l, h]:.3f}")
```

Cell (code):
```python
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for col, name in enumerate(MODELS):
    for row, kind in enumerate(("prev", "ind")):
        ax = axes[row][col]
        im = ax.imshow(SCORES[name][kind], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        l, h = SPECIMENS[name][kind]
        ax.scatter([h], [l], marker="*", s=180, c="red", edgecolors="white")
        ax.set_title(f"{name} — {'prev-token' if kind == 'prev' else 'induction'} score")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
fig.colorbar(im, ax=axes, shrink=0.8, label="mean attention fraction")
plt.show()
```

Cell (md) — **written after Step 2, using the actual numbers**. Content requirements:
- Name the four specimens (e.g. "GPT-2's top prev-token head is L?H? at 0.??").
- Literature sanity check: Wang et al. 2022 (*Interpretability in the Wild*, arXiv:2211.00593) report previous-token heads **2.2 and 4.11** and induction heads **5.5 and 6.9** in GPT-2 small. **Verify these labels against the paper via WebFetch of https://arxiv.org/abs/2211.00593 (or the HTML paper) before writing the cell**; if the fetch fails, write "as commonly cited (unverified against the primary source this run)". State plainly whether our top heads match or differ, and that runner-up heads visible in the heatmaps feed the smearing discussion in Act 3.
- Note the scores are fractions vs the toy's ≈1.0 — first evidence of smearing.

- [ ] **Step 4: Execute the notebook and verify outputs**

Shared-mechanics nbconvert + spot-check. Expected: no cell errors; specimen lines and heatmap figure present.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
git commit -m "induction-in-the-wild Act 1: <fill with actual result, e.g. 'GPT-2 L5H5 + Pythia L4H? found as top induction heads'>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Act 2a — `shift×3` → positional QK (GPT-2) vs RoPE (Pythia)

**Files:**
- Modify: `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `SPECIMENS`, `repeated_tokens`, `SEED`, `T`, `BATCH`, `utils` (Task 2).
- Produces notebook globals for Task 7:
  - `STRIPE: dict` — `{"gpt2": float}` — fraction of positional-QK attention mass on offset −1
  - `PYTHIA_PREV: dict` — `{"seeds": list[float], "natural": float}` — prev-token score of Pythia's specimen head across token draws + natural text

- [ ] **Step 1: Write the dev script**

`<scratchpad>/dev_act2a.py` — loads both models (same `from_pretrained` calls as Act 1), hard-codes the specimen coordinates found in Task 2, runs the Step 3 cell code, and asserts mechanics:

```python
assert pos_attn.shape == (64, 64)
assert torch.isfinite(pos_attn).all()
assert abs(pos_attn.sum(dim=-1)[1:].mean().item() - 1.0) < 1e-4  # rows are distributions
assert 0.0 <= STRIPE["gpt2"] <= 1.0
assert len(PYTHIA_PREV["seeds"]) == 3
print("mechanics OK", STRIPE, PYTHIA_PREV)
```

- [ ] **Step 2: Run the dev script**

Run: `uv run --no-sync python <scratchpad>/dev_act2a.py`
Expected: `mechanics OK` + the numbers. Record them for the markdown.

- [ ] **Step 3: Append Act 2a cells**

Cell (md):
````markdown
## Act 2a — In the toy, `W_Q1 = shift×3` attended to position i−1. Where is `shift` in GPT-2?

The toy could put `shift` *in the weights* because position lived in the residual stream (`where I am` block). GPT-2 works the same way: learned absolute positional embeddings `W_pos` are added to the residual stream, so we can compute the prev-token head's QK circuit **restricted to pure position** — `W_pos · W_Q · W_Kᵀ · W_posᵀ` — and look for the toy's subdiagonal stripe. If the head really asks "who sits at i−1?", the causally-masked softmax of that positional score matrix should put its mass on offset −1, token content ignored.

(Two honest approximations: attention biases `b_Q`, `b_K` are dropped, and real inputs are position *plus* token content — this isolates the positional component only.)
````

Cell (code):
```python
l, h = SPECIMENS["gpt2"]["prev"]
n_pos = 64
P = gpt2.W_pos[:n_pos]                                            # [n_pos, d_model]
pos_scores = P @ gpt2.W_Q[l, h] @ gpt2.W_K[l, h].T @ P.T / gpt2.cfg.d_head**0.5
causal = torch.triu(torch.ones(n_pos, n_pos, dtype=torch.bool, device=pos_scores.device), diagonal=1)
pos_attn = pos_scores.masked_fill(causal, float("-inf")).softmax(dim=-1)

STRIPE = {"gpt2": pos_attn.diagonal(-1).mean().item()}
print(f"GPT-2 L{l}H{h}: positional-QK mass on offset -1 = {STRIPE['gpt2']:.3f}"
      f"  (toy's shift stripe: 0.9996)")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.imshow(pos_attn.detach().cpu(), cmap="viridis")
ax.set_title(f"GPT-2 L{l}H{h} — attention from position alone")
ax.set_xlabel("key position")
ax.set_ylabel("query position")
plt.show()
```

Cell (md):
````markdown
### And in Pythia? There is no matrix to open.

Pythia-160m uses **rotary embeddings (RoPE)**: position is injected by rotating the query and key vectors *inside* the attention computation, as a function of the query–key distance. Position never enters the residual stream, so there is no `W_pos` to multiply through — the toy's `shift` has **no weight-space home** in this model. The claim "this head is position-driven" is still testable, just behaviorally: if the head attends to i−1 regardless of *which* tokens are present, its pattern should be identical across independent random token draws — and hold on natural text too.
````

Cell (code):
```python
def prev_score_on(model, tokens, layer, head):
    _, cache = model.run_with_cache(
        tokens, return_type=None, names_filter=utils.get_act_name("pattern", layer)
    )
    pat = cache["pattern", layer]
    q = torch.arange(1, tokens.shape[1])
    return pat[:, head, q, q - 1].mean().item()

l, h = SPECIMENS["pythia"]["prev"]
print(f"pythia.cfg.positional_embedding_type = {pythia.cfg.positional_embedding_type!r}"
      f"  (gpt2: {gpt2.cfg.positional_embedding_type!r})")

PYTHIA_PREV = {
    "seeds": [prev_score_on(pythia, repeated_tokens(pythia, seed=s), l, h) for s in (SEED, SEED + 1, SEED + 2)],
    "natural": prev_score_on(
        pythia,
        pythia.to_tokens("The lantern was lit at dusk and carried through the old streets of Cairo."),
        l, h,
    ),
}
print(f"Pythia L{l}H{h} prev-token score across three disjoint random draws: "
      + ", ".join(f"{v:.3f}" for v in PYTHIA_PREV["seeds"]))
print(f"same head on natural text: {PYTHIA_PREV['natural']:.3f}")
```

Cell (md) — written with the actual numbers. Content requirements: state whether the GPT-2 stripe fraction is close to the behavioral prev-token score from Act 1 (position alone explains the head) or lower (token content contributes); state whether Pythia's score is stable across draws (position-driven behavior without a positional matrix); close with the verdict-shaped sentence: **in GPT-2 the toy's `shift` exists as a matrix; in Pythia the job exists but the matrix does not** (or whatever the numbers actually show).

- [ ] **Step 4: Execute the notebook and verify outputs**

Shared-mechanics nbconvert + spot-check. Expected: no cell errors; stripe number, stripe figure, and Pythia invariance lines present.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
git commit -m "induction-in-the-wild Act 2a: <actual result, e.g. 'shift stripe carries XX% of GPT-2 prev-head positional attention; no weight-space shift in Pythia (RoPE)'>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Act 2b — `I×2` token match → composed QK diagonal

**Files:**
- Modify: `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `SPECIMENS`, `SCORES`, `SEED` (Task 2).
- Produces notebook global for Task 7:
  - `DIAG: dict` — `{name: {"top1": float, "median_rank": float, "baseline_top1": float}}` for both models

- [ ] **Step 1: Write the dev script**

`<scratchpad>/dev_act2b.py` — models + hard-coded specimens, the Step 3 cell code, mechanics assertions:

```python
for name, d in DIAG.items():
    assert 0.0 <= d["top1"] <= 1.0 and 0.0 <= d["baseline_top1"] <= 1.0
    assert 1.0 <= d["median_rank"] <= 1000.0
print("mechanics OK", DIAG)
```

- [ ] **Step 2: Run the dev script**

Run: `uv run --no-sync python <scratchpad>/dev_act2b.py`
Expected: `mechanics OK` + numbers for the markdown.

- [ ] **Step 3: Append Act 2b cells**

Cell (md):
````markdown
## Act 2b — In the toy, `W_Q2 = I×2` read "what I am" and `W_K2 = I×2` read "before me". Does the trained QK match token identity?

The toy's induction QK is an identity match: the query broadcasts *my token*, the key answers with *the token before me* — which layer 1 wrote there. The trained analog must be computed **through the composition**: the induction head's key reads the residual stream *after* the prev-token head has written to it, so the token→token match matrix is

$$M = W_E \, W_Q^{\text{ind}} \, (W_K^{\text{ind}})^\top \, (W_V^{\text{prev}} W_O^{\text{prev}})^\top \, W_E^\top$$

Row *a*, column *b* of `M` scores: "query token *a*, against a key position whose predecessor was token *b*". If the circuit matches token identity like the toy's `I×2`, the diagonal (*a = b*) should dominate each row. We check on a fixed-seed sample of 1,000 vocabulary tokens, and compare against a **baseline** where the key routes through a *non*-prev-token head from the same layer — the composition should be what creates the diagonal, not the embeddings themselves.

(Same honest approximations as before: LayerNorm between the layers is folded/ignored, biases dropped, RoPE ignored on the Pythia weights — a standard raw-weights approximation.)
````

Cell (code):
```python
def composed_qk_diag(model, prev_lh, ind_lh, n_sample=1000, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(50_000, generator=g)[:n_sample].to(model.cfg.device)
    E = model.W_E[ids]                                     # [n, d_model]
    (l1, h1), (l2, h2) = prev_lh, ind_lh
    w_ov_prev = model.W_V[l1, h1] @ model.W_O[l1, h1]      # [d_model, d_model]
    q_side = E @ model.W_Q[l2, h2]                         # [n, d_head]
    k_side = (E @ w_ov_prev) @ model.W_K[l2, h2]           # [n, d_head]
    M = q_side @ k_side.T
    diag = M.diagonal()
    top1 = (M.argmax(dim=1) == torch.arange(n_sample, device=M.device)).float().mean().item()
    median_rank = (M >= diag[:, None]).sum(dim=1).float().median().item()  # 1.0 = diagonal is the max
    return top1, median_rank

DIAG = {}
for name, model in MODELS.items():
    prev_lh, ind_lh = SPECIMENS[name]["prev"], SPECIMENS[name]["ind"]
    top1, median_rank = composed_qk_diag(model, prev_lh, ind_lh)
    # baseline: route the key through the same layer's WORST prev-token head instead
    l1 = prev_lh[0]
    worst_h = SCORES[name]["prev"][l1].argmin().item()
    base_top1, _ = composed_qk_diag(model, (l1, worst_h), ind_lh)
    DIAG[name] = {"top1": top1, "median_rank": median_rank, "baseline_top1": base_top1}
    print(f"{name}: diagonal is argmax for {top1:.1%} of 1000 tokens "
          f"(median rank {median_rank:.0f}/1000); baseline via L{l1}H{worst_h}: {base_top1:.1%}"
          f"   [toy: 100%]")
```

Cell (md) — written with the actual numbers. Content requirements: interpret top1 vs baseline (does composition create the match?), note median rank, and say what a muddy result means if it is muddy (named reason: raw-weights approximation, single-head route through a multi-head layer). For Pythia specifically, remind that RoPE was ignored on the weights.

- [ ] **Step 4: Execute the notebook and verify outputs**

Shared-mechanics nbconvert + spot-check. Expected: no cell errors; DIAG lines for both models.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
git commit -m "induction-in-the-wild Act 2b: <actual result, e.g. 'composed QK matches token identity for XX%/YY% of sampled vocab (GPT-2/Pythia)'>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Act 2c — K-composition score

**Files:**
- Modify: `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `SPECIMENS`, `utils`, `FactoredMatrix` import (Task 2).
- Produces notebook global for Task 7:
  - `KCOMP: dict` — `{name: {"prev_score": float, "prev_rank": int, "top": (l, h, float)}}` — the prev-token specimen's composition score, its rank among all earlier heads (1 = highest), and the top-scoring earlier head

- [ ] **Step 1: Write the dev script**

`<scratchpad>/dev_act2c.py` — models + hard-coded specimens, Step 3 cell code, mechanics assertions:

```python
for name, d in KCOMP.items():
    assert 0.0 <= d["prev_score"] <= 1.0
    assert d["prev_rank"] >= 1
print("mechanics OK", KCOMP)
```

- [ ] **Step 2: Run the dev script**

Run: `uv run --no-sync python <scratchpad>/dev_act2c.py`
Expected: `mechanics OK` + numbers.

- [ ] **Step 3: Append Act 2c cells**

Cell (md):
````markdown
## Act 2c — In the toy, `W_O1` wrote exactly what `W_K2` read. Is the wiring visible in weight space?

The toy's two layers compose through one channel: layer 1's output matrix writes the `before me` block, layer 2's key matrix reads it. Elhage et al. 2021 call this **K-composition** and give it a weight-space measure — for a candidate earlier head, how much of the induction head's key-side input could come through that head's OV:

$$\text{K-comp}(h_1 \to h_2) = \frac{\lVert W_{OV}^{h_1} \cdot (W_{QK}^{h_2})^\top \rVert_F}{\lVert W_{OV}^{h_1} \rVert_F \, \lVert W_{QK}^{h_2} \rVert_F}$$

The toy's score between its two layers is high by construction (one exact channel, everything else zero). In a trained model every earlier head gets a score; if the circuit is the toy's, the prev-token head should stand out against all of them. (Raw scores have a nonzero floor from random matrix overlap — the *ranking* is the evidence, not the absolute value. RoPE again ignored for Pythia.)
````

Cell (code):
```python
KCOMP = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, (name, model) in zip(axes, MODELS.items()):
    l2, h2 = SPECIMENS[name]["ind"]
    comp = utils.composition_scores(model.OV, model.QK[l2, h2].T).cpu()  # [n_layers, n_heads]
    earlier = comp[:l2].flatten()
    l1, h1 = SPECIMENS[name]["prev"]
    prev_score = comp[l1, h1].item()
    prev_rank = int((earlier > prev_score).sum().item()) + 1
    top_flat = earlier.argmax().item()
    top = (top_flat // model.cfg.n_heads, top_flat % model.cfg.n_heads, earlier.max().item())
    KCOMP[name] = {"prev_score": prev_score, "prev_rank": prev_rank, "top": top}

    labels = [f"L{l}H{h}" for l in range(l2) for h in range(model.cfg.n_heads)]
    colors = ["crimson" if (i // model.cfg.n_heads, i % model.cfg.n_heads) == (l1, h1) else "steelblue"
              for i in range(len(earlier))]
    ax.bar(range(len(earlier)), earlier, color=colors)
    ax.set_title(f"{name}: K-composition into induction head L{l2}H{h2}\n"
                 f"prev-token head L{l1}H{h1} in red — rank {prev_rank}/{len(earlier)}")
    ax.set_xticks(range(0, len(earlier), model.cfg.n_heads))
    ax.set_xticklabels(labels[::model.cfg.n_heads], rotation=90, fontsize=7)
    print(f"{name}: prev head L{l1}H{h1} K-comp={prev_score:.3f} (rank {prev_rank}); "
          f"top earlier head L{top[0]}H{top[1]}={top[2]:.3f}")
plt.show()
```

Cell (md) — written with the actual numbers. Content requirements: does the prev-token head tower (rank 1) or not; if another head outranks it, name it and check its prev-token score in `SCORES` (a second prev-token head outranking is *supporting* evidence of smearing, not a refutation — say which it is).

- [ ] **Step 4: Execute the notebook and verify outputs**

Shared-mechanics nbconvert + spot-check. Expected: no cell errors; K-comp lines + bar charts.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
git commit -m "induction-in-the-wild Act 2c: <actual result, e.g. 'prev-token head ranks 1/48 by K-composition into GPT-2 induction head'>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Act 2d — `I×4` copy → OV copying score

**Files:**
- Modify: `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `SPECIMENS`, `FactoredMatrix` (Task 2).
- Produces notebook global for Task 7:
  - `COPY: dict` — `{name: {"score": float, "frac_pos": float}}` — Σ Re(λ)/Σ |λ| and fraction of eigenvalues with positive real part, for the full OV circuit of each induction specimen

- [ ] **Step 1: Write the dev script**

`<scratchpad>/dev_act2d.py` — models + hard-coded specimens, Step 3 cell code, mechanics assertions:

```python
for name, d in COPY.items():
    assert -1.0 <= d["score"] <= 1.0
    assert 0.0 <= d["frac_pos"] <= 1.0
# toy check: eigenvalues of I*4 are all +4 -> score exactly 1.0
print("mechanics OK", COPY)
```

If `FactoredMatrix.eigenvalues` does not exist in this version, compute manually: nonzero eigenvalues of `X @ Y` equal eigenvalues of `Y @ X` — use `torch.linalg.eigvals((model.W_O[l,h] @ model.W_U) @ (model.W_E @ model.W_V[l,h]))` (a `[d_head, d_head]` product). Use whichever works in the dev script, then put that version in the notebook cell.

- [ ] **Step 2: Run the dev script**

Run: `uv run --no-sync python <scratchpad>/dev_act2d.py`
Expected: `mechanics OK` + numbers.

- [ ] **Step 3: Append Act 2d cells**

Cell (md):
````markdown
## Act 2d — In the toy, `W_V2 = I×4` and `W_O2 = I` copied the matched token. Does the trained OV copy?

The toy's OV hands over the attended position's own token, unchanged. Elhage et al.'s test for "is this OV a copying matrix?" runs the *full* vocabulary-to-vocabulary circuit — embed, through the head's OV, unembed:

$$C = W_E \, W_V \, W_O \, W_U \in \mathbb{R}^{|V| \times |V|}$$

and looks at its **eigenvalues**: a matrix that maps every token toward *itself* has positive eigenvalues (the toy's `I×4` has all eigenvalues +4 — copying score exactly 1.0). $C$ is 50,000×50,000, but its rank is at most $d_{head} = 64$, and the nonzero eigenvalues of $XY$ equal those of $YX$ — so we get the exact eigenvalues from a 64×64 product. No sampling.

Copying score = $\sum_i \text{Re}(\lambda_i) / \sum_i |\lambda_i|$, in [−1, 1].
````

Cell (code):
```python
COPY = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
for ax, (name, model) in zip(axes, MODELS.items()):
    l, h = SPECIMENS[name]["ind"]
    small = (model.W_O[l, h] @ model.W_U) @ (model.W_E @ model.W_V[l, h])  # [d_head, d_head]
    eigs = torch.linalg.eigvals(small.float()).cpu()
    score = (eigs.real.sum() / eigs.abs().sum()).item()
    frac_pos = (eigs.real > 0).float().mean().item()
    COPY[name] = {"score": score, "frac_pos": frac_pos}
    ax.scatter(eigs.real, eigs.imag, s=14)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(f"{name} L{l}H{h} full-OV eigenvalues\ncopying score {score:.3f}, "
                 f"{frac_pos:.0%} positive")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    print(f"{name} induction head L{l}H{h}: copying score {score:.3f} "
          f"({frac_pos:.0%} eigenvalues positive)   [toy I×4: 1.000, 100%]")
plt.show()
```

Cell (md) — written with the actual numbers. Content requirements: compare both scores to the toy's 1.0; a strongly positive spectrum = the toy's copy job found in the wild, a mixed spectrum = named honestly.

- [ ] **Step 4: Execute the notebook and verify outputs**

Shared-mechanics nbconvert + spot-check. Expected: no cell errors; copying lines + eigenvalue scatter.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
git commit -m "induction-in-the-wild Act 2d: <actual result, e.g. 'induction-head OV copies: score 0.9x (GPT-2) / 0.8x (Pythia) vs toy 1.0'>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Act 3 — Verdict table, honest gaps, recap + handoff; final verification

**Files:**
- Modify: `notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` (append cells)

**Interfaces:**
- Consumes ALL prior globals: `SCORES`, `SPECIMENS`, `STRIPE`, `PYTHIA_PREV`, `DIAG`, `KCOMP`, `COPY`.
- Produces: the final committed notebook.

- [ ] **Step 1: Append Act 3 cells**

(No dev script — this act only assembles already-computed values; the notebook execution is the test.)

Cell (md):
````markdown
## Act 3 — Verdict: did training find my circuit?

One row per toy job, one column per model. "Found as a matrix" means the job is legible in weight space; "behavior only" means the job is demonstrably done but has no single weight-space object to point at.
````

Cell (code):
```python
import pandas as pd

def _lh(name, kind):
    l, h = SPECIMENS[name][kind]
    return f"L{l}H{h}"

rows = []
for name in MODELS:
    prev_l, prev_h = SPECIMENS[name]["prev"]
    ind_l, ind_h = SPECIMENS[name]["ind"]
    rows.append({
        "model": name,
        "prev-token head": f"{_lh(name, 'prev')} (score {SCORES[name]['prev'][prev_l, prev_h]:.2f})",
        "induction head": f"{_lh(name, 'ind')} (score {SCORES[name]['ind'][ind_l, ind_h]:.2f})",
        "shift-QK": (f"matrix: stripe {STRIPE['gpt2']:.2f}" if name == "gpt2"
                     else f"behavior only (RoPE): {min(PYTHIA_PREV['seeds']):.2f}–{max(PYTHIA_PREV['seeds']):.2f} across draws"),
        "token-match-QK": f"diag top-1 {DIAG[name]['top1']:.0%} (baseline {DIAG[name]['baseline_top1']:.0%})",
        "K-composition": f"{KCOMP[name]['prev_score']:.3f}, rank {KCOMP[name]['prev_rank']}",
        "copy-OV": f"{COPY[name]['score']:.2f} ({COPY[name]['frac_pos']:.0%} eigs > 0)",
    })
verdict = pd.DataFrame(rows).set_index("model")
verdict
```

Cell (md) — **the honest-gaps + recap section, written from the actual numbers.** Content requirements:
- **What reproduced:** go job by job; state which of the toy's four jobs were found and with what strength.
- **What smeared:** the toy's scores were ≈1.0 by construction; name the actual fractional scores and the runner-up heads sharing each job (read them off the Act 1 heatmaps).
- **What didn't exist as a matrix:** Pythia's `shift` (RoPE) — the notebook's twist finding, stated as the answer to the blog's question: training *can* rediscover the circuit, but one of the toy's two matrices is an implementation choice, not a necessity.
- **Named limitations:** LayerNorm folded/ignored between composed weights, biases dropped, RoPE ignored in Pythia weight products, single-specimen analysis (top-1 heads only).
- **Handoff:** the next rung is **causal ablation** — knock out the found heads and watch the induction score collapse; correlation-to-causation is deliberately deferred.

- [ ] **Step 2: Final full execution and verification**

Run the Shared-mechanics nbconvert + spot-check one last time.
Expected: no cell errors; verdict DataFrame renders; total wall-clock under 10 minutes (time it: prefix the nbconvert command with `time`). Report the actual runtime.

- [ ] **Step 3: Verify notebook hygiene**

```bash
uv run --no-sync python -c "
import nbformat
nb = nbformat.read('notebooks/in_context_learning/induction_heads_in_the_wild.ipynb', as_version=4)
code = sum(1 for c in nb.cells if c.cell_type == 'code')
md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
print(f'{code} code cells, {md} markdown cells')
bad = [w for c in nb.cells for w in ('CLAUDE.md', 'AGENTS.md', 'TODO', 'TBD') if w in ''.join(c.source)]
print('process-talk/placeholder scan:', bad or 'clean')
"
```
Expected: `clean`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/in_context_learning/induction_heads_in_the_wild.ipynb
git commit -m "induction-in-the-wild Act 3: <one-line overall verdict, e.g. 'hand-crafted circuit reproduces in both models; shift matrix exists only where position lives in the residual stream'>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** Act 0 (toy re-run) → Task 1; Act 1 behavioral sweep + literature check → Task 2; Act 2a/2b/2c/2d → Tasks 3–6 (one per toy job, matching the spec's table); Act 3 verdict + honest gaps + handoff → Task 7; infra constraints (uv, no gates, baked outputs, branch discipline) → Global Constraints. Out-of-scope items (ablation, Arabic, more models, blog post) appear in no task. ✓
- **Type consistency:** `SPECIMENS[name][kind] -> (layer, head)` tuples consumed by Tasks 3–7 as defined in Task 2; `STRIPE`/`PYTHIA_PREV`/`DIAG`/`KCOMP`/`COPY` dict shapes match between producing tasks and Task 7's verdict cell. ✓
- **Known API risk:** `utils.composition_scores(left, right)` verified present in transformer_lens 2.15.4 with `left=OV, right=QK.T` = K-composition (checked against `HookedTransformer.all_composition_scores` source). `FactoredMatrix.eigenvalues` intentionally avoided in notebook code (Task 6 uses the explicit 64×64 `eigvals` trick, with the fallback note in the dev step).
