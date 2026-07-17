# Induction Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/in_context_learning/induction_ablation.ipynb` — the causal follow-up to `induction_heads_in_the_wild.ipynb`, which mean-ablates the found induction heads and reports whether induction behavior actually collapses (it does not, for single heads — that is the finding).

**Architecture:** One notebook, seven acts, built act-by-act. Each act's code is first validated in a throwaway dev script (mechanics assertions only — shapes, finiteness; **never** result gates), then appended to the notebook as cells via `nbformat`, then the whole notebook is re-executed in place so outputs stay baked. Cross-act state lives in notebook globals with exact names defined in each task's Interfaces block. Same two models, seed, and score definitions as the wild notebook, so numbers are comparable across notebooks.

**Tech Stack:** transformer_lens (`HookedTransformer`, `utils.get_act_name`, `model.run_with_hooks`, `model.hooks`), torch 2.5.1+rocm, matplotlib, pandas, nbformat. Spec: `docs/superpowers/specs/2026-07-17-induction-ablation-design.md`.

## Global Constraints

- **Branch:** all work on `induction-ablation` (already checked out, off `main`). Never commit to `main`. Do not merge without user confirmation.
- **Runtime bar:** notebook runs end-to-end in <10 min on the iGPU (both models already in the HF cache; forward passes only, no training).
- **No new dependencies.** Everything needed is installed. Always run Python via `uv run --no-sync python …` and jupyter via `uv run --no-sync jupyter …` (bare `uv run` would destroy the ROCm venv).
- **Honest negatives:** the notebook contains **no pass/fail gates** — every metric is computed, printed, and reported whatever its value. Assertions are allowed only in dev scripts and only for mechanics (shape/dtype/finiteness), never for result magnitudes.
- **Outputs baked:** the committed notebook contains executed outputs (follows the wild notebook's precedent). Re-execute in place before every commit.
- **No Arabic content** in this notebook (blog-companion thread; the dialect track is unaffected).
- **No process-talk in notebook markdown** — markdown cells are pedagogical only; never mention CLAUDE.md, repo conventions, tasks, or this plan.
- **Notebook execution discipline:** run `nbconvert --execute --inplace` synchronously in the foreground and never edit the .ipynb while an execution is running (last-writer-wins clobbering).
- **Commit messages name the result**, not the change (e.g. "single-head ablation collapses induction 0.7%; the cluster collapses it 81%", not "add act 2").
- **Dev scripts** go in `/tmp/claude-1000/-home-yassermakram-code-fanous-llm-lens/12a53545-9681-4eb7-ae9a-7cc1c0f12ecd/scratchpad/` — never committed.
- **GPU env:** every dev script and the notebook setup cell begin with `os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")` **before** importing torch.
- **Inherited constants** (from the wild notebook, do not change): `SEED = 42`, `BATCH, T = 32, 50`, induction-score gate `0.2`, random-token range `randint(100, 50_000)`, induction-score offset `q - T + 1`.

## Verified mechanics (measured on this machine, 2026-07-17 — expect these values)

These come from dev probes already run against the real models. They are the expected
outputs; **if the notebook prints something different, report what it prints** — do not
tune toward these.

| Quantity | GPT-2 | Pythia-160m |
|---|---|---|
| clean induction loss gap (BATCH=16) | 12.679 nats | 17.999 nats |
| top-1 induction head | L5H5 (0.927) | L4H6 (0.985) |
| top-1 ablation collapse | **0.7 %** | **4.4 %** |
| above-gate cluster size (>0.2) | 18 heads | 21 heads |
| cluster ablation collapse | **81.0 %** | **85.4 %** |
| hydra: L6H9 induction score, clean → L5H5-ablated | 0.917 → **0.963** | — |
| mediation: induction stripe, clean → prev-ablated | 0.928 → **0.636** | — |
| natural-text delta, repeated-bigram vs elsewhere | +0.517 vs +0.049 nats | — |
| DLA matched-token logit, clean → feeder-ablated | +4.769 → +3.387 | — |
| OV surgery (Pythia L4H6) | — | copying 0.423 → **0.759**; gap 17.999 → 16.979; natural 3.366 → 3.379 |

**Two mechanics facts worth knowing before you write code:**

1. **Zero-ablation barely moves the gap** (−0.0 % collapse on GPT-2 L5H5) while mean-ablation
   moves it 0.7 %. Both are ~nothing; Act 1's pedagogical point is the *principle*
   (on-distribution), and the honest framing is that at single-head scale neither tool shows
   an effect, because of the cluster — not that mean-ablation is dramatically sharper.
2. **The copying score cannot reach 1.0 after eigenvalue surgery.** The spectral projector
   cleanly zeroes the negative-real-part directions (dropped |λ| ≈ 1e-6 vs kept |λ| ≈ 27),
   but the surviving eigenvalues are still *complex*: for λ = a+bi with a>0, |λ| = √(a²+b²) > a,
   so `Σ Re(λ) / Σ |λ|` saturates at 0.759, not 1.0. The residual gap is **imaginary
   (rotational) mass, not negative mass**. Say this in the markdown; do not "fix" it.

## Shared mechanics (referenced by every task)

**Appending cells.** Each task appends cells with a one-off script following this exact pattern:

```python
# append_cells_actN.py  (run from repo root: uv run --no-sync python <scratchpad>/append_cells_actN.py)
import nbformat

PATH = "notebooks/in_context_learning/induction_ablation.ipynb"
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
  notebooks/in_context_learning/induction_ablation.ipynb
```

Expected: exits 0. Then spot-check the new outputs:

```bash
uv run --no-sync python -c "
import nbformat
nb = nbformat.read('notebooks/in_context_learning/induction_ablation.ipynb', as_version=4)
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

### Task 1: Notebook scaffold + Act 0 (setup and the yardstick)

**Files:**
- Create: `notebooks/in_context_learning/induction_ablation.ipynb`

**Interfaces:**
- Consumes: nothing.
- Produces (notebook globals for all later acts):
  - `SEED: int = 42`, `device: str`, `commit: str`, `BATCH: int = 32`, `T: int = 50`, `GATE: float = 0.2`
  - `MODELS: dict[str, HookedTransformer]` — keys `"gpt2"`, `"pythia"`
  - `repeated_tokens(model, batch=BATCH, block=T, seed=SEED) -> Tensor [batch, 2T+1]`
  - `TOKENS: dict[str, Tensor]` — eval batch per model
  - `loss_gap(lpt) -> float` — first-half mean minus second-half mean
  - `halves(lpt) -> tuple[float, float]`
  - `BASE: dict[str, dict]` — `{"first": float, "second": float, "gap": float}` per model
  - `collapse(gap_abl, name) -> float` — `(BASE[name]["gap"] - gap_abl) / BASE[name]["gap"]`

- [ ] **Step 1: Create the notebook with title + setup cells**

Write a script `make_nb.py` in the scratchpad that creates the notebook with the kernelspec
metadata (matching the wild notebook so interactive Jupyter opens it cleanly):

```python
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "fanous-llm-lens", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
        "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.11.15",
    },
}

TITLE = """# Knocking out an induction head

[*Induction heads in the wild*](https://barmag.github.io/) found the hand-crafted circuit inside GPT-2 small and Pythia-160m: a previous-token head (GPT-2 **L4H11**, Pythia **L3H2**) feeding a K-composed induction head (GPT-2 **L5H5**, Pythia **L4H6**), with a copying OV. Every one of those findings was **correlational**. A matrix looked right. A score was high. A rank was 1.

That notebook ended on a promise:

> Knock out L4H11 and L5H5, or L3H2 and L4H6, and watch whether the induction score collapses. That is the next notebook.

This is that notebook. We cut heads out of the forward pass and watch what breaks.

**Hypotheses.**

1. **H1 (knock-out).** If L5H5 / L4H6 *is* the induction head, removing it should collapse induction behavior — the model should lose its ability to predict the repeated half of a repeated random sequence. Predicted wrinkle: only a *partial* collapse, because the wild notebook already showed the behavior lives in a small cluster (GPT-2's L6H9 scores 0.917 against L5H5's 0.930).
2. **H2 (mediation).** If K-composition is causal, removing the *previous-token* head should kill the *induction* head's attention stripe — an effect one layer removed from the intervention.
3. **H3 (stretch).** If Pythia L4H6's negative OV eigenvalue mass is non-copying work sharing the same head, surgically removing those directions should preserve induction while costing something elsewhere.

**Papers.** Olsson et al. 2022, [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (ablations, per-token induction loss); Wang et al. 2023, [*Interpretability in the Wild*](https://arxiv.org/abs/2211.00593) (backup heads — the hydra effect); Elhage et al. 2021, [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (OV eigenvalues).
"""

SETUP = '''import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")  # Strix Halo gfx1151 runs the gfx1100 wheels

import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer, utils

SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"device={device}  seed={SEED}  commit={commit}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))'''

ACT0_MD = """## Act 0 — The yardstick

Olsson et al. measure induction with a **repeated random sequence**: sample a block of random tokens, concatenate it with itself, and ask the model to predict. The first copy is unpredictable by construction — random tokens carry no signal, so the model can do no better than its prior. The second copy is perfectly predictable *if and only if* the model can look back, find where the current token appeared before, and copy what followed.

So the difference between those two halves is the behavior itself, measured in nats:

> **induction loss gap** = mean loss on the first (unrepeated) half − mean loss on the second (repeated) half.

Positive by construction when induction works. Every intervention in this notebook is scored by how much of this gap it destroys:

> **collapse fraction** = (gap_clean − gap_ablated) / gap_clean

0% means the intervention did nothing. 100% means induction is gone.

Both models, same seed and batch as the previous notebook, so the numbers line up across the two."""

ACT0_CODE = '''gpt2 = HookedTransformer.from_pretrained("gpt2", device=device)
pythia = HookedTransformer.from_pretrained("pythia-160m", device=device)
MODELS = {"gpt2": gpt2, "pythia": pythia}

BATCH, T, GATE = 32, 50, 0.2


def repeated_tokens(model, batch=BATCH, block=T, seed=SEED):
    """[BOS, block, block] -> [batch, 2T+1]. Same generator and seed as the previous notebook."""
    g = torch.Generator().manual_seed(seed)
    block_toks = torch.randint(100, 50_000, (batch, block), generator=g)  # both vocabs exceed 50k
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    return torch.cat([bos, block_toks, block_toks], dim=1).to(model.cfg.device)


def halves(lpt):
    """loss_per_token[:, i] scores predicting token i+1. Cols 0..T-1 = first copy, T..2T-1 = second."""
    return lpt[:, 0:T].mean().item(), lpt[:, T:2 * T].mean().item()


def loss_gap(lpt):
    first, second = halves(lpt)
    return first - second


TOKENS, BASE = {}, {}
for name, model in MODELS.items():
    TOKENS[name] = repeated_tokens(model)
    lpt = model(TOKENS[name], return_type="loss", loss_per_token=True)
    first, second = halves(lpt)
    BASE[name] = {"first": first, "second": second, "gap": first - second}
    print(f"{name}: first(unrepeated)={first:6.3f}  second(repeated)={second:5.3f}  "
          f"gap={first - second:6.3f} nats")


def collapse(gap_abl, name):
    return (BASE[name]["gap"] - gap_abl) / BASE[name]["gap"]'''

for kind, src in [("md", TITLE), ("code", SETUP), ("md", ACT0_MD), ("code", ACT0_CODE)]:
    nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md"
                    else nbformat.v4.new_code_cell(src))

nbformat.write(nb, "notebooks/in_context_learning/induction_ablation.ipynb")
print(f"created with {len(nb.cells)} cells")
```

Run: `uv run --no-sync python <scratchpad>/make_nb.py`
Expected: `created with 4 cells`

- [ ] **Step 2: Execute the notebook**

Run the Shared-mechanics nbconvert command, then the spot-check.
Expected: no cell errors. Printed gap ≈ 12.7 (gpt2) and ≈ 18.0 (pythia) — BATCH=32 may
shift these a little from the BATCH=16 probe values; report what prints.

- [ ] **Step 3: Add a results markdown cell naming the measured numbers**

Append one `md` cell that states the two models' actual printed gaps and what they mean
(the unrepeated half costs ~13/~18 nats because random tokens are unpredictable; the
repeated half costs a fraction of a nat — the model has learned to copy). Use the real
numbers from Step 2's output, not the table above.

- [ ] **Step 4: Re-execute and commit**

```bash
git add notebooks/in_context_learning/induction_ablation.ipynb
git commit -m "induction-ablation Act 0: induction loss gap is <X> nats (GPT-2) / <Y> nats (Pythia) — the yardstick every ablation is scored against

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Act 1 — Calibrate the scalpel (mean vs zero ablation)

**Files:**
- Modify: `notebooks/in_context_learning/induction_ablation.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `TOKENS`, `BASE`, `loss_gap`, `collapse`, `repeated_tokens`, `T`, `SEED` (Task 1).
- Produces:
  - `z_means(model, ref_tokens) -> dict[int, Tensor [n_heads, d_head]]`
  - `ZM: dict[str, dict[int, Tensor]]` — synthetic-reference mean `hook_z` per model per layer
  - `ablation_hooks(targets, zm) -> list[tuple[str, callable]]` where `targets: list[tuple[int, int]]`
  - `run_ablated(model, tokens, targets, zm, names_filter=None)` — returns `loss_per_token` when
    `names_filter is None`, else `(None, cache)`
  - `CALIB: dict[str, float]` — `{"mean": collapse, "zero": collapse}` for GPT-2 L5H5

- [ ] **Step 1: Write the dev script and verify mechanics**

Create `<scratchpad>/dev_act1.py` that loads gpt2, builds `ZM`, and asserts mechanics only:

```python
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from collections import defaultdict
import torch
from transformer_lens import HookedTransformer, utils

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
SEED, BATCH, T = 42, 8, 50

def repeated_tokens(model, batch=BATCH, block=T, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    bt = torch.randint(100, 50_000, (batch, block), generator=g)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    return torch.cat([bos, bt, bt], dim=1).to(model.cfg.device)

def z_means(model, ref_tokens):
    _, c = model.run_with_cache(ref_tokens, return_type=None,
                                names_filter=lambda n: n.endswith("hook_z"))
    return {l: c["z", l].mean(dim=(0, 1)) for l in range(model.cfg.n_layers)}

zm = z_means(model, repeated_tokens(model, seed=SEED + 1000))
assert zm[0].shape == (model.cfg.n_heads, model.cfg.d_head), zm[0].shape
assert torch.isfinite(zm[0]).all()
print("z_means OK", zm[0].shape)
```

Run: `uv run --no-sync python <scratchpad>/dev_act1.py`
Expected: `z_means OK torch.Size([12, 64])`

- [ ] **Step 2: Append the Act 1 cells**

Markdown cell — the paper hook and the on-distribution argument:

```
## Act 1 — Calibrate the scalpel

To remove a head we replace its output with something else. The naive choice is **zero**: set the head's `hook_z` (its per-head output, before `W_O`) to 0. But zero is not a neutral value. A trained network never sees a head output zero; the rest of the model has no idea what to do with it. Zeroing takes the model **off-distribution**, and then any damage you measure is partly the damage of feeding the model an input it was never built for — not the loss of the head's job.

The standard fix, and what the interpretability literature settled on: **mean-ablation**. Replace the head's output with its *average* output over a reference distribution. The head stops carrying information about *this particular sequence*, but the downstream layers still receive a vector of the size and shape they expect. What you subtract is the head's signal, not its existence.

We show both, once, on GPT-2's L5H5 — then use mean-ablation everywhere after.
```

Code cell:

```python
def z_means(model, ref_tokens):
    """Per-head mean hook_z over batch and position: {layer: [n_heads, d_head]}."""
    _, cache = model.run_with_cache(
        ref_tokens, return_type=None, names_filter=lambda n: n.endswith("hook_z")
    )
    return {l: cache["z", l].mean(dim=(0, 1)) for l in range(model.cfg.n_layers)}


# Reference batch: same distribution as the eval batch, disjoint draw.
ZM = {name: z_means(m, repeated_tokens(m, seed=SEED + 1000)) for name, m in MODELS.items()}


def ablation_hooks(targets, zm):
    """targets: [(layer, head), ...] -> TransformerLens fwd_hooks replacing each head's z."""
    by_layer = defaultdict(list)
    for layer, head in targets:
        by_layer[layer].append(head)

    def make(heads, layer):
        def hook(z, hook):          # z: [batch, pos, head, d_head]
            for h in heads:
                z[:, :, h, :] = zm[layer][h]
            return z
        return hook

    return [(utils.get_act_name("z", l), make(hs, l)) for l, hs in by_layer.items()]


def run_ablated(model, tokens, targets, zm, names_filter=None):
    hooks = ablation_hooks(targets, zm)
    if names_filter is None:
        return model.run_with_hooks(tokens, return_type="loss", loss_per_token=True, fwd_hooks=hooks)
    with model.hooks(fwd_hooks=hooks):
        return model.run_with_cache(tokens, return_type=None, names_filter=names_filter)


IND = {"gpt2": (5, 5), "pythia": (4, 6)}      # induction heads, from the previous notebook
PREV = {"gpt2": (4, 11), "pythia": (3, 2)}    # previous-token heads

ZERO = {name: {l: torch.zeros_like(v) for l, v in zm.items()} for name, zm in ZM.items()}

CALIB = {}
for kind, ref in [("mean", ZM), ("zero", ZERO)]:
    lpt = run_ablated(gpt2, TOKENS["gpt2"], [IND["gpt2"]], ref["gpt2"])
    CALIB[kind] = collapse(loss_gap(lpt), "gpt2")
    print(f"GPT-2 L5H5 {kind}-ablated: gap={loss_gap(lpt):6.3f}  "
          f"collapse={CALIB[kind]:5.1%}  (clean gap {BASE['gpt2']['gap']:.3f})")
```

- [ ] **Step 3: Execute and read the result**

Run the Shared-mechanics nbconvert + spot-check.
Expected: both collapse numbers near 0% (probe measured mean 0.7%, zero −0.0%).

- [ ] **Step 4: Append the honest results markdown**

Append an `md` cell reporting the two measured numbers. **The honest framing** (do not
oversell mean-ablation): both interventions destroy essentially none of the induction gap.
Zero-ablation's off-distribution worry is real in principle, and mean-ablation is the right
default for everything that follows — but at *single-head* scale neither tool registers an
effect at all, and that non-effect is Act 2's subject, not a defect of the scalpel. Write it
with the actual printed numbers.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_ablation.ipynb
git commit -m "induction-ablation Act 1: mean- and zero-ablating GPT-2 L5H5 both collapse the gap ~0% — the scalpel is calibrated, the head is not the circuit

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Act 2 — Knock out the induction head (escalation ladder + hydra check)

**Files:**
- Modify: `notebooks/in_context_learning/induction_ablation.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `TOKENS`, `BASE`, `ZM`, `IND`, `run_ablated`, `loss_gap`, `collapse`, `T`, `GATE`.
- Produces:
  - `ind_scores(cache, model) -> Tensor [n_layers, n_heads]`
  - `SCORES: dict[str, Tensor]` — clean induction score grid per model
  - `CLUSTER: dict[str, list[tuple[int, int]]]` — above-gate heads per model
  - `TOP3: dict[str, list[tuple[int, int]]]`
  - `LADDER: dict[str, dict[str, float]]` — `{"top-1": collapse, "top-3": collapse, "cluster": collapse}`
  - `HYDRA: dict[str, Tensor]` — induction score grid under top-1 ablation

- [ ] **Step 1: Append Act 2a cells (single-head knock-out)**

Markdown:

```
## Act 2a — Knock out the induction head

Olsson et al. establish induction heads causally by ablating them and watching in-context learning degrade. So: mean-ablate the single best induction head each model has — GPT-2's L5H5 (score 0.930), Pythia's L4H6 (0.987) — and measure how much of the loss gap dies with it.

If "the induction head" is the induction head, this is where the behavior collapses.
```

Code:

```python
def ind_scores(cache, model):
    """Induction attention score per head: mass at offset -(T-1), same definition as before."""
    q = torch.arange(T + 1, 2 * T + 1)
    out = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        out[layer] = cache["pattern", layer][:, :, q, q - T + 1].mean(dim=(0, 2)).cpu()
    return out


SCORES, LADDER = {}, {}
for name, model in MODELS.items():
    _, cache = model.run_with_cache(
        TOKENS[name], return_type=None, names_filter=lambda n: n.endswith("pattern")
    )
    SCORES[name] = ind_scores(cache, model)
    del cache
    lpt = run_ablated(model, TOKENS[name], [IND[name]], ZM[name])
    LADDER[name] = {"top-1": collapse(loss_gap(lpt), name)}
    l, h = IND[name]
    print(f"{name}: ablate L{l}H{h} (score {SCORES[name][l, h]:.3f})  "
          f"gap {BASE[name]['gap']:6.3f} -> {loss_gap(lpt):6.3f}   "
          f"collapse={LADDER[name]['top-1']:5.1%}")
```

- [ ] **Step 2: Execute; expect a near-zero collapse**

Run nbconvert + spot-check. Expected ≈0.7% (gpt2), ≈4.4% (pythia). Report what prints.

- [ ] **Step 3: Append Act 2b (the hydra check)**

Markdown — hook to Wang et al.'s backup heads:

```
### Act 2b — The hydra check

Nothing collapsed. Before concluding the head does nothing, consider what *Interpretability in the Wild* found when it ablated GPT-2's name-mover heads: other heads **took over**. Wang et al. call them backup heads — heads that sit quiet in the clean model and step into the job when the primary is removed. Cut off one head, two grow back.

The previous notebook already flagged the suspects: L6H9 scores 0.917 on induction against L5H5's 0.930. That is not a runner-up, it is a twin.

So re-measure *every* head's induction score inside the ablated forward pass, and compare to clean. If the runners-up strengthen, the circuit is a hydra and single-head ablation was never going to work.
```

Code:

```python
HYDRA = {}
for name, model in MODELS.items():
    _, cache = run_ablated(
        model, TOKENS[name], [IND[name]], ZM[name],
        names_filter=lambda n: n.endswith("pattern"),
    )
    HYDRA[name] = ind_scores(cache, model)
    del cache
    l, h = IND[name]
    delta = HYDRA[name] - SCORES[name]
    delta[l, h] = 0.0                       # the ablated head itself is not a backup
    flat = delta.flatten().argsort(descending=True)[:3]
    print(f"{name}: heads that strengthen most under L{l}H{h} ablation")
    for f in flat:
        ll, hh = divmod(f.item(), model.cfg.n_heads)
        print(f"    L{ll}H{hh}: {SCORES[name][ll, hh]:.3f} -> {HYDRA[name][ll, hh]:.3f} "
              f"({delta[ll, hh]:+.3f})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, (name, model) in zip(axes, MODELS.items()):
    x, y = SCORES[name].flatten(), HYDRA[name].flatten()
    ax.scatter(x, y, s=16, alpha=0.7)
    lims = [0, max(x.max().item(), y.max().item()) * 1.05]
    ax.plot(lims, lims, color="grey", lw=0.8, ls="--")
    l, h = IND[name]
    ax.scatter([SCORES[name][l, h]], [HYDRA[name][l, h]], marker="*", s=200, c="red",
               edgecolors="white", zorder=3, label=f"ablated L{l}H{h}")
    ax.set_title(f"{name} — induction score, clean vs L{l}H{h}-ablated")
    ax.set_xlabel("clean induction score")
    ax.set_ylabel("score under ablation")
    ax.legend(loc="lower right")
plt.show()
```

- [ ] **Step 4: Append Act 2c (escalate)**

Markdown:

```
### Act 2c — Escalate

If one head is not the circuit, how many are? Take every head whose clean induction score clears 0.2 — the previous notebook's gate — and remove the whole set at once. Between the two, an intermediate rung: the top 3 heads by score.

This is the escalation ladder: one head, three heads, the whole cluster. Where the gap finally falls is a measurement of how distributed the behavior is.
```

Code:

```python
CLUSTER, TOP3 = {}, {}
for name, model in MODELS.items():
    n_heads = model.cfg.n_heads
    CLUSTER[name] = [(l, h) for l in range(model.cfg.n_layers) for h in range(n_heads)
                     if SCORES[name][l, h] > GATE]
    TOP3[name] = [divmod(f.item(), n_heads)
                  for f in SCORES[name].flatten().argsort(descending=True)[:3]]
    for label, targets in [("top-3", TOP3[name]), ("cluster", CLUSTER[name])]:
        lpt = run_ablated(model, TOKENS[name], targets, ZM[name])
        LADDER[name][label] = collapse(loss_gap(lpt), name)
        print(f"{name}: ablate {label:8s} (n={len(targets):2d})  gap -> {loss_gap(lpt):6.3f}  "
              f"collapse={LADDER[name][label]:5.1%}")
    print(f"    {name} cluster: " + ", ".join(f"L{l}H{h}({SCORES[name][l, h]:.2f})"
                                              for l, h in CLUSTER[name]))

fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
rungs = ["top-1", "top-3", "cluster"]
width = 0.36
for i, name in enumerate(MODELS):
    ax.bar([r + (i - 0.5) * width for r in range(len(rungs))],
           [LADDER[name][r] for r in rungs], width, label=name)
ax.set_xticks(range(len(rungs)))
ax.set_xticklabels([f"{r}\n(n={len(CLUSTER[n]) if r == 'cluster' else (3 if r == 'top-3' else 1)})"
                    if r == "cluster" else r for r in rungs])
ax.set_ylabel("fraction of induction loss gap destroyed")
ax.set_title("Escalation ladder — how many heads must go before induction dies?")
ax.axhline(0, color="black", lw=0.8)
ax.legend()
plt.show()
```

- [ ] **Step 5: Execute**

Run nbconvert + spot-check. Expected: cluster ≈81% (gpt2, 18 heads), ≈85% (pythia, 21 heads).

- [ ] **Step 6: Append results markdown**

An `md` cell with the real numbers, making the point plainly: the top-1 head is *not* the
circuit; ablating it destroys ~1–4% of the behavior while a twin head strengthens to cover.
The behavior only dies when ~18–21 heads go. Name the hydra explicitly (quote the measured
L6H9 clean→ablated numbers). State the honest caveat: ablating 18 heads is a large lesion —
some of that 81% may be general damage, not induction-specific, which is exactly what Act 4
tests on real text.

- [ ] **Step 7: Commit**

```bash
git add notebooks/in_context_learning/induction_ablation.ipynb
git commit -m "induction-ablation Act 2: the induction head is a hydra — top-1 ablation destroys <X>% of the gap while L6H9 strengthens <A>->/<B>; the full cluster destroys <Y>%

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Act 3 — Knock out the feeder (mediation)

**Files:**
- Modify: `notebooks/in_context_learning/induction_ablation.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `TOKENS`, `BASE`, `ZM`, `IND`, `PREV`, `SCORES`, `run_ablated`, `loss_gap`, `collapse`, `T`.
- Produces: `MED: dict[str, dict]` — `{"stripe_clean": float, "stripe_abl": float, "gap_abl": float, "collapse": float}` per model.

- [ ] **Step 1: Append the Act 3 cells**

Markdown:

```
## Act 3 — Knock out the feeder

The previous notebook's cleanest finding was **K-composition**: of all 60 heads that could feed GPT-2's L5H5 key, the previous-token head L4H11 ranked **first** (0.097); for Pythia's L4H6, L3H2 ranked first of 48 (0.147). That is a weight-space claim — the wiring *looks* right.

Here is its causal test. Ablate **only the previous-token head** and read out the *induction* head's attention. The induction head itself is untouched; if its stripe to matched positions collapses anyway, then the previous-token head's output is what the induction head was keying on. Cause, one layer downstream of the cut.

This is also a fairer test of a single-head ablation than Act 2: the previous-token head has no twin at 0.917.
```

Code:

```python
MED = {}
for name, model in MODELS.items():
    li, hi = IND[name]
    lp, hp = PREV[name]
    q = torch.arange(T + 1, 2 * T + 1)

    _, clean_cache = model.run_with_cache(
        TOKENS[name], return_type=None, names_filter=utils.get_act_name("pattern", li)
    )
    stripe_clean = clean_cache["pattern", li][:, hi, q, q - T + 1].mean().item()
    del clean_cache

    _, abl_cache = run_ablated(
        model, TOKENS[name], [PREV[name]], ZM[name],
        names_filter=utils.get_act_name("pattern", li),
    )
    stripe_abl = abl_cache["pattern", li][:, hi, q, q - T + 1].mean().item()
    del abl_cache

    lpt = run_ablated(model, TOKENS[name], [PREV[name]], ZM[name])
    MED[name] = {
        "stripe_clean": stripe_clean, "stripe_abl": stripe_abl,
        "gap_abl": loss_gap(lpt), "collapse": collapse(loss_gap(lpt), name),
    }
    print(f"{name}: ablate prev-token L{lp}H{hp} -> induction head L{li}H{hi} stripe "
          f"{stripe_clean:.3f} -> {stripe_abl:.3f} "
          f"({(stripe_clean - stripe_abl) / stripe_clean:.1%} gone);  "
          f"model loss gap {BASE[name]['gap']:.3f} -> {MED[name]['gap_abl']:.3f} "
          f"(collapse {MED[name]['collapse']:.1%})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, name in zip(axes, MODELS):
    li, hi = IND[name]
    lp, hp = PREV[name]
    ax.bar(["clean", f"L{lp}H{hp}\nablated"],
           [MED[name]["stripe_clean"], MED[name]["stripe_abl"]],
           color=["steelblue", "crimson"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("induction stripe (attention mass at matched position)")
    ax.set_title(f"{name} — L{li}H{hi}'s stripe when its feeder is cut")
plt.show()
```

- [ ] **Step 2: Execute**

Run nbconvert + spot-check. Expected (gpt2): stripe 0.928 → 0.636. Pythia unmeasured in the
probe — report whatever prints.

- [ ] **Step 3: Append results markdown**

State the measured stripe drop per model and the model-level collapse. The point: this is
the wild notebook's rank-1 K-composition finding *causally confirmed* — cutting the feeder
degrades the untouched downstream head's attention. Note honestly whether the stripe drops
partially rather than to zero (it does for GPT-2: 0.636 remains), and what that means —
other heads also feed the key, so the previous-token head is a major but not sole source.
Compare the model-level loss-gap collapse to Act 2's top-1 number.

- [ ] **Step 4: Commit**

```bash
git add notebooks/in_context_learning/induction_ablation.ipynb
git commit -m "induction-ablation Act 3: K-composition is causal — cutting the prev-token head drops the untouched induction head's stripe <A>-><B> (GPT-2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Act 4 — Does it matter on real text?

**Files:**
- Modify: `notebooks/in_context_learning/induction_ablation.ipynb` (append cells)

**Interfaces:**
- Consumes: `MODELS`, `CLUSTER`, `PREV`, `IND`, `z_means`, `run_ablated`, `ablation_hooks`, `T`, `TOKENS`, `ZM`.
- Produces:
  - `NAT: dict[str, Tensor]` — natural-text token batch per model `[n_docs, 256]`
  - `ZM_NAT: dict[str, dict[int, Tensor]]` — natural-text reference means
  - `BIGRAM: dict[str, Tensor]` — bool mask `[n_docs, 255]`
  - `NATDELTA: dict[str, dict]` — `{"repeated": float, "elsewhere": float, "clean": float}`
  - `DLA: dict[str, dict]` — `{"clean": float, "feeder_ablated": float}`

- [ ] **Step 1: Verify the corpus is present, with a streaming fallback**

The Pile cache `notebooks/in_context_learning/checkpoints/icl_pile/corpus.txt` exists on this
machine but is **gitignored**, so the notebook must regenerate it if absent. Check:

```bash
ls -la notebooks/in_context_learning/checkpoints/icl_pile/corpus.txt
```

- [ ] **Step 2: Append the Act 4 cells**

Markdown:

```
## Act 4 — Does any of this matter on real text?

Repeated random tokens are a laboratory instrument: they isolate induction by making it the *only* usable strategy. Real text is not like that. A model predicting English has grammar, frequency, and semantics to lean on — induction is one contributor among many.

Olsson et al. make this move too, measuring per-token loss on natural text rather than only on synthetic sequences. So: ablate the whole induction cluster and measure the per-token loss increase on Pile text, split two ways —

- positions where the bigram ending here **has already appeared** in this document (induction has something to copy), versus
- **everywhere else** (it does not).

If the cluster is doing induction rather than generic work, the damage should concentrate on the first group. This is also the control for Act 2's caveat: 18 heads is a big lesion, and this split says whether what we removed was induction-specific.

Reference means are recomputed **on natural text** here — mean-ablation is only on-distribution with respect to the distribution the means came from.
```

Code cell (corpus + mask):

```python
import os

CORPUS = "notebooks/in_context_learning/checkpoints/icl_pile/corpus.txt"
N_DOCS, N_CTX = 8, 256

if os.path.exists(CORPUS):
    with open(CORPUS, encoding="utf-8") as f:
        raw = f.read(2_000_000)
    print(f"[corpus] cache hit: {len(raw):,} chars from {CORPUS}")
else:
    from datasets import load_dataset

    stream = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    parts, total = [], 0
    for row in stream:
        parts.append(row["text"])
        total += len(row["text"]) + 1
        if total >= 2_000_000:
            break
    raw = "\n".join(parts)
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    with open(CORPUS, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"[corpus] streamed + cached {len(raw):,} chars -> {CORPUS}")

docs = [d for d in raw.split("\n") if len(d) > 2000][:N_DOCS]
print(f"[corpus] {len(docs)} documents; first 100 chars: {docs[0][:100]!r}")


def repeated_bigram_mask(tokens):
    """mask[b, i] = True if bigram (tok[i-1], tok[i]) already appeared -> induction can act."""
    b, n = tokens.shape
    mask = torch.zeros(b, n - 1, dtype=torch.bool)
    for bi in range(b):
        seen, seq = set(), tokens[bi].tolist()
        for i in range(1, n - 1):
            key = (seq[i - 1], seq[i])
            if key in seen:
                mask[bi, i] = True
            seen.add(key)
    return mask


NAT, BIGRAM, ZM_NAT = {}, {}, {}
for name, model in MODELS.items():
    NAT[name] = model.to_tokens(docs)[:, :N_CTX]
    BIGRAM[name] = repeated_bigram_mask(NAT[name])
    ZM_NAT[name] = z_means(model, NAT[name])
    print(f"{name}: natural batch {tuple(NAT[name].shape)}, "
          f"{BIGRAM[name].float().mean():.1%} of positions are repeated bigrams")
```

Code cell (delta split):

```python
NATDELTA = {}
for name, model in MODELS.items():
    clean = model(NAT[name], return_type="loss", loss_per_token=True).cpu()
    abl = run_ablated(model, NAT[name], CLUSTER[name], ZM_NAT[name]).cpu()
    d = abl - clean
    m = BIGRAM[name]
    NATDELTA[name] = {
        "clean": clean.mean().item(),
        "repeated": d[m].mean().item(),
        "elsewhere": d[~m].mean().item(),
    }
    print(f"{name}: clean natural loss {NATDELTA[name]['clean']:.3f} nats;  "
          f"cluster-ablation delta: repeated-bigram {NATDELTA[name]['repeated']:+.3f}  "
          f"elsewhere {NATDELTA[name]['elsewhere']:+.3f}")

fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
width = 0.36
for i, name in enumerate(MODELS):
    ax.bar([x + (i - 0.5) * width for x in range(2)],
           [NATDELTA[name]["repeated"], NATDELTA[name]["elsewhere"]], width, label=name)
ax.set_xticks(range(2))
ax.set_xticklabels(["repeated bigram\n(induction can act)", "everywhere else"])
ax.set_ylabel("per-token loss increase (nats)")
ax.set_title("Cluster ablation on Pile text — where does the damage land?")
ax.legend()
plt.show()
```

Code cell (DLA):

```python
def dla_matched(model, name, fwd_hooks=()):
    """Matched-token logit contribution of the induction head: W_U-projection of its output.

    LayerNorm's final scale is not applied — this is the head's raw write into the logits.
    """
    l, h = IND[name]
    with model.hooks(fwd_hooks=list(fwd_hooks)):
        _, cache = model.run_with_cache(
            TOKENS[name], return_type=None, names_filter=utils.get_act_name("z", l)
        )
    out = cache["z", l][:, :, h, :] @ model.W_O[l, h]     # [batch, pos, d_model]
    qs = torch.arange(T + 1, 2 * T)                       # second-copy queries with a next token
    tgt = TOKENS[name][:, qs + 1]                         # the token induction should copy
    w = model.W_U[:, tgt]                                 # [d_model, batch, Q]
    return (out[:, qs, :] * w.permute(1, 2, 0)).sum(-1).mean().item()


DLA = {}
for name, model in MODELS.items():
    DLA[name] = {
        "clean": dla_matched(model, name),
        "feeder_ablated": dla_matched(model, name, ablation_hooks([PREV[name]], ZM[name])),
    }
    li, hi = IND[name]
    lp, hp = PREV[name]
    print(f"{name}: L{li}H{hi} matched-token logit contribution  clean {DLA[name]['clean']:+.3f}"
          f"  ->  L{lp}H{hp}-ablated {DLA[name]['feeder_ablated']:+.3f}")
```

- [ ] **Step 3: Execute**

Run nbconvert + spot-check. Expected (gpt2): ~21% repeated-bigram positions; delta +0.517
repeated vs +0.049 elsewhere; DLA +4.769 → +3.387.

- [ ] **Step 4: Append results markdown**

Report the split with real numbers and the ratio. The claim it licenses: the lesion is
induction-specific — damage concentrates ~10× on positions where induction has something to
copy, so Act 2's 81% was not generic damage. Then the DLA beat: the induction head's direct
push on the matched token's logit, and how much of that push survives when its feeder is cut
(Act 3's mediation, now measured in logits rather than attention). Name the LayerNorm caveat.

- [ ] **Step 5: Commit**

```bash
git add notebooks/in_context_learning/induction_ablation.ipynb
git commit -m "induction-ablation Act 4: the lesion is induction-specific — cluster ablation costs <X> nats at repeated bigrams vs <Y> elsewhere on Pile text

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Act 5 (stretch) — Purify the copier with OV eigenvalue surgery

**Files:**
- Modify: `notebooks/in_context_learning/induction_ablation.ipynb` (append cells)

**Interfaces:**
- Consumes: `pythia`, `MODELS`, `TOKENS`, `NAT`, `IND`, `loss_gap`, `BASE`, `T`.
- Produces: `SURGERY: dict` — `{"score_before": float, "score_after": float, "frac_pos_before": float, "frac_pos_after": float, "gap_before": float, "gap_after": float, "nat_before": float, "nat_after": float, "cond": float}`

**Critical mechanics (verified — do not re-derive):**
- The 64×64 full-OV circuit is `M = (W_O[l,h] @ W_U) @ (W_E @ W_V[l,h])`.
- Spectral projector: `M = V Λ V⁻¹`; with `D = diag(Re(λ) > 0)`, `P = V D V⁻¹` is **real**
  (measured imaginary residual 2.4e-06) because conjugate pairs share a real part and are
  kept or dropped together. `cond(V) ≈ 74` — well-conditioned, no defective-matrix blowup.
- Writing back `W_V[l,h] ← W_V[l,h] @ P` makes the composed circuit `M P = V Λ D V⁻¹` —
  exactly the positive-eigendirection reconstruction, with `W_O` untouched.
- `model.W_V` is a read-only stacked property; write through
  `model.blocks[l].attn.W_V.data[h]` (dtype float32, shape `[d_model, d_head]` per head).
- **Copying score after surgery is ~0.759, not 1.0** — see Global "two mechanics facts".

- [ ] **Step 1: Append the Act 5 cells**

Markdown:

```
## Act 5 — Purify the copier

Every ablation so far has been subtractive: remove a head, see what breaks. This act does the opposite — it *edits* a head to be more like the hand-crafted one, and asks what that costs.

The previous notebook's weakest reproduction was Pythia L4H6's copying OV: score **0.423**, with only 69% of its eigenvalue mass on the positive side. The toy's `W_V2 = I×4` scores 1.000. The natural reading is **superposition**: those 64 OV dimensions are not a dedicated copier, they are doing copying *and other jobs at once*.

That reading makes a prediction we can test by surgery. Eigendecompose the 64×64 OV circuit, keep only the eigendirections with positive real part, and write the result back into `W_V`. If the negative mass is genuinely other work, the model should keep its induction ability while losing something elsewhere.

The projector `P = V·diag(Re(λ)>0)·V⁻¹` is real (conjugate eigenvalue pairs share a real part, so they are kept or dropped together) and `W_V ← W_V·P` makes the composed OV circuit exactly the positive-eigendirection reconstruction. The original weights are snapshotted first, so this cell restores the model and stays re-runnable.
```

Code:

```python
l, h = IND["pythia"]
W_V_orig = pythia.blocks[l].attn.W_V.data[h].clone()   # snapshot: this cell must be re-runnable


def full_ov_eigs(model, l, h):
    small = ((model.W_O[l, h] @ model.W_U) @ (model.W_E @ model.W_V[l, h])).float()
    return torch.linalg.eig(small)


def copying_score(eigs):
    return (eigs.real.sum() / eigs.abs().sum()).item()


eigs, V = full_ov_eigs(pythia, l, h)
keep = eigs.real > 0
P = V @ torch.diag(keep.to(V.dtype)) @ torch.linalg.inv(V)
SURGERY = {
    "cond": torch.linalg.cond(V).item(),
    "imag_residual": P.imag.abs().max().item(),
    "score_before": copying_score(eigs),
    "frac_pos_before": keep.float().mean().item(),
    "gap_before": BASE["pythia"]["gap"],
    "nat_before": pythia(NAT["pythia"], return_type="loss").item(),
}
print(f"projector: cond(V)={SURGERY['cond']:.1f}  max|Im(P)|={SURGERY['imag_residual']:.2e}"
      f"  -> keeping {int(keep.sum())}/{len(keep)} eigendirections")

pythia.blocks[l].attn.W_V.data[h] = (W_V_orig.float() @ P.real).to(W_V_orig.dtype)

eigs_after = torch.linalg.eigvals(
    ((pythia.W_O[l, h] @ pythia.W_U) @ (pythia.W_E @ pythia.W_V[l, h])).float()
)
lpt = pythia(TOKENS["pythia"], return_type="loss", loss_per_token=True)
SURGERY.update({
    "score_after": copying_score(eigs_after),
    "frac_pos_after": (eigs_after.real > 0).float().mean().item(),
    "gap_after": loss_gap(lpt),
    "nat_after": pythia(NAT["pythia"], return_type="loss").item(),
})

pythia.blocks[l].attn.W_V.data[h] = W_V_orig            # restore
_restored = loss_gap(pythia(TOKENS["pythia"], return_type="loss", loss_per_token=True))

print(f"copying score   {SURGERY['score_before']:.3f} -> {SURGERY['score_after']:.3f}"
      f"   ({SURGERY['frac_pos_before']:.0%} -> {SURGERY['frac_pos_after']:.0%} positive)")
print(f"induction gap   {SURGERY['gap_before']:.3f} -> {SURGERY['gap_after']:.3f} nats")
print(f"natural loss    {SURGERY['nat_before']:.3f} -> {SURGERY['nat_after']:.3f} nats")
print(f"restored gap    {_restored:.3f}  (matches clean {BASE['pythia']['gap']:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
for ax, (e, title) in zip(axes, [(eigs, "before"), (eigs_after, "after surgery")]):
    ax.scatter(e.real.cpu().detach(), e.imag.cpu().detach(), s=14)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(f"Pythia L{l}H{h} full-OV spectrum — {title}")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
plt.show()
```

- [ ] **Step 2: Execute**

Run nbconvert + spot-check.
Expected: score 0.423 → ~0.759; gap 17.999 → ~16.979; natural 3.366 → ~3.379; restored gap
equals clean. If the projector's `cond(V)` is large or `max|Im(P)|` is not ~1e-6, the
reconstruction is unreliable — report that with the named reason (per the spec's failure
handling) rather than presenting the numbers as clean.

- [ ] **Step 3: Append results markdown — H3 is not confirmed; say so**

The measured outcome contradicts H3's clean form and must be reported as such:

- The surgery **does** purify the copier (0.423 → 0.759).
- But induction got slightly **worse**, not better or equal (gap 17.999 → 16.979, ~−6%).
- And natural-text loss rose only slightly (3.366 → 3.379, ~+0.4%).

So the prediction "preserves or improves induction while degrading elsewhere" is **not
borne out**: the negative-eigenvalue directions were contributing a little to induction
itself, not purely to other work. Superposition is real but not cleanly separable along
this axis.

Also explain the 0.759 ceiling: the dropped directions are cleanly zeroed (residual |λ| ≈
1e-6 against a kept max of ~27), so what remains between 0.759 and 1.0 is **imaginary**
eigenvalue mass — rotation, not anti-copying. `Σ Re(λ)/Σ|λ| < 1` for any complex spectrum.
"Positive-real-part" and "pure copier" are not the same edit.

- [ ] **Step 4: Commit**

```bash
git add notebooks/in_context_learning/induction_ablation.ipynb
git commit -m "induction-ablation Act 5: OV surgery purifies Pythia L4H6's copier 0.423->0.759 but costs induction ~6% — the negative eigendirections were not purely other work

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Act 6 — Verdict table + honest gaps

**Files:**
- Modify: `notebooks/in_context_learning/induction_ablation.ipynb` (append cells)

**Interfaces:**
- Consumes: `LADDER`, `MED`, `NATDELTA`, `DLA`, `SURGERY`, `HYDRA`, `SCORES`, `CLUSTER`, `IND`, `PREV`, `BASE`, `CALIB`.
- Produces: the final verdict DataFrame; nothing downstream.

- [ ] **Step 1: Append the verdict cell**

Markdown: `## Act 6 — Verdict` with one line framing the table as intervention × model ×
gap destroyed.

Code:

```python
import pandas as pd

rows = []
for name in MODELS:
    li, hi = IND[name]
    lp, hp = PREV[name]
    rows.append({
        "model": name,
        "clean gap (nats)": f"{BASE[name]['gap']:.2f}",
        f"ablate induction head": f"L{li}H{hi}: {LADDER[name]['top-1']:.1%}",
        "ablate top-3": f"{LADDER[name]['top-3']:.1%}",
        f"ablate cluster": f"n={len(CLUSTER[name])}: {LADDER[name]['cluster']:.1%}",
        f"ablate prev-token head": f"L{lp}H{hp}: {MED[name]['collapse']:.1%}",
        "stripe under feeder ablation": f"{MED[name]['stripe_clean']:.2f} -> {MED[name]['stripe_abl']:.2f}",
        "natural text (rep. bigram / else)": f"{NATDELTA[name]['repeated']:+.2f} / {NATDELTA[name]['elsewhere']:+.2f}",
    })
verdict = pd.DataFrame(rows).set_index("model")
display(verdict.T)
```

- [ ] **Step 2: Execute and verify the table renders**

Run nbconvert + spot-check.

- [ ] **Step 3: Append the closing markdown — answer the blog's promise directly**

Write `## Honest gaps, and the answer to the blog's promise` covering, with real numbers:

- **The promise, answered.** The blog said: knock out L4H11 and L5H5, or L3H2 and L4H6, and
  watch whether the induction score collapses. Answer: **it does not.** Removing the single
  best induction head destroys ~0.7% (GPT-2) / ~4.4% (Pythia) of the behavior. The named
  wrinkle was the whole story — induction lives in a cluster, and a twin head strengthens to
  cover the loss (GPT-2's L6H9: 0.917 → 0.963 the moment L5H5 is cut).
- **What *is* causal.** The full cluster (18/21 heads) destroys ~81%/85%, and Act 4 shows
  that damage is induction-specific (~10× concentrated at repeated bigrams). The
  previous-token head's mediation effect confirms K-composition causally: cutting the feeder
  drops the untouched induction head's stripe 0.928 → 0.636 and its matched-token logit push
  +4.77 → +3.39 (GPT-2).
- **What the surgery says.** Act 5's H3 was not confirmed — purifying the OV cost induction
  a little rather than nothing, so the negative eigendirections were not purely other work.
- **Named limitations.** Mean-ablation over a synthetic reference is one ablation family
  (resample/patch ablation would isolate different information); the cluster is defined by an
  attention-score gate of 0.2, an arbitrary threshold whose membership drives the 81% figure;
  ablating 18–21 heads at once cannot separate "induction died" from "layers 5–11 got badly
  damaged" beyond what Act 4's split shows; the hydra check measures backup *attention*, not
  backup *output*; DLA ignores the final LayerNorm scale; everything is a single seed and a
  single 32×101 synthetic batch plus 8 Pile documents; no path patching, so the mediation
  result shows the feeder matters, not the complete route by which it matters.
- **Handoff.** The natural next rung is path patching (Wang et al.'s method) to map the full
  route rather than cut one edge, and resample ablation to separate "this head's information"
  from "this head's presence".

- [ ] **Step 4: Final full execution, timed**

```bash
time uv run --no-sync jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  notebooks/in_context_learning/induction_ablation.ipynb
```
Expected: exits 0, under 10 minutes. Report the actual runtime.

- [ ] **Step 5: Verify notebook hygiene**

```bash
uv run --no-sync python -c "
import nbformat
nb = nbformat.read('notebooks/in_context_learning/induction_ablation.ipynb', as_version=4)
code = sum(1 for c in nb.cells if c.cell_type == 'code')
md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
print(f'{code} code cells, {md} markdown cells')
bad = [w for c in nb.cells for w in ('CLAUDE.md', 'AGENTS.md', 'TODO', 'TBD') if w in ''.join(c.source)]
print('process-talk/placeholder scan:', bad or 'clean')
"
```
Expected: `clean`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/in_context_learning/induction_ablation.ipynb
git commit -m "induction-ablation Act 6: the blog's promise answered — knocking out 'the' induction head collapses induction ~1-4%, not at all; the circuit is a hydra of 18-21 heads

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Report to the user and stop**

Do **not** merge. Summarize the verdict table and offer the
`superpowers:finishing-a-development-branch` options (PR vs merge).

---

## Self-review notes

- **Spec coverage:** Act 0 yardstick → Task 1; Act 1 scalpel calibration (mean vs zero) →
  Task 2; Act 2a/2b/2c ladder + hydra → Task 3; Act 3 mediation → Task 4; Act 4 natural text
  + DLA → Task 5; Act 5 OV surgery → Task 6; Act 6 verdict + honest gaps → Task 7. Metrics
  (loss gap, induction attention score, natural-text delta, DLA) defined once in Tasks 1/3/5
  and reused. Mean-ablation reference (synthetic for Acts 2–3, natural-text for Act 4) → Tasks
  2 and 5. Failure handling for Act 5 instability → Task 6 Step 2. Out-of-scope items
  (gradient fine-tuning, path patching, resample ablation as primary, Masri) appear in no
  task — path patching and resample appear only as named handoffs in Task 7. ✓
- **Deviations from the spec, deliberate:**
  1. **Act 4 corpus.** The spec says "the same corpus sample as the wild notebook", but the
     wild notebook has no corpus — its only natural text is one hand-written sentence, far
     too small for a per-token loss delta split by repeated bigrams. Task 5 uses the Pile
     cache (`monology/pile-uncopyrighted`) that `icl_from_scratch.ipynb` already streams,
     with a regeneration fallback since the cache is gitignored.
  2. **Ladder middle rung.** The spec's ladder jumps top-1 → whole cluster (1 → 18 heads).
     Task 3 adds a top-3 rung; it is one extra forward pass and makes "how distributed?"
     answerable rather than binary.
  3. **Pythia hydra runner-up** is computed by argsort over the measured delta rather than
     "read off the wild notebook's Act 1 heatmap" (a baked PNG); this is more robust and
     gives the same information.
  4. **Act 1's framing** is corrected against measurement: the spec implies mean-ablation is
     the sharper tool, but both mean- and zero-ablation collapse ~0% at single-head scale.
     Task 2 keeps the pedagogical beat but states the measured non-difference honestly.
- **Type consistency:** `IND`/`PREV: dict[str, tuple[int,int]]` defined in Task 2, consumed
  by Tasks 3–6. `ZM: dict[str, dict[int, Tensor]]` (Task 2) vs `ZM_NAT` (Task 5) — both
  keyed the same, both accepted by `run_ablated(model, tokens, targets, zm)`. `ind_scores`
  defined once in Task 3, reused in the hydra check. `LADDER[name]` keys `"top-1"`,
  `"top-3"`, `"cluster"` written in Task 3, read in Task 7. `collapse(gap, name)` signature
  fixed in Task 1 and used unchanged throughout. ✓
- **Known API risk:** `model.hooks(fwd_hooks=…)` as a context manager wrapping
  `run_with_cache` is the only way to cache *under* an ablation; verified working on
  transformer_lens in this venv (both the hydra check and the mediation readout use it).
  `model.blocks[l].attn.W_V.data[h]` write-back verified, including exact restore.
