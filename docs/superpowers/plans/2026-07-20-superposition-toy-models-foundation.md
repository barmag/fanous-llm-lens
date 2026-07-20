# Toy Models of Superposition — Foundation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single self-contained narrative notebook that reproduces the core of
Elhage et al. 2022 (*Toy Models of Superposition*) far enough to prove superposition is
real and driven by sparsity.

**Architecture:** A tiny synthetic ReLU autoencoder (`x' = ReLU(WᵀWx + b)`) trained on
sparse features. The notebook follows the paper's own build order: linear ceiling →
ReLU+dense → crank sparsity → pentagon (n=5, m=2) → quantified sparsity sweep
(n=20, m=5). Reusable functions live in small "library" cells (marked `# lib:`) so a
pytest file can extract and unit-test them the way the repo already tests notebook cells
(cf. `tests/education/test_stage1b_graph.py`). Narrative/driver/plot cells are validated
by executing the whole notebook headless.

**Tech Stack:** Python, PyTorch (`2.5.1+rocm6.2`), NumPy, Matplotlib, Jupyter/nbconvert,
pytest. All already in the repo — no new dependencies.

## Global Constraints

- **No new dependencies.** torch, numpy, matplotlib, pytest, jupyter/nbconvert only.
- **Run everything through the venv:** prefix commands with `uv run --no-sync` (bare
  `uv run` drops the ROCm torch and camel-tools).
- **Faithful to the paper.** Flagship geometric demo: `n=5, m=2`. Quantified
  demonstration: `n=20, m=5, importance Iᵢ = 0.7ⁱ`. Geometric-reveal sparsities:
  `S ∈ {0.0, 0.8, 0.9}` (matches the intro figure's 0% / 80% / 90%). Data:
  `xᵢ = 0` with prob `S`, else uniform`[0,1)`. Loss: importance-weighted MSE
  `mean_batch Σᵢ Iᵢ (xᵢ − x'ᵢ)²`.
- **Training config (stated replication values):** Adam, `lr=1e-3`, `steps=10_000`,
  `batch_size=1024`, fixed seeds. The paper does not print optimizer steps in-text; these
  match the public replication conventions and keep runtime to seconds on CPU.
- **Reproducibility:** fixed `SEED = 0`; weight init and every batch drawn from seeded
  `torch.Generator`s. Log the seed in the notebook.
- **Honest negatives are results.** No hard pass/fail gate that hides a negative. If the
  pentagon does not form at our seed (the paper notes these toys hit local minima), the
  notebook reports what it got and why (seed sensitivity), and the driver `assert`s use
  loose thresholds that check the *phenomenon* (more features represented under sparsity),
  not an exact geometry.
- **Notebook hygiene:** markdown is pedagogical only — no process-talk, no citing repo
  conventions inside cells. Small, single-purpose, re-runnable cells. Clear all outputs
  before commit (`uv run --no-sync jupyter nbconvert --clear-output --inplace <nb>` then
  `nbstripout <nb>`).
- **nbconvert safety:** to VERIFY a notebook runs, execute it to a scratch output path
  (`--to notebook --execute --output /tmp/.../out.ipynb`), never `--inplace` while the
  notebook is also being edited (a concurrent in-place execute clobbers edits). Use
  `--clear-output --inplace` only as the final pre-commit step when no edit is in flight.
  Run nbconvert from the repo root.
- **No Arabic/Masri, no real model, no TransformerLens, no network at run time.**

## File Structure

- **Create** `notebooks/superposition/toy_models_of_superposition.ipynb` — the narrative
  notebook. Reusable functions in `# lib:`-marked code cells; everything else is
  markdown, driver, or plot cells.
- **Create** `tests/superposition/test_toy_models.py` — extracts and unit-tests the
  `# lib:` cells headless (no training in the fast tests except one tiny convergence
  check).
- **Reference (already committed)** `docs/papers/elhage2022-toy-models-superposition.html`.

### The `# lib:` cell convention

Every reusable function lives in its own code cell whose **first line** is a marker
comment: `# lib: <name>`. The test harness execs, in notebook order, exactly the cells
whose stripped source starts with `# lib:` — robust to inserting narrative cells and to
reordering. Driver cells (training runs) and plot cells are **not** marked `# lib:` and
are only exercised by the full-notebook headless execute.

### Shared test harness (referenced by every test task)

`tests/superposition/test_toy_models.py` opens with this loader. Later tasks add test
functions below it; they do not redefine it.

```python
"""Unit tests for the library cells of the Toy Models of Superposition notebook.

We exec the `# lib:`-marked code cells straight out of the notebook into a fresh
namespace, so the tests need no GPU and (except one tiny convergence check) no training.
"""
import json
from pathlib import Path

import torch

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "superposition" / "toy_models_of_superposition.ipynb"
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
```

---

### Task 1: Scaffold the notebook — Act 0 (the puzzle) + imports library cell

**Files:**
- Create: `notebooks/superposition/toy_models_of_superposition.ipynb`

**Interfaces:**
- Produces: an importable `# lib: imports` cell exposing `torch`, `F`, `np`, `plt`,
  and `SEED = 0` into the notebook namespace.

- [ ] **Step 1: Create the notebook with the title + thesis markdown cell**

Markdown cell (cell 0):

```markdown
# Toy Models of Superposition

**Thesis.** A nonlinear model given *sparse* inputs will represent more features than it
has dimensions — *superposition* — and sparsity is the knob that turns it on. The cost is
*interference*.

This notebook rebuilds the core of Elhage et al. (2022), *Toy Models of Superposition*,
following the paper's own build: a linear ceiling, then a ReLU model on dense data, then
we crank sparsity and watch five features arrange themselves as a pentagon in two
dimensions. Everything here is synthetic — no language model, no Arabic — because the
goal is to understand the *mechanism*.
```

- [ ] **Step 2: Add the Act 0 "puzzle" markdown cell**

```markdown
## Act 0 — The puzzle

In real networks, single neurons are often *polysemantic*: one neuron responds to several
unrelated things. Why would a model do that instead of giving each concept its own
neuron? The paper's answer is **superposition** — when features are rare (sparse), a model
can pack more of them than it has dimensions by storing them in overlapping directions,
paying a little *interference* in return. We'll reproduce that from scratch and find the
exact knob — sparsity — that decides whether it happens.
```

- [ ] **Step 3: Add the imports library cell**

```python
# lib: imports
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
print("torch", torch.__version__, "| seed", SEED)
```

- [ ] **Step 4: Verify the notebook executes headless**

Run:
```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_check.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
```
Expected: exits 0; prints `torch 2.5.1+rocm6.2 | seed 0` in the executed copy.

- [ ] **Step 5: Clear outputs and commit**

```bash
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb
git commit -m "superposition: scaffold notebook — Act 0 puzzle + imports"
```

---

### Task 2: Data generator + importance weights (library cells + tests)

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb`
- Create: `tests/superposition/test_toy_models.py`

**Interfaces:**
- Consumes: `torch` from the `# lib: imports` cell.
- Produces:
  - `make_batch(n_features, sparsity, batch_size, generator=None) -> Tensor[batch_size, n_features]`
  - `importance_weights(n_features, decay=0.7) -> Tensor[n_features]`

- [ ] **Step 1: Add the Act 1 opener markdown cell**

```markdown
## Act 1 — Features, and the linear ceiling

The paper models each feature as a direction in activation space. Our synthetic data
mirrors what it assumes real features look like: mostly-zero (sparse) and non-negative.
Each feature is zero with probability `S` (the sparsity) and otherwise uniform on
`[0, 1)`. Each feature also has an *importance* — a scalar weight on its share of the
loss — that decays across features, so the model has to make trade-offs.
```

- [ ] **Step 2: Add the `make_batch` library cell**

```python
# lib: make_batch
def make_batch(n_features, sparsity, batch_size, generator=None):
    """Sparse synthetic features.

    Each entry is 0 with probability `sparsity`, otherwise uniform on [0, 1).
    Returns a tensor of shape [batch_size, n_features].
    """
    vals = torch.rand(batch_size, n_features, generator=generator)
    keep = torch.rand(batch_size, n_features, generator=generator) >= sparsity
    return vals * keep
```

- [ ] **Step 3: Add the `importance_weights` library cell**

```python
# lib: importance
def importance_weights(n_features, decay=0.7):
    """Importance Iᵢ = decay**i, shape [n_features]. Matches the paper's Iᵢ = 0.7^i."""
    return decay ** torch.arange(n_features, dtype=torch.float32)
```

- [ ] **Step 4: Add a small demo cell (not a lib cell) showing a batch**

```python
gen = torch.Generator().manual_seed(SEED)
demo = make_batch(n_features=5, sparsity=0.8, batch_size=8, generator=gen)
print("batch shape:", tuple(demo.shape), "| fraction nonzero:", (demo > 0).float().mean().item())
print("importance:", importance_weights(5).tolist())
```

- [ ] **Step 5: Write the failing tests**

Create `tests/superposition/test_toy_models.py` with the shared harness from the "Shared
test harness" section above, then append:

```python
def test_make_batch_shape_and_range():
    lib = load_lib()
    gen = torch.Generator().manual_seed(0)
    x = lib["make_batch"](n_features=6, sparsity=0.5, batch_size=100, generator=gen)
    assert x.shape == (100, 6)
    assert x.min().item() >= 0.0
    assert x.max().item() < 1.0


def test_make_batch_sparsity_fraction():
    lib = load_lib()
    gen = torch.Generator().manual_seed(0)
    x = lib["make_batch"](n_features=10, sparsity=0.9, batch_size=20_000, generator=gen)
    frac_nonzero = (x > 0).float().mean().item()
    assert abs(frac_nonzero - 0.1) < 0.02  # ~10% survive at S=0.9


def test_importance_decay():
    lib = load_lib()
    imp = lib["importance_weights"](5, decay=0.7)
    assert imp.shape == (5,)
    assert abs(imp[0].item() - 1.0) < 1e-6
    assert abs(imp[3].item() - 0.7 ** 3) < 1e-6
```

- [ ] **Step 6: Run the tests to verify they pass** (the lib cells already exist)

Run:
```bash
uv run --no-sync pytest tests/superposition/test_toy_models.py -q
```
Expected: 3 passed.

- [ ] **Step 7: Verify the notebook still executes headless**

Run:
```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_check.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
```
Expected: exits 0; the demo cell prints a shape of `(8, 5)`.

- [ ] **Step 8: Clear outputs and commit**

```bash
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb tests/superposition/test_toy_models.py
git commit -m "superposition: sparse data generator + importance weights, with tests"
```

---

### Task 3: The toy model (library cell + tests)

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb`
- Modify: `tests/superposition/test_toy_models.py`

**Interfaces:**
- Consumes: `torch`, `F` from imports.
- Produces: `ToyModel(n_features, n_hidden, use_relu=True)` — `torch.nn.Module` with
  `.W` (`Parameter[n_hidden, n_features]`), `.b` (`Parameter[n_features]`), and
  `forward(x: Tensor[B, n_features]) -> Tensor[B, n_features]` computing
  `h = x @ W.T`, `out = h @ W + b`, then `ReLU` iff `use_relu`.

- [ ] **Step 1: Add the model explanation markdown cell**

```markdown
### The model

We embed `n` features into `m < n` dimensions with a matrix `W` of shape `[m, n]` (column
`i` is feature `i`'s direction), then read them back out through `Wᵀ` and add a bias:

- **Linear model:** `x' = Wᵀ W x`. It can represent at most `m` features orthogonally —
  superposition is impossible by construction (`WᵀW` is invertible ⟺ no superposition).
- **ReLU model:** `x' = ReLU(Wᵀ W x + b)`. The non-linearity is what lets it tolerate the
  interference that superposition creates.
```

- [ ] **Step 2: Add the `ToyModel` library cell**

```python
# lib: toymodel
class ToyModel(torch.nn.Module):
    """Embed n features into m<n dims via W [m, n], read back through Wᵀ, add bias.

    forward: h = x @ W.T ; out = h @ W + b ; ReLU(out) if use_relu else out.
    """
    def __init__(self, n_features, n_hidden, use_relu=True):
        super().__init__()
        self.use_relu = use_relu
        self.W = torch.nn.Parameter(torch.empty(n_hidden, n_features))
        torch.nn.init.xavier_normal_(self.W)
        self.b = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        h = x @ self.W.T           # [B, m]
        out = h @ self.W + self.b  # [B, n]
        return F.relu(out) if self.use_relu else out
```

- [ ] **Step 3: Add a shape/smoke demo cell (not a lib cell)**

```python
m = ToyModel(n_features=5, n_hidden=2)
gen = torch.Generator().manual_seed(SEED)
xb = make_batch(5, sparsity=0.8, batch_size=4, generator=gen)
print("W:", tuple(m.W.shape), "| b:", tuple(m.b.shape), "| out:", tuple(m(xb).shape))
```

- [ ] **Step 4: Write the failing tests**

Append to `tests/superposition/test_toy_models.py`:

```python
def test_toymodel_param_shapes():
    lib = load_lib()
    model = lib["ToyModel"](n_features=5, n_hidden=2)
    assert tuple(model.W.shape) == (2, 5)
    assert tuple(model.b.shape) == (5,)


def test_toymodel_forward_shape():
    lib = load_lib()
    model = lib["ToyModel"](n_features=5, n_hidden=2)
    x = torch.rand(7, 5)
    assert tuple(model(x).shape) == (7, 5)


def test_relu_is_nonnegative_linear_can_be_negative():
    lib = load_lib()
    torch.manual_seed(0)
    x = torch.rand(32, 5)
    relu_model = lib["ToyModel"](5, 2, use_relu=True)
    lin_model = lib["ToyModel"](5, 2, use_relu=False)
    # force a negative bias so the pre-activation has negative entries
    with torch.no_grad():
        relu_model.b.fill_(-1.0)
        lin_model.b.copy_(relu_model.b)
        lin_model.W.copy_(relu_model.W)
    assert relu_model(x).min().item() >= 0.0
    assert lin_model(x).min().item() < 0.0
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
uv run --no-sync pytest tests/superposition/test_toy_models.py -q
```
Expected: 6 passed.

- [ ] **Step 6: Verify headless execute, clear outputs, commit**

```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_check.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb tests/superposition/test_toy_models.py
git commit -m "superposition: ToyModel (linear + ReLU output), with tests"
```

---

### Task 4: Training loop + feature norms + Act 1 linear-ceiling result

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb`
- Modify: `tests/superposition/test_toy_models.py`

**Interfaces:**
- Consumes: `make_batch`, `importance_weights`, `ToyModel`.
- Produces:
  - `train(model, sparsity, importance, steps=10_000, lr=1e-3, batch_size=1024, seed=0) -> list[float]`
    (importance-weighted MSE; trains in place; returns loss samples every 500 steps).
  - `feature_norms(model) -> Tensor[n_features]` (L2 norm of each column of `W`).

- [ ] **Step 1: Add the `train` library cell**

```python
# lib: train
def train(model, sparsity, importance, steps=10_000, lr=1e-3, batch_size=1024, seed=0):
    """Train `model` to reconstruct sparse features under importance-weighted MSE.

    Loss = mean over batch of Σᵢ importanceᵢ · (xᵢ − x'ᵢ)². Returns loss sampled every
    500 steps.
    """
    n_features = model.W.shape[1]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x = make_batch(n_features, sparsity, batch_size, generator=gen)
        out = model(x)
        loss = (importance * (x - out) ** 2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            losses.append(loss.item())
    return losses
```

- [ ] **Step 2: Add the `feature_norms` library cell**

```python
# lib: feature_norms
def feature_norms(model):
    """L2 norm of each feature's embedding column (how strongly it is represented)."""
    return model.W.detach().norm(dim=0)
```

- [ ] **Step 3: Add the Act 1 result cell (linear model, dense, n=5 m=2)**

```python
imp5 = importance_weights(5)                       # Iᵢ = 0.7^i
lin = ToyModel(5, 2, use_relu=False)
train(lin, sparsity=0.0, importance=imp5, seed=SEED)
norms = feature_norms(lin)
print("linear / dense feature norms:", [round(v, 3) for v in norms.tolist()])
n_represented = int((norms > 0.5 * norms.max()).sum())
print("features clearly represented:", n_represented)
assert n_represented <= 2, "a 2-dim linear model should keep at most its top-2 features"
```

- [ ] **Step 4: Add the interpretation markdown cell**

```markdown
The linear model behaves like PCA: with two dimensions it keeps only the two most
important features and discards the rest. This is the ceiling — no non-linearity, so no
way to tolerate interference, so no superposition. Next we add the ReLU.
```

- [ ] **Step 5: Write the failing tests**

Append to `tests/superposition/test_toy_models.py`:

```python
def test_train_reduces_loss():
    lib = load_lib()
    torch.manual_seed(0)
    model = lib["ToyModel"](3, 2, use_relu=True)
    imp = lib["importance_weights"](3)
    losses = lib["train"](model, sparsity=0.0, importance=imp, steps=1000, seed=0)
    assert losses[-1] < losses[0]


def test_feature_norms_shape():
    lib = load_lib()
    model = lib["ToyModel"](5, 2)
    fn = lib["feature_norms"](model)
    assert fn.shape == (5,)
    assert (fn >= 0).all()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```bash
uv run --no-sync pytest tests/superposition/test_toy_models.py -q
```
Expected: 8 passed.

- [ ] **Step 7: Verify headless execute (the in-cell assert is the notebook's own gate)**

Run:
```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_check.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
```
Expected: exits 0 (the `n_represented <= 2` assert holds).

- [ ] **Step 8: Clear outputs and commit**

```bash
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb tests/superposition/test_toy_models.py
git commit -m "superposition: training + feature norms; Act 1 linear ceiling reproduces (top-2 only)"
```

---

### Task 5: Act 2 — the ReLU model on dense data behaves the same

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb`

**Interfaces:**
- Consumes: `ToyModel`, `train`, `importance_weights`, `feature_norms`.
- Produces: nothing new (driver + markdown only).

- [ ] **Step 1: Add the Act 2 opener markdown cell**

```markdown
## Act 2 — Add the ReLU, keep the data dense

Does the ReLU alone create superposition? No. On dense data the ReLU model keeps the same
top-2 features the linear model did. The non-linearity only pays off once the data is
sparse — which is the whole point of Act 3.
```

- [ ] **Step 2: Add the Act 2 driver cell**

```python
relu_dense = ToyModel(5, 2, use_relu=True)
train(relu_dense, sparsity=0.0, importance=imp5, seed=SEED)
norms = feature_norms(relu_dense)
print("ReLU / dense feature norms:", [round(v, 3) for v in norms.tolist()])
n_represented = int((norms > 0.5 * norms.max()).sum())
print("features clearly represented:", n_represented)
assert n_represented <= 2, "ReLU on DENSE data should still keep only its top-2 features"
```

- [ ] **Step 3: Add the interpretation markdown cell**

```markdown
Same story as the linear model: two features in, three discarded. The ReLU changed
nothing while the data is dense. Now we make the data sparse and watch it change
everything.
```

- [ ] **Step 4: Verify headless execute**

Run:
```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_check.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
```
Expected: exits 0 (the `n_represented <= 2` assert holds for the dense ReLU model).

- [ ] **Step 5: Clear outputs and commit**

```bash
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb
git commit -m "superposition: Act 2 — ReLU on dense data still keeps only top-2 (no superposition yet)"
```

---

### Task 6: Act 3 — crank sparsity → the pentagon (viz helpers + reveal)

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb`
- Modify: `tests/superposition/test_toy_models.py`

**Interfaces:**
- Consumes: `ToyModel`, `train`, `importance_weights`, `feature_norms`.
- Produces:
  - `plot_features_2d(W, ax=None, title=None)` — draws each column of a `[2, n]` `W` as a
    ray from the origin.
  - `plot_WtW(W, ax=None, title=None) -> AxesImage` — heatmap of `WᵀW` (`[n, n]`),
    diverging colormap in `[-1, 1]`.
  - `count_represented(model, frac=0.5) -> int` — number of feature columns whose norm
    exceeds `frac` × the max column norm.

- [ ] **Step 1: Add the Act 3 opener markdown cell**

```markdown
## Act 3 — Crank the sparsity: superposition appears

Now we sweep the sparsity `S` upward and retrain the ReLU model each time. As features get
rarer, the model stops throwing three of them away. It packs all five into two dimensions
by placing them in non-orthogonal directions — accepting some interference because sparse
features rarely collide. At `S = 0.9` the five directions form a **pentagon**.
```

- [ ] **Step 2: Add the `plot_features_2d` library cell**

```python
# lib: plot_features_2d
def plot_features_2d(W, ax=None, title=None):
    """Draw each column of a [2, n] weight matrix as a ray from the origin."""
    Wn = W.detach().cpu().numpy()
    ax = ax or plt.gca()
    for i in range(Wn.shape[1]):
        ax.plot([0.0, Wn[0, i]], [0.0, Wn[1, i]], marker="o")
    lim = float(np.abs(Wn).max()) * 1.1 + 1e-9
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, lw=0.5, color="gray"); ax.axvline(0, lw=0.5, color="gray")
    if title:
        ax.set_title(title)
```

- [ ] **Step 3: Add the `plot_WtW` library cell**

```python
# lib: plot_WtW
def plot_WtW(W, ax=None, title=None):
    """Heatmap of WᵀW: diagonal = how strongly each feature is represented,
    off-diagonal = interference between features."""
    WtW = (W.T @ W).detach().cpu().numpy()
    ax = ax or plt.gca()
    im = ax.imshow(WtW, cmap="RdBu", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(WtW.shape[0])); ax.set_yticks(range(WtW.shape[0]))
    if title:
        ax.set_title(title)
    return im
```

- [ ] **Step 4: Add the `count_represented` library cell**

```python
# lib: count_represented
def count_represented(model, frac=0.5):
    """How many feature columns have norm > frac * the largest column norm."""
    norms = feature_norms(model)
    return int((norms > frac * norms.max()).sum())
```

- [ ] **Step 5: Add the Act 3 reveal driver cell**

```python
sparsities = [0.0, 0.8, 0.9]
models = []
fig, axes = plt.subplots(2, len(sparsities), figsize=(4 * len(sparsities), 8))
for j, S in enumerate(sparsities):
    mdl = ToyModel(5, 2, use_relu=True)
    train(mdl, sparsity=S, importance=imp5, seed=SEED)
    models.append(mdl)
    plot_features_2d(mdl.W, ax=axes[0, j], title=f"S = {S}  ({count_represented(mdl)} represented)")
    plot_WtW(mdl.W, ax=axes[1, j], title=f"WᵀW  (S = {S})")
fig.suptitle("Features go from 2 orthogonal → antipodal pairs → a pentagon as sparsity rises")
fig.tight_layout()
plt.show()

reps = [count_represented(m) for m in models]
print("features represented at S =", sparsities, "→", reps)
assert reps[-1] > reps[0], "high sparsity should represent MORE features than dense"
```

- [ ] **Step 6: Add the honest-negative markdown cell**

```markdown
The count of represented features climbs with sparsity — the existence proof. If a given
seed lands the sparse model in a local minimum (the paper notes these toys have
"energy-level" jumps and can get stuck), the exact geometry may not be a clean pentagon;
what matters is the direction of the effect — sparsity buys representation of more
features at the cost of the off-diagonal interference now visible in `WᵀW`.
```

- [ ] **Step 7: Write the failing test (behavioural: sparse represents more than dense)**

Append to `tests/superposition/test_toy_models.py`:

```python
def test_sparse_represents_more_than_dense():
    lib = load_lib()
    torch.manual_seed(0)
    imp = lib["importance_weights"](5)
    dense = lib["ToyModel"](5, 2, use_relu=True)
    sparse = lib["ToyModel"](5, 2, use_relu=True)
    lib["train"](dense, sparsity=0.0, importance=imp, steps=4000, seed=0)
    lib["train"](sparse, sparsity=0.9, importance=imp, steps=4000, seed=0)
    assert lib["count_represented"](sparse) > lib["count_represented"](dense)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run:
```bash
uv run --no-sync pytest tests/superposition/test_toy_models.py -q
```
Expected: 9 passed.

- [ ] **Step 9: Verify headless execute, clear outputs, commit**

```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_check.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb tests/superposition/test_toy_models.py
git commit -m "superposition: Act 3 — the pentagon; sparse ReLU model represents all 5 features in 2 dims"
```

---

### Task 7: Act 4 — the sparsity phase, quantified (n=20, m=5)

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb`
- Modify: `tests/superposition/test_toy_models.py`

**Interfaces:**
- Consumes: `ToyModel`, `train`, `importance_weights`, `count_represented`, `plot_WtW`.
- Produces: `sparsity_sweep(n_features, n_hidden, sparsities, decay=0.7, seed=0) -> (list[int], list[Tensor])`
  returning, per sparsity, the number of represented features and the trained `W` (detached).

- [ ] **Step 1: Add the Act 4 opener markdown cell**

```markdown
## Act 4 — The sparsity phase, quantified

The pentagon is one seed at one size. To show the *effect* rather than an anecdote, we
move to the paper's demonstration size — `n = 20` features in `m = 5` dimensions,
`Iᵢ = 0.7^i` — and sweep sparsity over a grid, counting how many features end up
represented at each point. The count rises with sparsity: that curve is "sparsity drives
superposition" as a number, not a vibe.
```

- [ ] **Step 2: Add the `sparsity_sweep` library cell**

```python
# lib: sparsity_sweep
def sparsity_sweep(n_features, n_hidden, sparsities, decay=0.7, seed=0):
    """Train one ReLU model per sparsity; return (counts_represented, [W per sparsity])."""
    imp = importance_weights(n_features, decay=decay)
    counts, weights = [], []
    for S in sparsities:
        mdl = ToyModel(n_features, n_hidden, use_relu=True)
        train(mdl, sparsity=S, importance=imp, seed=seed)
        counts.append(count_represented(mdl))
        weights.append(mdl.W.detach().clone())
    return counts, weights
```

- [ ] **Step 3: Add the Act 4 driver + plot cell**

```python
sweep_S = [0.0, 0.5, 0.7, 0.9, 0.97, 0.99]
counts, weights = sparsity_sweep(n_features=20, n_hidden=5, sparsities=sweep_S, seed=SEED)

fig, (ax_curve, ax_lo, ax_hi) = plt.subplots(1, 3, figsize=(15, 4))
ax_curve.plot(sweep_S, counts, marker="o")
ax_curve.set_xlabel("sparsity S"); ax_curve.set_ylabel("features represented (of 20)")
ax_curve.set_title("More sparsity → more features in superposition")
plot_WtW(weights[0], ax=ax_lo, title=f"WᵀW at S = {sweep_S[0]} (dense)")
plot_WtW(weights[-1], ax=ax_hi, title=f"WᵀW at S = {sweep_S[-1]} (sparse)")
fig.tight_layout()
plt.show()

print("features represented across S =", sweep_S, "→", counts)
assert counts[-1] > counts[0], "the sparse end must represent more features than the dense end"
```

- [ ] **Step 4: Add the interpretation markdown cell**

```markdown
At the dense end the model uses its five dimensions for five features and `WᵀW` is nearly
diagonal. At the sparse end it represents many more than five, and `WᵀW` fills in with
off-diagonal interference — the signature of superposition. Same mechanism as the
pentagon, now at scale.
```

- [ ] **Step 5: Write the failing test**

Append to `tests/superposition/test_toy_models.py`:

```python
def test_sparsity_sweep_monotone_endpoints():
    lib = load_lib()
    torch.manual_seed(0)
    counts, weights = lib["sparsity_sweep"](
        n_features=20, n_hidden=5, sparsities=[0.0, 0.99], seed=0
    )
    assert len(counts) == 2 and len(weights) == 2
    assert weights[0].shape == (5, 20)
    assert counts[-1] > counts[0]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:
```bash
uv run --no-sync pytest tests/superposition/test_toy_models.py -q
```
Expected: 10 passed.

- [ ] **Step 7: Verify headless execute, clear outputs, commit**

```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_check.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb tests/superposition/test_toy_models.py
git commit -m "superposition: Act 4 — sparsity sweep (n=20, m=5) quantifies the phase; count rises with S"
```

---

### Task 8: Recap + handoff, and final hygiene pass

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb`

**Interfaces:**
- Consumes: nothing (markdown only).
- Produces: nothing.

- [ ] **Step 1: Add the recap + handoff markdown cell**

```markdown
## Recap, and what comes next

**What we proved.** Superposition is real: a ReLU model on sparse data represents more
features than it has dimensions (five in two, the pentagon; and many-in-five at scale).
Sparsity is the knob — dense data gives an orthogonal top-`m` basis (PCA-like), and only
as features get rare does the model superpose them. The cost is interference, visible as
off-diagonal mass in `WᵀW`.

**What we deferred — book two.** *Why* the directions pick a pentagon (and tetrahedrons,
digons, and other uniform polytopes) rather than any old non-orthogonal set; the full
phase diagram; "feature dimensionality" (the fraction of a dimension a feature gets); and
computation performed *in* superposition. Each is its own study, and each builds directly
on the toy we just made.
```

- [ ] **Step 2: Run the full test suite**

Run:
```bash
uv run --no-sync pytest tests/superposition/ -q
```
Expected: 10 passed.

- [ ] **Step 3: Final headless execute of the whole notebook**

Run:
```bash
uv run --no-sync jupyter nbconvert --to notebook --execute \
  --output /tmp/tms_final.ipynb \
  notebooks/superposition/toy_models_of_superposition.ipynb
```
Expected: exits 0; total wall-clock well under 10 minutes.

- [ ] **Step 4: Clear outputs + nbstripout, then commit**

```bash
uv run --no-sync jupyter nbconvert --clear-output --inplace \
  notebooks/superposition/toy_models_of_superposition.ipynb
uv run --no-sync nbstripout notebooks/superposition/toy_models_of_superposition.ipynb
git add notebooks/superposition/toy_models_of_superposition.ipynb
git commit -m "superposition: Act 5 recap + handoff to book two; final hygiene pass"
```

- [ ] **Step 5: (Optional) glossary entry**

If `docs/glossary.md` lacks a "Superposition" entry, add one line under the mech-interp
terms:

```markdown
**Superposition** — a model representing more features than it has dimensions by placing
them in non-orthogonal directions, tolerated because sparse features rarely co-activate.
Demonstrated in `notebooks/superposition/toy_models_of_superposition.ipynb`.
```

Then:
```bash
git add docs/glossary.md
git commit -m "docs: glossary entry for superposition"
```

---

## Self-Review

**Spec coverage:**
- Thesis / stated-at-top → Task 1 Step 1. ✓
- Act 0 puzzle → Task 1. ✓
- Act 1 linear ceiling (n=5,m=2, top-2, PCA-like) → Tasks 2–4. ✓
- Act 2 ReLU+dense same as linear → Task 5. ✓
- Act 3 crank sparsity → pentagon, star plot + WᵀW → Task 6. ✓
- Act 4 quantified sparsity sweep → Task 7. ✓
- Recap + handoff (defers geometry mechanics, phase diagram, feature dimensionality,
  computation-in-superposition) → Task 8. ✓
- Code units `ToyModel` / `make_batch` / `train` / viz helpers → Tasks 2,3,4,6,7. ✓
- Honest-negative rule, no hard gate → loose behavioural asserts + honest-negative
  markdown in Tasks 4,5,6; called out in Global Constraints. ✓
- No Arabic, no real model, no new deps → Global Constraints; stated in notebook Task 1. ✓
- Runtime <10 min on one seed → Task 8 Step 3 checks it. ✓
- Faithful config (n=5/m=2 flagship; n=20/m=5, Iᵢ=0.7^i; S∈{0,0.8,0.9}) → Global
  Constraints + Tasks 6,7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step
shows the exact command and expected result. ✓

**Type consistency:** `make_batch(n_features, sparsity, batch_size, generator)`,
`importance_weights(n_features, decay)`, `ToyModel(n_features, n_hidden, use_relu)` with
`.W [m,n]`/`.b [n]`, `train(model, sparsity, importance, steps, lr, batch_size, seed)`,
`feature_norms(model)`, `count_represented(model, frac)`, `plot_features_2d(W, ax, title)`,
`plot_WtW(W, ax, title)`, `sparsity_sweep(n_features, n_hidden, sparsities, decay, seed)`
— names and signatures are consistent across every task that references them. ✓
