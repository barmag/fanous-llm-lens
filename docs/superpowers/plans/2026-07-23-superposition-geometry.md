# Superposition Geometry (Book Two) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/superposition/superposition_geometry.ipynb` — book two of the superposition series — showing that superposed features arrange into uniform polytopes, measured via feature dimensionality, reached through energy-level jumps, and deformed by non-uniformity.

**Architecture:** A narrative notebook with `# lib:`-marked code cells tested by exec-ing them from the .ipynb (same convention as book one, `tests/superposition/test_toy_models.py`). Acts 0–6 + recap. Heavy runs (Act 3 sweep, Act 5 dynamics) cache `.pt` files under `notebooks/superposition/cache/` (git-ignored). The Act 3 sweep uses the Colab's batched-instances device: all 20 sparsities trained in one loop.

**Tech Stack:** PyTorch (CPU sufficient; ROCm GPU optional), matplotlib, numpy, pytest, nbformat/jupyter nbconvert. Env: `uv run --no-sync` (never bare `uv run` — it drops the ROCm torch).

**Spec:** `docs/superpowers/specs/2026-07-23-superposition-geometry-design.md`

## Global Constraints

- Reference hyperparameters pinned from the paper's Colab (`anthropics/toy-models-of-superposition`, `toy_models.ipynb`, "Feature geometry" section): n_features=200, n_hidden=20, n_instances=20, `1−S = 20^−linspace(0,1,20)`, importance ≡ 1, AdamW lr=1e-3 constant, 10 000 steps, batch 1024.
- Feature dimensionality formula (paper + Colab `compute_dimensionality`): `Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)²` with Ŵᵢ the unit column; `D* = m/‖W‖²_F`.
- Sparsity convention: `S` = P(feature is zero), as in book one. (The Colab uses density `1−S`; convert at the boundary.)
- Notebook markdown is pedagogical only — never cite CLAUDE.md/AGENTS.md conventions inside cells.
- Narrative notebook → ships **executed with outputs** (do NOT clear outputs before commit).
- Never edit the .ipynb while a detached `nbconvert --execute --inplace` on it is running.
- Run `nbconvert --execute` from `notebooks/superposition/` so the relative `cache/` path resolves.
- Honest negatives: save results first, report whatever the metric is; no pass/fail gate in the notebook that could hide a negative. (Asserts belong in pytest, not in notebook analysis cells.)
- Commit messages name the result, not the change.
- All work on branch `superposition-geometry` (already created off up-to-date main).

## File Structure

- `notebooks/superposition/superposition_geometry.ipynb` — the notebook (created Task 1, grown through Task 9)
- `tests/superposition/test_geometry.py` — exec-from-notebook lib-cell tests (created Task 1, grown alongside)
- `notebooks/superposition/toy_models_of_superposition.ipynb` — **Task 0 bug fix only** (split a polluted lib cell)
- `.gitignore` — add `notebooks/superposition/cache/`
- `docs/glossary.md` — add "feature dimensionality" entry (Task 9)

**Notebook editing:** append/insert cells with the NotebookEdit tool, or an `nbformat` script when restructuring. Cell sources are given verbatim in each task. Keep `# lib:` markers exactly as written — the tests find cells by that prefix.

---

### Task 0: Fix book one's broken lib cell (tests are red on main)

Book one's `# lib: per_feature_loss` cell had Act-crossover analysis code appended (references `models`, `imp5` defined in non-lib cells). `load_lib` execs every lib cell in a fresh namespace → `NameError: name 'models' is not defined` → **all 10 tests in `tests/superposition/test_toy_models.py` currently fail**. Fix: split the cell — lib function stays, analysis moves to a new non-lib cell that keeps the outputs (the plot belongs to the analysis half).

**Files:**
- Modify: `notebooks/superposition/toy_models_of_superposition.ipynb` (the `# lib: per_feature_loss` cell)
- Test: `tests/superposition/test_toy_models.py` (existing, unchanged)

**Interfaces:**
- Produces: a green baseline test suite; the lib-cell convention book two copies.

- [ ] **Step 1: Confirm the failure**

Run: `uv run --no-sync pytest tests/superposition/test_toy_models.py -q`
Expected: `10 failed`, errors like `NameError: name 'models' is not defined`

- [ ] **Step 2: Split the cell with an nbformat script**

```python
# scratch script — run with: uv run --no-sync python <script>
import nbformat

p = "notebooks/superposition/toy_models_of_superposition.ipynb"
nb = nbformat.read(p, as_version=4)
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and cell.source.lstrip().startswith("# lib: per_feature_loss"):
        marker = "\n\ndense_tr, sparse_tr"
        assert marker in cell.source, "cell already fixed?"
        lib_src, rest = cell.source.split(marker, 1)
        new = nbformat.v4.new_code_cell(source="dense_tr, sparse_tr" + rest)
        new.outputs = cell.outputs          # the plot belongs to the analysis half
        new.execution_count = cell.execution_count
        cell.source = lib_src.rstrip() + "\n"
        cell.outputs = []
        cell.execution_count = None
        nb.cells.insert(idx + 1, new)
        break
else:
    raise SystemExit("per_feature_loss lib cell not found")
nbformat.write(nb, p)
print("split ok")
```

- [ ] **Step 3: Verify tests pass**

Run: `uv run --no-sync pytest tests/superposition/test_toy_models.py -q`
Expected: `10 passed`

- [ ] **Step 4: Commit**

```bash
git add notebooks/superposition/toy_models_of_superposition.ipynb
git commit -m "superposition: book one lib tests restored 10/10 — crossover analysis split out of the per_feature_loss lib cell"
```

---

### Task 1: Scaffold — notebook shell, test loader, cache dir

**Files:**
- Create: `notebooks/superposition/superposition_geometry.ipynb`
- Create: `tests/superposition/test_geometry.py`
- Modify: `.gitignore` (add cache dir)

**Interfaces:**
- Produces: `load_lib()` in the test file (execs `# lib:` cells from the new notebook); `# lib: imports` cell defining `torch`, `F`, `np`, `plt`, `Path`, `SEED=0`, `CACHE` (a `Path("cache")` that is `mkdir`-ed).

- [ ] **Step 1: Write the failing test**

Create `tests/superposition/test_geometry.py`:

```python
"""Unit tests for the library cells of the Superposition Geometry notebook (book two).

Same convention as test_toy_models.py: exec the `# lib:`-marked code cells straight out
of the notebook into a fresh namespace. No GPU needed.
"""
import json
from pathlib import Path

import torch

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "superposition" / "superposition_geometry.ipynb"
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


def test_lib_cells_exec_cleanly():
    ns = load_lib()
    assert ns["SEED"] == 0
    assert "torch" in ns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: FAIL (`FileNotFoundError` — notebook doesn't exist yet)

- [ ] **Step 3: Create the notebook with title, Act 0, and imports lib cell**

Cell 1 — markdown:

```markdown
# The Geometry of Superposition

*Book two of the superposition series. Book one
([`toy_models_of_superposition.ipynb`](./toy_models_of_superposition.ipynb)) showed that
sparsity turns superposition on. This notebook asks the question book one left open:
when a model superposes features, **what shape do they make — and why?***

Reference: Elhage et al. 2022, [*Toy Models of
Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) — the
"Geometry of Superposition" and "Learning Dynamics" sections. Hyperparameters are pinned
from the paper's public Colab.

**Hypothesis.** Superposition is not amorphous. Features arrange into *uniform
polytopes* — digons, triangles, tetrahedra, pentagons, square antiprisms — measurable as
fractional *feature dimensionality* clinging to ½, ⅔, ¾, ⅖, ⅜. Training reaches these
configurations through discrete *energy-level jumps*, and non-uniformity deforms the
geometry smoothly until it snaps.

*Scope note: like book one, this notebook is pure synthetic toy — no real language
model, and no Arabic; the dialect thread resumes once these tools meet real models.*
```

Cell 2 — markdown (Act 0):

```markdown
## Act 0 — The question book one left open

Book one ended on a pentagon: five features, two hidden dimensions, high sparsity — and
the trained weight columns landed at five *equal* angles, 72° apart. We treated that as
an observation. But nothing in the loss says "be regular." Why not four features crammed
and one straggler? Why the most symmetric arrangement available?

Here is the tease: our model is solving a physics problem. Place charged particles on a
sphere and let them repel — they settle into maximally-symmetric configurations (the
[Thomson problem](https://en.wikipedia.org/wiki/Thomson_problem)). Interference between
features acts like repulsion between charges. By the end of this notebook that analogy
will be quantitative: we will *measure* the fraction of a dimension each feature gets,
watch those fractions cling to a handful of exact values (½, ⅔, ¾, ⅖, ⅜), and identify
the polytope behind each value.

The plan: rebuild the toy (Act 1) → build the measuring instrument and calibrate it on
known solutions (Act 2) → sweep sparsity and find the plateaus (Act 3) → show the
plateaus are polytopes (Act 4) → watch training jump between them (Act 5) → perturb one
feature and watch the geometry stretch and snap (Act 6).
```

Cell 3 — code (`# lib: imports`):

```python
# lib: imports
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
CACHE = Path("cache")  # created by the cells that write to it (keeps pytest's cwd clean)
print("torch", torch.__version__, "| seed", SEED)
```

- [ ] **Step 4: Add cache dir to .gitignore**

Append to `.gitignore`:

```
# Superposition book two caches sweep/dynamics runs here
notebooks/superposition/cache/
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb tests/superposition/test_geometry.py .gitignore
git commit -m "superposition geometry: Act 0 scaffold — the pentagon question, Thomson tease, lib-cell test loader"
```

---

### Task 2: Act 1 — restate the toy, retrain the calibration pentagon

Restate book one's core lib cells (same code, uniform importance for this book), plus the two plot helpers reused later. Then train the n=5, m=2, importance≡1, `1−S=0.05` pentagon — the known-good case Acts 2 and 6 calibrate against.

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (append cells)
- Test: `tests/superposition/test_geometry.py`

**Interfaces:**
- Consumes: `load_lib` from Task 1.
- Produces: `make_batch(n_features, sparsity, batch_size, generator=None) -> [B, n]`; `ToyModel(n_features, n_hidden, use_relu=True)` with `W [m, n]`, `b [n]`; `train(model, sparsity, importance, steps=10_000, lr=1e-3, batch_size=1024, seed=0) -> list[float]`; `plot_features_2d(W, ax=None, title=None)`; `plot_WtW(W, ax=None, title=None)`. Notebook state: `pentagon` (trained ToyModel), `UNIFORM_DENSITY = 0.05`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/superposition/test_geometry.py`:

```python
def test_make_batch_shape_and_sparsity():
    ns = load_lib()
    gen = torch.Generator().manual_seed(0)
    x = ns["make_batch"](10, 0.9, 4096, generator=gen)
    assert x.shape == (4096, 10)
    frac_zero = (x == 0).float().mean().item()
    assert 0.88 < frac_zero < 0.92


def test_toymodel_forward_shape():
    ns = load_lib()
    mdl = ns["ToyModel"](5, 2)
    out = mdl(torch.rand(7, 5))
    assert out.shape == (7, 5)
    assert (out >= 0).all()  # ReLU output


def test_train_reduces_loss():
    ns = load_lib()
    torch.manual_seed(0)
    mdl = ns["ToyModel"](5, 2)
    losses = ns["train"](mdl, sparsity=0.0, importance=torch.ones(5), steps=600)
    assert losses[-1] < losses[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: 3 new FAIL with `KeyError: 'make_batch'` etc.

- [ ] **Step 3: Append Act 1 cells to the notebook**

Markdown cell:

```markdown
## Act 1 — Rebuild the toy

Same model as book one, restated so this notebook stands alone: embed `n` features into
`m < n` dimensions through `W [m, n]`, read back through `Wᵀ`, add a bias, ReLU. Data is
sparse — each feature is 0 with probability `S`, else uniform on [0, 1). One deliberate
change from book one: **importance is uniform** (`Iᵢ = 1`). The geometry section of the
paper studies *uniform* superposition — all features identical — because symmetric
problems have symmetric solutions, and that symmetry is exactly what we want to explain.

Shapes to hold onto: `x [B, n] → h = xWᵀ [B, m] → x' = ReLU(hW + b) [B, n]`.

First, the calibration case: n=5, m=2 at density `1−S = 0.05` (the paper's setting for
this small model). If the pentagon reproduces, we have our known-good solution.
```

Code cell (`# lib: make_batch`):

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

Code cell (`# lib: toymodel`):

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

Code cell (`# lib: train`):

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

Code cell (`# lib: plot_features_2d`):

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

Code cell (`# lib: plot_WtW`):

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

Code cell (train the calibration pentagon — not a lib cell):

```python
# Act 1: the calibration pentagon — n=5, m=2, uniform importance, density 1−S = 0.05
UNIFORM_DENSITY = 0.05

torch.manual_seed(SEED)
pentagon = ToyModel(5, 2)
losses = train(pentagon, sparsity=1 - UNIFORM_DENSITY, importance=torch.ones(5), seed=SEED)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plot_features_2d(pentagon.W, ax=axes[0], title="W columns — the pentagon returns")
plot_WtW(pentagon.W, ax=axes[1], title="WᵀW")
plt.tight_layout(); plt.show()

W_p = pentagon.W.detach()
angles = torch.atan2(W_p[1], W_p[0]).rad2deg().sort().values
gaps = torch.diff(torch.cat([angles, angles[:1] + 360]))
print("column norms:", [f"{v:.3f}" for v in W_p.norm(dim=0)])
print("angular gaps (deg):", [f"{v:.1f}" for v in gaps], "| regular pentagon = 72.0 each")
```

Markdown cell after it:

```markdown
Whatever the run produced is the result: if the gaps sit near 72° with roughly equal
norms, we have the regular pentagon and a calibration target. If the run landed in a
different configuration (these toys have local minima — Act 5 is about exactly that),
we note what formed and continue; the instrument in Act 2 works either way.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb tests/superposition/test_geometry.py
git commit -m "superposition geometry: Act 1 — toy restated with uniform importance; calibration pentagon trains at 1−S=0.05"
```

---

### Task 3: Act 2 — the instrument: feature dimensionality

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (append cells)
- Test: `tests/superposition/test_geometry.py`

**Interfaces:**
- Consumes: `ToyModel`, `train`, `pentagon` notebook state from Task 2.
- Produces: `feature_dimensionality(W, eps=1e-6) -> [n] tensor`; `frobenius_dims_per_feature(W) -> float`. Both take `W [m, n]` (columns = features).

- [ ] **Step 1: Write the failing tests**

Append to `tests/superposition/test_geometry.py`:

```python
def _pentagon_W():
    ang = 2 * torch.pi * torch.arange(5) / 5
    return torch.stack([torch.cos(ang), torch.sin(ang)])  # [2, 5], unit columns


def test_dimensionality_identity_is_one():
    ns = load_lib()
    D = ns["feature_dimensionality"](torch.eye(4))
    assert torch.allclose(D, torch.ones(4), atol=1e-5)


def test_dimensionality_antipodal_is_half():
    ns = load_lib()
    W = torch.tensor([[1.0, -1.0]])  # one dim, two antipodal features
    D = ns["feature_dimensionality"](W)
    assert torch.allclose(D, torch.full((2,), 0.5), atol=1e-5)


def test_dimensionality_pentagon_is_two_fifths():
    ns = load_lib()
    D = ns["feature_dimensionality"](_pentagon_W())
    assert torch.allclose(D, torch.full((5,), 0.4), atol=1e-4)


def test_dimensionality_zero_column_is_zero():
    ns = load_lib()
    W = torch.eye(3)
    W[:, 2] = 0.0
    D = ns["feature_dimensionality"](W)
    assert D[2].item() == 0.0
    assert torch.allclose(D[:2], torch.ones(2), atol=1e-5)


def test_frobenius_dims_per_feature():
    ns = load_lib()
    assert abs(ns["frobenius_dims_per_feature"](torch.eye(4)) - 1.0) < 1e-6
    # pentagon: 5 unit-norm features in 2 dims -> 2/5 dims per feature
    assert abs(ns["frobenius_dims_per_feature"](_pentagon_W()) - 0.4) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: 5 new FAIL with `KeyError: 'feature_dimensionality'`

- [ ] **Step 3: Append Act 2 cells**

Markdown cell:

```markdown
## Act 2 — The instrument: feature dimensionality

The paper's measuring device. Define the **dimensionality of feature i** as

$$D_i = \frac{\lVert W_i \rVert^2}{\sum_j (\hat W_i \cdot W_j)^2}$$

Numerator: how strongly feature *i* is represented. Denominator: how many features share
the direction it lives in (each `Wⱼ` projected onto `Ŵᵢ`, squared, summed — the `j = i`
term contributes `‖Wᵢ‖²`, so a feature always shares with at least itself). Read it as
"the fraction of a dimension feature *i* gets to keep."

Predictions before we measure — the name-then-experiment ritual:

| configuration | prediction | arithmetic |
|---|---|---|
| dedicated orthogonal direction | 1 | ‖Wᵢ‖²/‖Wᵢ‖² |
| antipodal pair (digon) | ½ | 1/(1+1) |
| regular pentagon vertex | ⅖ | 1/(1 + 2cos²72° + 2cos²144°) = 1/2.5 |
| dropped feature | 0 | 0/anything |

A companion summary statistic: `D* = m/‖W‖²_F`, "dimensions per feature." Since
represented features have `‖Wᵢ‖ ≈ 1` and dropped ones `≈ 0`, `‖W‖²_F` counts learned
features, and `D*` is the budget each one gets on average.

The instrument earns trust only by reading the known cases correctly — hand-built exact
configurations first, then the pentagon we actually trained in Act 1.
```

Code cell (`# lib: feature_dimensionality`):

```python
# lib: feature_dimensionality
def feature_dimensionality(W, eps=1e-6):
    """Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)², W [m, n] (paper's compute_dimensionality).

    Features with ~zero norm are defined as D = 0 (unlearned) rather than 0/0.
    """
    W = W.detach()
    norms = W.norm(dim=0)                            # [n]
    W_unit = W / norms.clamp(min=eps)                # [m, n]
    interference = ((W_unit.T @ W) ** 2).sum(dim=1)  # [n]
    D = norms ** 2 / interference.clamp(min=eps)
    return torch.where(norms > eps, D, torch.zeros_like(D))
```

Code cell (`# lib: frobenius_dims_per_feature`):

```python
# lib: frobenius_dims_per_feature
def frobenius_dims_per_feature(W):
    """D* = m / ‖W‖²_F — average dimensions per learned feature."""
    W = W.detach()
    return W.shape[0] / (W.norm() ** 2).item()
```

Code cell (calibration — not lib):

```python
# Act 2: calibrate the instrument on known cases
exact_pentagon = torch.stack([
    torch.cos(2 * torch.pi * torch.arange(5) / 5),
    torch.sin(2 * torch.pi * torch.arange(5) / 5),
])
cases = {
    "identity (4 dedicated dims)": torch.eye(4),
    "antipodal pair": torch.tensor([[1.0, -1.0]]),
    "exact pentagon": exact_pentagon,
    "trained pentagon (Act 1)": pentagon.W,
}
for name, W in cases.items():
    D = feature_dimensionality(W)
    print(f"{name:32s} D = {[f'{d:.3f}' for d in D]}  D* = {frobenius_dims_per_feature(W):.3f}")
```

Markdown cell after it:

```markdown
The exact cases must land on 1, ½, ⅖ by arithmetic; the *trained* pentagon is the real
test — its Dᵢ should sit near 0.4 without ever being told about pentagons. The
instrument now scales to any `W`, which is what Act 3 needs: 200 features, 20 dims, 20
sparsities at once.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb tests/superposition/test_geometry.py
git commit -m "superposition geometry: Act 2 — dimensionality instrument reads 1, 1/2, 2/5 on known cases; trained pentagon measures ~0.4"
```

---

### Task 4: The batched-instances device (Colab-faithful)

The Colab trains all 20 sparsity settings *simultaneously* as instances of one batched
tensor — the sweep becomes a single 10k-step loop. Build it as lib cells with tests
before Act 3 uses it.

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (append cells)
- Test: `tests/superposition/test_geometry.py`

**Interfaces:**
- Consumes: nothing new (parallel to `ToyModel`).
- Produces: `BatchedToyModel(n_instances, n_features, n_hidden)` with `W [I, m, n]`, `b [I, n]`; `make_batch_batched(n_instances, n_features, sparsities, batch_size, generator=None) -> [B, I, n]` (`sparsities` is a `[I]` tensor of S values); `train_batched(model, sparsities, steps=10_000, lr=1e-3, batch_size=1024, seed=0, snapshot_every=None) -> dict` with keys `losses` (list of `(step, float)`), `snap_steps` (list[int]), `snapshots` (list of `W` clones `[I, m, n]`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/superposition/test_geometry.py`:

```python
def test_batched_model_shapes():
    ns = load_lib()
    mdl = ns["BatchedToyModel"](3, 10, 4)
    x = torch.rand(8, 3, 10)
    out = mdl(x)
    assert out.shape == (8, 3, 10)
    assert (out >= 0).all()


def test_batched_model_matches_single_instance():
    """One instance of the batched model computes the same function as ToyModel."""
    ns = load_lib()
    single = ns["ToyModel"](6, 3)
    batched = ns["BatchedToyModel"](1, 6, 3)
    with torch.no_grad():
        batched.W.copy_(single.W.unsqueeze(0))
        batched.b.copy_(single.b.unsqueeze(0))
    x = torch.rand(5, 6)
    assert torch.allclose(single(x), batched(x.unsqueeze(1))[:, 0], atol=1e-6)


def test_make_batch_batched_per_instance_sparsity():
    ns = load_lib()
    gen = torch.Generator().manual_seed(0)
    S = torch.tensor([0.0, 0.9])
    x = ns["make_batch_batched"](2, 50, S, 2048, generator=gen)
    assert x.shape == (2048, 2, 50)
    zero_frac = (x == 0).float().mean(dim=(0, 2))
    assert zero_frac[0].item() < 0.02
    assert 0.88 < zero_frac[1].item() < 0.92


def test_train_batched_reduces_loss_and_snapshots():
    ns = load_lib()
    torch.manual_seed(0)
    mdl = ns["BatchedToyModel"](2, 8, 3)
    log = ns["train_batched"](mdl, torch.tensor([0.5, 0.9]), steps=300, snapshot_every=100)
    assert log["losses"][-1][1] < log["losses"][0][1]
    assert log["snap_steps"][0] == 0 and log["snap_steps"][-1] == 299
    assert log["snapshots"][0].shape == (2, 3, 8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: 4 new FAIL with `KeyError: 'BatchedToyModel'`

- [ ] **Step 3: Append the interlude + lib cells**

Markdown cell:

```markdown
### Interlude — twenty models for the price of one

Act 3 needs one trained model per sparsity value. Trained one at a time that is 20 runs;
the paper's Colab instead gives `W` a leading *instance* dimension — `W [I, m, n]` — and
trains all twenty models in a single loop, each instance seeing its own sparsity. Same
model, vectorized; einsum keeps each instance's forward pass separate. (Training
follows the Colab exactly: AdamW, constant lr 1e-3, uniform importance, 10k steps,
batch 1024 — and the loss is *averaged* over features per instance, then summed over
instances, so instances don't trade off against each other.)
```

Code cell (`# lib: batched_toymodel`):

```python
# lib: batched_toymodel
class BatchedToyModel(torch.nn.Module):
    """I independent ToyModels trained at once: W [I, m, n], b [I, n].

    forward: x [B, I, n] → h [B, I, m] → ReLU(out) [B, I, n]. Instance i never mixes
    with instance j — the einsums contract only within an instance.
    """
    def __init__(self, n_instances, n_features, n_hidden):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(n_instances, n_hidden, n_features))
        for i in range(n_instances):
            torch.nn.init.xavier_normal_(self.W.data[i])
        self.b = torch.nn.Parameter(torch.zeros(n_instances, n_features))

    def forward(self, x):
        h = torch.einsum("bif,imf->bim", x, self.W)
        out = torch.einsum("bim,imf->bif", h, self.W) + self.b
        return F.relu(out)
```

Code cell (`# lib: make_batch_batched`):

```python
# lib: make_batch_batched
def make_batch_batched(n_instances, n_features, sparsities, batch_size, generator=None):
    """Per-instance sparse features: sparsities [I] of S values → batch [B, I, n]."""
    vals = torch.rand(batch_size, n_instances, n_features, generator=generator)
    keep = (
        torch.rand(batch_size, n_instances, n_features, generator=generator)
        >= sparsities[None, :, None]
    )
    return vals * keep
```

Code cell (`# lib: train_batched`):

```python
# lib: train_batched
def train_batched(model, sparsities, steps=10_000, lr=1e-3, batch_size=1024, seed=0,
                  snapshot_every=None):
    """Colab-faithful training: AdamW, constant lr, importance ≡ 1.

    Loss = per-instance mean over batch and features, summed over instances.
    Optionally clones W every `snapshot_every` steps (plus the last step).
    Returns dict(losses=[(step, loss)], snap_steps=[...], snapshots=[W clones]).
    """
    n_instances, _, n_features = model.W.shape
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log = {"losses": [], "snap_steps": [], "snapshots": []}
    for step in range(steps):
        x = make_batch_batched(n_instances, n_features, sparsities, batch_size, generator=gen)
        out = model(x)
        loss = ((x - out) ** 2).mean(dim=(0, 2)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log["losses"].append((step, loss.item()))
        if snapshot_every and (step % snapshot_every == 0 or step == steps - 1):
            log["snap_steps"].append(step)
            log["snapshots"].append(model.W.detach().clone())
    return log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb tests/superposition/test_geometry.py
git commit -m "superposition geometry: batched-instances device — one loop trains 20 sparsities, single-instance parity verified"
```

---

### Task 5: Act 3 — the uniform sweep and the plateau plot

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (append cells)
- Test: `tests/superposition/test_geometry.py` (plot helper smoke via `load_lib` only — no new test; the sweep itself is a notebook result, not a unit)

**Interfaces:**
- Consumes: `BatchedToyModel`, `train_batched`, `feature_dimensionality`, `frobenius_dims_per_feature`.
- Produces: `plot_dimensionality_sweep(sparsities, W, ax=None)` lib cell (`W [I, m, n]`); notebook state: `sweep_S` (`[20]` tensor), `sweep_W` (`[20, 20, 200]` tensor) — reused by Act 4.

- [ ] **Step 1: Append Act 3 markdown**

```markdown
## Act 3 — The uniform sweep

The paper's setup, from its Colab: **n = 200 features, m = 20 dims, 20 sparsity
instances**, density `1−S` log-spaced from 1 (fully dense) down to 1/20, importance ≡ 1.
Every feature is identical — same importance, same sparsity — so any structure in the
solution is structure the *loss landscape* chose, not structure we baked in.

Two readouts per instance: `D* = m/‖W‖²_F` (average dims per learned feature, one point
per instance) and the full per-feature scatter of `Dᵢ` (200 points per instance). If
superposition were amorphous, the scatter would smear. The paper's claim: it clings to
a handful of exact fractions.
```

- [ ] **Step 2: Append the smoke-test cell**

```python
# Act 3a: smoke test — time 100 steps before committing to 10k
import time

sweep_S = 1.0 - 20.0 ** -torch.linspace(0, 1, 20)   # density 1 → 1/20, log-spaced

torch.manual_seed(SEED)
_probe = BatchedToyModel(20, 200, 20)
t0 = time.time()
train_batched(_probe, sweep_S, steps=100, seed=SEED)
per_step = (time.time() - t0) / 100
print(f"{per_step*1000:.0f} ms/step → est. full run {per_step * 10_000 / 60:.1f} min")
```

- [ ] **Step 3: Append the cached sweep cell**

```python
# Act 3b: the sweep — cached; delete cache/uniform_sweep.pt to retrain
CACHE.mkdir(exist_ok=True)
sweep_path = CACHE / "uniform_sweep.pt"
if sweep_path.exists():
    blob = torch.load(sweep_path)
    sweep_W, sweep_losses = blob["W"], blob["losses"]
    print("loaded cache:", sweep_path)
else:
    torch.manual_seed(SEED)
    sweep_model = BatchedToyModel(20, 200, 20)
    log = train_batched(sweep_model, sweep_S, steps=10_000, seed=SEED)
    sweep_W = sweep_model.W.detach().clone()
    sweep_losses = log["losses"]
    torch.save({"W": sweep_W, "losses": sweep_losses, "S": sweep_S, "seed": SEED}, sweep_path)
    print("trained and cached:", sweep_path)
print("final summed loss:", f"{sweep_losses[-1][1]:.4f}")
```

- [ ] **Step 4: Append the plot helper lib cell + payoff plot**

Code cell (`# lib: plot_dimensionality_sweep`):

```python
# lib: plot_dimensionality_sweep
PLATEAUS = [
    (1.0, "1 — dedicated"),
    (3 / 4, "¾ — tetrahedron"),
    (2 / 3, "⅔ — triangle"),
    (1 / 2, "½ — digon"),
    (2 / 5, "⅖ — pentagon"),
    (3 / 8, "⅜ — square antiprism"),
]

def plot_dimensionality_sweep(sparsities, W, ax=None):
    """Per-feature Dᵢ scatter + D* line vs 1/(1−S), log-x, plateau guides. W [I, m, n]."""
    ax = ax or plt.gca()
    x = 1.0 / (1.0 - sparsities.cpu().numpy())
    for frac, name in PLATEAUS:
        ax.axhline(frac, lw=0.5, ls="--", color="gray")
        ax.annotate(name, (x[-1] * 1.08, frac), fontsize=7, va="center",
                    annotation_clip=False)
    dstars = []
    for i in range(W.shape[0]):
        D = feature_dimensionality(W[i]).numpy()
        jitter = x[i] * (1 + np.random.default_rng(i).uniform(-0.03, 0.03, len(D)))
        ax.scatter(jitter, D, s=3, alpha=0.35, color="tab:blue", linewidths=0)
        dstars.append(frobenius_dims_per_feature(W[i]))
    ax.plot(x, dstars, color="black", lw=1.5, marker="o", ms=3, label="D* = m/‖W‖²_F")
    ax.set_xscale("log")
    ax.set_xlabel("1/(1−S)  (sparser →)")
    ax.set_ylabel("feature dimensionality Dᵢ")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right")
```

Code cell (payoff plot):

```python
# Act 3c: the payoff plot
fig, ax = plt.subplots(figsize=(9, 5.5))
plot_dimensionality_sweep(sweep_S, sweep_W, ax=ax)
ax.set_title("Dimensionality clings to fractions — superposition is quantized")
plt.tight_layout(); plt.show()

# one-line numeric claim: fraction of learned features within 0.02 of a plateau value
D_all = torch.cat([feature_dimensionality(sweep_W[i]) for i in range(sweep_W.shape[0])])
learned = D_all[D_all > 0.05]
plateau_vals = torch.tensor([f for f, _ in PLATEAUS])
dist = (learned[:, None] - plateau_vals).abs().min(dim=1).values  # nearest plateau only
near = int((dist < 0.02).sum())
print(f"{near}/{len(learned)} learned features ({near/len(learned):.0%}) sit within "
      f"0.02 of a named plateau (1, ¾, ⅔, ½, ⅖, ⅜)")
```

Markdown cell after it:

```markdown
Whatever fraction printed is the result. The paper's version of this plot shows dense
bands exactly at ½ and ⅖ with sparser bands at ¾, ⅔, ⅜; stragglers *between* plateaus
are real too — the paper attributes them to imperfect convergence and non-uniform
polytopes, and Act 5 will show solutions migrating between levels mid-training. If a
named plateau (⅜ especially) is missing at this seed and scale, that is a finding to
report, not to hide.
```

- [ ] **Step 5: Run the whole test suite (lib cells must still exec cleanly)**

Run: `uv run --no-sync pytest tests/superposition/ -q`
Expected: `23 passed` (13 geometry + 10 book one)

- [ ] **Step 6: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb
git commit -m "superposition geometry: Act 3 — uniform sweep n=200 m=20; per-feature dimensionality scatter with plateau guides, cached"
```

---

### Task 6: Act 4 — the plateaus are polytopes

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (append cells)
- Test: `tests/superposition/test_geometry.py`

**Interfaces:**
- Consumes: `sweep_S`, `sweep_W` notebook state from Task 5; `feature_dimensionality`.
- Produces: `interference_components(W, norm_cutoff=0.5, cos_threshold=0.15) -> list[list[int]]`; `project_component(W, component, d=2) -> [d, k] tensor`; `plot_component_2d(W, component, ax=None, title=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/superposition/test_geometry.py`:

```python
def test_interference_components_block_structure():
    ns = load_lib()
    # two orthogonal digons + one dropped feature in 4 dims
    W = torch.zeros(4, 5)
    W[0, 0], W[0, 1] = 1.0, -1.0   # digon A in dim 0
    W[1, 2], W[1, 3] = 1.0, -1.0   # digon B in dim 1
    comps = ns["interference_components"](W)
    assert sorted(map(sorted, comps)) == [[0, 1], [2, 3]]  # feature 4 dropped


def test_project_component_recovers_pentagon_angles():
    ns = load_lib()
    ang = 2 * torch.pi * torch.arange(5) / 5
    P = torch.stack([torch.cos(ang), torch.sin(ang)])          # [2, 5]
    Q, _ = torch.linalg.qr(torch.randn(7, 7))
    W = Q[:, :2] @ P                                           # pentagon hidden in 7 dims
    coords = ns["project_component"](W, [0, 1, 2, 3, 4], d=2)  # [2, 5]
    C = coords.T @ coords                                      # Gram matrix
    cosines = (C / C.diag().sqrt().outer(C.diag().sqrt())).flatten()
    expected = {round(v, 3) for v in
                [1.0, float(np.cos(2 * np.pi / 5)), float(np.cos(4 * np.pi / 5))]}
    got = {round(v, 3) for v in cosines.tolist()}
    assert got == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: 2 new FAIL with `KeyError: 'interference_components'`

- [ ] **Step 3: Append Act 4 markdown + lib cells**

Markdown cell:

```markdown
## Act 4 — The plateaus are polytopes

A plateau at p/q should mean: *q features sharing p dimensions*. To check, split each
model's features into groups that interfere with each other but not with anyone else —
connected components of the interference graph (nodes: features with non-trivial norm;
edges: |cos(Wᵢ, Wⱼ)| above a small threshold). The paper calls these *tegum factors*:
the model tiles its 20 dimensions with small independent polytopes in orthogonal
subspaces.

Each component spans its own low-dimensional subspace, so we can PCA it down and simply
*look*. Predictions, one per plateau: ½ → digon (2 features, 1 dim), ⅔ → triangle
(3-in-2), ¾ → tetrahedron (4-in-3), ⅖ → pentagon (5-in-2), ⅜ → square antiprism
(8-in-3). And the Gram matrix gives the angles numerically — a regular tetrahedron's
off-diagonal cosines are exactly −⅓.
```

Code cell (`# lib: interference_components`):

```python
# lib: interference_components
def interference_components(W, norm_cutoff=0.5, cos_threshold=0.15):
    """Connected components of the interference graph over represented features.

    Nodes: features with column norm > norm_cutoff. Edges: |cos(Wᵢ, Wⱼ)| > cos_threshold.
    Returns components as sorted index lists, largest first. W [m, n].
    """
    W = W.detach()
    norms = W.norm(dim=0)
    nodes = [i for i in range(W.shape[1]) if norms[i] > norm_cutoff]
    if not nodes:
        return []
    Wu = W[:, nodes] / norms[nodes]
    adj = (Wu.T @ Wu).abs() > cos_threshold
    comps, seen = [], set()
    for s in range(len(nodes)):
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(nodes[u])
            stack.extend(v for v in range(len(nodes)) if adj[u, v] and v not in seen)
        comps.append(sorted(comp))
    return sorted(comps, key=len, reverse=True)
```

Code cell (`# lib: project_component`):

```python
# lib: project_component
def project_component(W, component, d=2):
    """PCA a component's feature columns into their own top-d subspace. Returns [d, k]."""
    Wc = W.detach()[:, component]                    # [m, k]
    U, _, _ = torch.linalg.svd(Wc, full_matrices=False)
    return U[:, :d].T @ Wc
```

Code cell (`# lib: plot_component_2d`):

```python
# lib: plot_component_2d
def plot_component_2d(W, component, ax=None, title=None):
    """Star plot of a component's features in their own 2-D PCA plane."""
    coords = project_component(W, component, d=2).cpu().numpy()
    ax = ax or plt.gca()
    for i in range(coords.shape[1]):
        ax.plot([0, coords[0, i]], [0, coords[1, i]], marker="o")
    lim = float(np.abs(coords).max()) * 1.2 + 1e-9
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
```

Code cell (the gallery — not lib):

```python
# Act 4a: census of component sizes across the sweep, then a gallery
from collections import Counter

census = {}
for i in range(sweep_W.shape[0]):
    sizes = Counter(len(c) for c in interference_components(sweep_W[i]))
    census[f"1/(1-S)={1/(1-sweep_S[i]):.1f}"] = dict(sorted(sizes.items()))
for k, v in census.items():
    print(f"{k:16s} component sizes → {v}")
```

```python
# Act 4b: gallery — for each named plateau, find a component whose mean Dᵢ is nearest
targets = [(1 / 2, "digon"), (2 / 3, "triangle"), (3 / 4, "tetrahedron"),
           (2 / 5, "pentagon"), (3 / 8, "square antiprism")]
fig, axes = plt.subplots(1, len(targets), figsize=(3 * len(targets), 3.2))
found = {}
for ax, (frac, name) in zip(axes, targets):
    best = None  # (|meanD − frac|, instance, component)
    for i in range(sweep_W.shape[0]):
        D = feature_dimensionality(sweep_W[i])
        for comp in interference_components(sweep_W[i]):
            gap = abs(D[comp].mean().item() - frac)
            if best is None or gap < best[0]:
                best = (gap, i, comp)
    gap, i, comp = best
    found[name] = (i, comp, gap)
    mean_D = feature_dimensionality(sweep_W[i])[comp].mean().item()
    plot_component_2d(sweep_W[i], comp, ax=ax,
                      title=f"{name}\nk={len(comp)}, mean D = {mean_D:.3f}")
fig.suptitle("Components nearest each plateau, PCA-projected to their own plane")
plt.tight_layout(); plt.show()
```

```python
# Act 4c: the angles, numerically — Gram matrix of the tetrahedron-candidate component
i, comp, gap = found["tetrahedron"][0], found["tetrahedron"][1], found["tetrahedron"][2]
Wc = sweep_W[i][:, comp]
Wu = Wc / Wc.norm(dim=0)
G = (Wu.T @ Wu)
print(f"tetrahedron candidate: instance {i}, features {comp}, mean D gap {gap:.3f}")
print("pairwise cosines (regular tetrahedron → −0.333):")
print(np.array2string(G.numpy(), precision=3, suppress_small=True))
```

Markdown cell after it:

```markdown
Read the gallery honestly. A 5-vertex component whose PCA plane shows five evenly-spaced
rays *is* a pentagon; an 8-vertex component at D≈⅜ lives in 3 dims, so its 2-D shadow
looks like two nested squares — that shadow plus the size-8/D=⅜ signature is the
antiprism evidence, and if no such component formed at this seed, the gallery shows
whatever did. Note what the paper predicts about tegum splits: instead of e.g. a
triangular bipyramid at ⅗, models prefer co-occurring triangles (⅔) and digons (½) —
which is why the census skews to small components.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb tests/superposition/test_geometry.py
git commit -m "superposition geometry: Act 4 — interference components identify the polytopes behind the plateaus, with Gram-matrix angles"
```

---

### Task 7: Act 5 — energy-level jumps

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (append cells)
- Test: `tests/superposition/test_geometry.py` (no new units — reuses `train_batched(snapshot_every=…)` already tested)

**Interfaces:**
- Consumes: `BatchedToyModel`, `train_batched` (with `snapshot_every`), `feature_dimensionality`.
- Produces: `plot_dynamics(snap_steps, dim_traj, loss_steps, losses)` lib cell; notebook state: `dyn_log`.

- [ ] **Step 1: Append Act 5 markdown**

```markdown
## Act 5 — Energy-level jumps

Book one apologized for these toys' local minima. The paper's learning-dynamics section
reframes them: training doesn't slide smoothly to a solution, it *jumps between
geometries* — a feature pair collapses into a digon, a triangle absorbs a stray feature
— and each reorganization shows up as a sudden drop in the loss. The paper's evocative
name: energy-level jumps, features hopping between the discrete dimensionalities of
Act 3.

Setup, following the paper: one instance, many identical features, sparsity in the digon
regime (we use n=100, m=10 at 1−S = 0.15 — near the ½ plateau of Act 3's sweep, scaled
down so snapshots stay light). Snapshot `W` every 50 steps, then plot every feature's
Dᵢ trajectory above the loss curve and look for coincident jumps.
```

- [ ] **Step 2: Append the cached dynamics run**

```python
# Act 5a: dynamics run — cached; delete cache/dynamics.pt to retrain
CACHE.mkdir(exist_ok=True)
dyn_path = CACHE / "dynamics.pt"
if dyn_path.exists():
    dyn_log = torch.load(dyn_path)
    print("loaded cache:", dyn_path)
else:
    torch.manual_seed(SEED)
    dyn_model = BatchedToyModel(1, 100, 10)
    dyn_log = train_batched(dyn_model, torch.tensor([0.85]), steps=10_000,
                            seed=SEED, snapshot_every=50)
    torch.save(dyn_log, dyn_path)
    print("trained and cached:", dyn_path)
dim_traj = torch.stack([feature_dimensionality(W[0]) for W in dyn_log["snapshots"]])
print("dim_traj:", tuple(dim_traj.shape), "(snapshots × features)")
```

- [ ] **Step 3: Append the twin-plot lib cell + plot**

Code cell (`# lib: plot_dynamics`):

```python
# lib: plot_dynamics
def plot_dynamics(snap_steps, dim_traj, loss_steps, losses):
    """Per-feature dimensionality trajectories over training, above the loss curve."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   height_ratios=[2, 1])
    for f in range(dim_traj.shape[1]):
        ax1.plot(snap_steps, dim_traj[:, f], lw=0.7, alpha=0.5)
    for frac, name in PLATEAUS:
        ax1.axhline(frac, lw=0.5, ls="--", color="gray")
    ax1.set_ylabel("feature dimensionality Dᵢ")
    ax1.set_ylim(-0.05, 1.1)
    ax2.plot(loss_steps, losses, color="black", lw=1)
    ax2.set_yscale("log")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("loss")
    return fig
```

Code cell:

```python
# Act 5b: the twin plot + a numeric claim about coincidence
loss_steps = [s for s, _ in dyn_log["losses"]]
loss_vals = [v for _, v in dyn_log["losses"]]
fig = plot_dynamics(dyn_log["snap_steps"], dim_traj, loss_steps, loss_vals)
fig.suptitle("Features jump between dimensionality levels; the loss drops when they do")
plt.tight_layout(); plt.show()

# largest single-interval dimensionality reorganization vs the loss change there
moves = (dim_traj[1:] - dim_traj[:-1]).abs().sum(dim=1)
k = int(moves.argmax())
s0, s1 = dyn_log["snap_steps"][k], dyn_log["snap_steps"][k + 1]
before = [v for s, v in dyn_log["losses"] if s <= s0][-1]
after = [v for s, v in dyn_log["losses"] if s >= s1][0]
print(f"largest reorganization: steps {s0}→{s1}, total |ΔD| = {moves[k]:.2f}; "
      f"loss {before:.4f} → {after:.4f} across it")
```

Markdown cell after it:

```markdown
The claim to check by eye: vertical reshuffles in the top panel line up with cliffs in
the bottom one. If this seed produced a smooth loss curve with no visible jump, that is
the reported result (the paper notes jumps are clearest with many features at moderate
sparsity; a different seed can be tried and *both* reported). Either way, Act 3's
stragglers now have an explanation: a model photographed mid-jump.
```

- [ ] **Step 4: Run the test suite**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: `15 passed` (lib cells still exec cleanly)

- [ ] **Step 5: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb
git commit -m "superposition geometry: Act 5 — dimensionality trajectories tracked over training, jumps aligned with loss drops"
```

---

### Task 8: Act 6 — non-uniform superposition: stretch and snap

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (append cells)
- Test: `tests/superposition/test_geometry.py` (no new units — reuses tested pieces)

**Interfaces:**
- Consumes: `ToyModel`, `train`, `plot_features_2d`, `feature_dimensionality`, `UNIFORM_DENSITY`.
- Produces: final experimental act; no new lib cells.

- [ ] **Step 1: Append Act 6 markdown**

```markdown
## Act 6 — Non-uniform superposition: stretch and snap

Real features are never uniform. The paper's entry point: take the n=5, m=2 pentagon
(uniform density 0.05) and vary *one* feature's sparsity, leaving the rest untouched.
Prediction from the repulsion picture: make feature 0 **denser** and it interferes more
often, so the others give it room — the pentagon stretches away from it. Make it
**sparser** and it takes less room — the others close in. Push far enough and the
configuration should stop deforming and *snap* to a different geometry (the paper
observes pentagon → digon-plus-pairs transitions).

One training run per density value, everything else fixed (importance ≡ 1, seed fixed).
```

- [ ] **Step 2: Append the perturbation sweep + strip plot**

```python
# Act 6a: vary feature 0's density around the uniform 0.05
densities = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
perturbed = {}
for d0 in densities:
    torch.manual_seed(SEED)
    mdl = ToyModel(5, 2)
    # per-feature sparsity: feature 0 gets 1−d0, the rest keep 1−UNIFORM_DENSITY.
    # make_batch takes a scalar S, so train with a per-feature keep-mask via a wrapper:
    S_vec = torch.full((5,), 1 - UNIFORM_DENSITY)
    S_vec[0] = 1 - d0
    gen = torch.Generator().manual_seed(SEED)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    for step in range(10_000):
        vals = torch.rand(1024, 5, generator=gen)
        keep = torch.rand(1024, 5, generator=gen) >= S_vec
        x = vals * keep
        out = mdl(x)
        loss = ((x - out) ** 2).sum(dim=-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    perturbed[d0] = mdl.W.detach().clone()

fig, axes = plt.subplots(1, len(densities), figsize=(2.6 * len(densities), 2.8))
for ax, d0 in zip(axes, densities):
    plot_features_2d(perturbed[d0], ax=ax, title=f"1−S₀ = {d0}")
    # feature 0 highlighted: redraw it thick and black
    w0 = perturbed[d0][:, 0].numpy()
    ax.plot([0, float(w0[0])], [0, float(w0[1])], color="black", lw=2.5, marker="o")
fig.suptitle("One feature's density varied (black ray); uniform pentagon at 0.05")
plt.tight_layout(); plt.show()
```

```python
# Act 6b: quantify — feature 0's norm, its D₀, and the min angular gap to a neighbor
print(f"{'1−S₀':>6s} {'‖W₀‖':>7s} {'D₀':>7s} {'nearest-neighbor angle (deg)':>30s}")
for d0 in densities:
    W = perturbed[d0]
    w0 = W[:, 0]
    D0 = feature_dimensionality(W)[0].item()
    angles = []
    for j in range(1, 5):
        wj = W[:, j]
        if wj.norm() > 0.3:
            cosang = (w0 @ wj) / (w0.norm() * wj.norm() + 1e-9)
            angles.append(float(torch.rad2deg(torch.arccos(cosang.clamp(-1, 1)))))
    nn = min(angles) if angles and w0.norm() > 0.3 else float("nan")
    print(f"{d0:6.2f} {w0.norm():7.3f} {D0:7.3f} {nn:30.1f}")
```

Markdown cell after it:

```markdown
Reading the strip: at 0.05 (center) the uniform pentagon; toward the dense end the
black ray's neighbors should retreat (nearest-neighbor angle grows); toward the sparse
end they close in, until at some value the five-fold structure gives way entirely —
feature 0's D₀ leaving the ⅖ level marks the snap. Whichever configuration it snaps
*to* at this seed is reported as found; the paper's uniform-geometry theory survives
non-uniformity as *deformation between snaps*, which is what makes it relevant to real,
messy models.
```

- [ ] **Step 3: Run the test suite**

Run: `uv run --no-sync pytest tests/superposition/test_geometry.py -q`
Expected: `15 passed`

- [ ] **Step 4: Commit**

```bash
git add notebooks/superposition/superposition_geometry.ipynb
git commit -m "superposition geometry: Act 6 — pentagon stretches with one feature's density and snaps past the deformation limit"
```

---

### Task 9: Recap, glossary, full execution, hygiene

**Files:**
- Modify: `notebooks/superposition/superposition_geometry.ipynb` (recap cell; then execute in place)
- Modify: `docs/glossary.md` (feature dimensionality entry)

**Interfaces:**
- Consumes: everything.
- Produces: the executed notebook with outputs, committed.

- [ ] **Step 1: Append the recap + handoff markdown**

```markdown
## Recap — what we proved

- **Act 2.** Feature dimensionality reads known solutions exactly: dedicated → 1,
  digon → ½, pentagon → ⅖; the *trained* pentagon measured ≈ 0.4 unprompted.
- **Act 3.** Across 20 sparsities (n=200, m=20), learned features' Dᵢ concentrates on a
  handful of fractions rather than smearing — superposition is quantized. (The printed
  fraction-near-plateau number is the claim.)
- **Act 4.** The plateaus are geometry: interference components at each plateau are the
  predicted polytopes — q features sharing p dims — with Gram-matrix angles to match.
- **Act 5.** Training reaches these configurations by discrete reorganizations; loss
  drops coincide with features jumping between dimensionality levels.
- **Act 6.** Non-uniformity deforms the pentagon smoothly (neighbors retreat from a
  denser feature, close on a sparser one) until it snaps to a new configuration.

**What we deferred — book three.** Everything here is *storage*: the model only
reconstructs its inputs. The paper's most consequential result is that models can
*compute* in superposition — its toy computes |x| through a ReLU hidden layer while
features share neurons. That, plus correlated-feature geometry (real features co-occur,
and correlation reshapes the polytopes), is book three's territory. And one thread runs
further ahead: if features are directions in superposition, *recovering* them from a
real model is a dictionary-learning problem — the road to sparse autoencoders, and to
asking where Masri lives in a real residual stream.
```

- [ ] **Step 2: Add the glossary entry**

In `docs/glossary.md`, find the existing superposition entry, and add alongside it (matching the file's existing format — read it first):

```markdown
- **Feature dimensionality** — for feature *i* with embedding column `Wᵢ`,
  `Dᵢ = ‖Wᵢ‖² / Σⱼ(Ŵᵢ·Wⱼ)²`: the fraction of a hidden dimension the feature gets to
  keep after sharing with everything that interferes with it. Superposed solutions
  concentrate at fractions of small integers (½ digon, ⅔ triangle, ¾ tetrahedron,
  ⅖ pentagon, ⅜ square antiprism) — the signature that superposition is organized into
  uniform polytopes. (Elhage et al. 2022; `superposition_geometry.ipynb`.)
```

- [ ] **Step 3: Execute the notebook end-to-end (from the notebook dir)**

```bash
rm -rf notebooks/superposition/cache   # prove the cold path works end-to-end
cd notebooks/superposition
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace superposition_geometry.ipynb
cd ../..
```

Expected: completes without error. **Do not edit the .ipynb while this runs.** Record the wall-clock time; if the cold pass exceeds ~30 min, halve Act 3's steps and note the deviation in the notebook markdown.

- [ ] **Step 4: Re-run the executed notebook to verify the cached path is fast**

```bash
cd notebooks/superposition
time uv run --no-sync jupyter nbconvert --to notebook --execute --inplace superposition_geometry.ipynb
cd ../..
```

Expected: < 10 minutes (cache hit for Acts 3 and 5).

- [ ] **Step 5: Full test suite + lint**

```bash
uv run --no-sync pytest tests/ -q
uv run --no-sync ruff check .
```

Expected: all tests pass; no new ruff findings.

- [ ] **Step 6: Commit the executed notebook (outputs kept) + glossary**

```bash
git add notebooks/superposition/superposition_geometry.ipynb docs/glossary.md
git commit -m "superposition geometry: executed end-to-end — [FILL IN: the plateau-fraction number from Act 3, e.g. 'NN% of learned features sit on a named plateau']"
```

(The bracketed part is the one intentional fill-in: the commit message must name the measured result, which only exists after execution.)

- [ ] **Step 7: Push and open PR**

```bash
git push -u origin superposition-geometry
gh pr create --title "The Geometry of Superposition — book two" --body "$(cat <<'EOF'
Book two of the superposition series: feature dimensionality, the plateau sweep
(n=200, m=20, 20 sparsities in one batched loop, pinned to the paper's Colab),
polytope identification via interference components, energy-level jumps, and the
non-uniform stretch-and-snap experiment.

Also fixes book one's lib-cell tests (10/10 red on main — analysis code had been
appended to the per_feature_loss lib cell).

Spec: docs/superpowers/specs/2026-07-23-superposition-geometry-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** hypothesis → Act 0/recap; dimensionality+plateaus → Tasks 3, 5; polytopes → Task 6; energy jumps → Task 7; non-uniform → Task 8; restated lib cells → Task 2; batched device (spec amendment) → Task 4; cache+gitignore → Task 1; glossary + executed commit → Task 9. Book-one test fix (found during planning) → Task 0.
- **Known judgment calls for the executor:** Act 5's regime (n=100, m=10, S=0.85) and Act 4's thresholds (norm_cutoff=0.5, cos_threshold=0.15) are reasonable defaults, not Colab-pinned (the Colab does not publish these two experiments' exact configs in the sections we use). If results look degenerate, adjust and *say so in the notebook markdown* — never silently.
- **Honesty:** no notebook cell asserts a result; all asserts live in pytest. Every act's markdown names what a negative would look like and that it gets reported.
