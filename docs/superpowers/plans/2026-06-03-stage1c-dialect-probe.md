# Stage 1c Dialect-Probe Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the noisy mGPT raw-embedding PCA in the Stage 1c subword notebook with a self-contained zero-layer model trained on our own MSA/Masri corpus, then a supervised dialect probe contrasted against unsupervised PCA.

**Architecture:** The reference notebook loads the Stage 1b corpus, trains a small BPE tokenizer, trains a zero-layer transformer (`embedding → unembedding → softmax next token`) to produce `W_E`, derives dialect labels from per-stream token frequency, and renders a side-by-side figure: unsupervised PCA (dialect colours mixed) vs a logistic-regression probe axis (dialect split + accuracy). The analysis cell is factored into small pure helpers so it can be unit-tested with a synthetic `W_E` and no training/network.

**Tech Stack:** Python, Jupyter (authored as raw `.ipynb` JSON via the Write tool), `tokenizers` (BPE), `torch` (training loop), `scikit-learn` (PCA + LogisticRegression), `plotly` (`make_subplots`), `datasets` (corpus), `pytest`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `notebooks/education/stage1_c_subword_reference.ipynb` | Full rewrite. 10 cells (5 markdown / 4 code / 1 badge). Code cells: install, corpus+BPE, train model, analysis. |
| `notebooks/education/stage1_c_subword_experiment.ipynb` | Full rewrite, mirror structure with TODO scaffolds in the code cells (no implementation). |
| `tests/education/test_stage1c_probe.py` | New. Fast unit tests (no network/training) on the analysis cell's helpers + figure. |
| `notebooks/education/verify_notebooks.py` | Modify `mock_stage1_c` (lines 59-82): drop the transformers/scatter_3d mock, mock only `go.Figure.show`. |

**Code-cell index contract:** after the rewrite, `code_cells = [install(0), corpus_bpe(1), train_model(2), analysis(3)]`. The unit test loads `code_cells[3]` — the same convention as `test_stage1b_graph.py`.

**Analysis-cell interface** (defined in code cell 3, relied on by the test):

- `dialect_labels(msa_ids, masri_ids, min_count=5, hi=0.7, lo=0.3) -> dict[int, str]`
- `probe_dialect(W_E, labels, seed=0, test_size=0.3) -> dict` with keys `accuracy`, `direction`, `projections`, `boundary`
- `run_pca(W_E, ids, seed=0) -> np.ndarray` of shape `(len(ids), 2)`
- `plot_pca_vs_probe(W_E, labels, probe, id_to_token) -> go.Figure`
- Driver exposes module globals `labels`, `probe`, `fig`, using injected `W_E`, `msa_ids`, `masri_ids`, `id_to_token`.

---

## Task 1: Failing unit test for the analysis cell

**Files:**
- Create: `tests/education/test_stage1c_probe.py`

- [ ] **Step 1: Write the failing test**

Create `tests/education/test_stage1c_probe.py` with this exact content:

```python
"""Tests for the Stage 1c dialect-probe analysis cell.

We load the analysis cell (code-cell index 3) out of the reference notebook
and exec it against a tiny synthetic, linearly-separable W_E plus synthetic
id streams, so the test needs no network, no GPU, and no training. The cell
defines dialect_labels, probe_dialect, run_pca, plot_pca_vs_probe and a
driver that builds `labels`, `probe`, `fig`.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "education" / "stage1_c_subword_reference.ipynb"
)

# Synthetic vocabulary of 10 subword ids.
#   ids 0-3  -> MSA-only      (cluster near +x)
#   ids 6-9  -> Masri-only    (cluster near -x)
#   ids 4-5  -> Shared        (near origin)
_MSA_IDS = [0, 1, 2, 3] * 5 + [4, 5] * 3
_MASRI_IDS = [6, 7, 8, 9] * 5 + [4, 5] * 3

# Linearly separable embeddings: MSA at +5 on axis 0, Masri at -5, shared at 0.
_W_E = np.zeros((10, 4), dtype=float)
for _i in range(10):
    if _i <= 3:
        _W_E[_i] = [5.0 + 0.1 * _i, 0.1 * _i, 0.0, 0.0]
    elif _i >= 6:
        _W_E[_i] = [-5.0 - 0.1 * _i, 0.1 * _i, 0.0, 0.0]
    else:
        _W_E[_i] = [0.1 * _i, 0.1, 0.0, 0.0]

_ID_TO_TOKEN = {i: f"tok{i}" for i in range(10)}


def _load_analysis_cell():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # code cells: 0=install, 1=corpus+bpe, 2=train, 3=analysis
    src = "".join(code_cells[3]["source"])
    ns = {
        "W_E": _W_E,
        "msa_ids": _MSA_IDS,
        "masri_ids": _MASRI_IDS,
        "id_to_token": _ID_TO_TOKEN,
    }
    go.Figure.show = lambda self: None  # headless no-op
    exec(compile(src, f"{NB}:analysiscell", "exec"), ns)
    return ns


def test_dialect_labels_assigns_by_frequency():
    ns = _load_analysis_cell()
    labels = ns["dialect_labels"](_MSA_IDS, _MASRI_IDS, min_count=5)
    assert labels[0] == "MSA"      # appears only in MSA stream
    assert labels[9] == "Masri"    # appears only in Masri stream
    assert labels[4] == "Shared"   # 50/50 split across streams


def test_dialect_labels_drops_rare_tokens():
    ns = _load_analysis_cell()
    # token 7 appears 5 times (>= min_count 5); raise the floor above that.
    labels = ns["dialect_labels"](_MSA_IDS, _MASRI_IDS, min_count=100)
    assert labels == {}


def test_probe_separates_separable_embeddings():
    ns = _load_analysis_cell()
    labels = ns["dialect_labels"](_MSA_IDS, _MASRI_IDS, min_count=5)
    probe = ns["probe_dialect"](_W_E, labels, seed=0)
    assert probe["accuracy"] >= 0.8
    assert probe["direction"].shape == (_W_E.shape[1],)
    # every labelled token gets a scalar projection
    assert set(probe["projections"]) == set(labels)


def test_run_pca_returns_2d():
    ns = _load_analysis_cell()
    coords = ns["run_pca"](_W_E, [0, 1, 2, 6, 7, 8])
    assert coords.shape == (6, 2)


def test_figure_has_two_panels_with_accuracy():
    ns = _load_analysis_cell()
    fig = ns["fig"]
    assert isinstance(fig, go.Figure)
    # make_subplots(cols=2) creates a second x-axis
    assert fig.layout.xaxis2 is not None
    # both a PCA scatter and a probe histogram are drawn
    assert any(t.type == "scatter" for t in fig.data)
    assert any(t.type == "histogram" for t in fig.data)
    # the probe accuracy is surfaced somewhere in the figure text
    ann_texts = [a.text or "" for a in fig.layout.annotations]
    assert any("acc" in t.lower() for t in ann_texts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/education/test_stage1c_probe.py -v`
Expected: FAIL — the current notebook has only 3 code cells, so `code_cells[3]` raises `IndexError` (or the old cell lacks the helper names). All five tests error/fail.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/education/test_stage1c_probe.py
git commit -m "test(stage1c): add failing unit tests for dialect-probe analysis cell"
```

---

## Task 2: Rewrite the reference notebook

**Files:**
- Modify (full overwrite): `notebooks/education/stage1_c_subword_reference.ipynb`
- Test: `tests/education/test_stage1c_probe.py`

This task authors the `.ipynb` as raw JSON with the Write tool. All `outputs` are `[]` and `execution_count` is `null` (satisfies the "clear outputs before commit" convention). The notebook has 10 cells in this order. Use the exact cell sources below.

- [ ] **Step 1: Write the reference notebook**

Write `notebooks/education/stage1_c_subword_reference.ipynb` as a notebook JSON (`nbformat` 4, `nbformat_minor` 5) whose `cells` array contains the following, in order. Each code cell's `source` is the literal text shown. **Critical convention:** the Arabic range in the cleaning regex MUST be written as the `\u0621-\u064A` unicode escapes (exactly as Stage 1b does) — do NOT substitute literal Arabic characters. Arabic words inside markdown and the example-tokenisation list are real Arabic characters; only the regex range uses escapes.

**Cell 0 — markdown (badge, unchanged):**

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barmag/fanous-llm-lens/blob/main/notebooks/education/stage1_c_subword_reference.ipynb)
```

**Cell 1 — markdown (intro / hypothesis / limits):**

```markdown
# Stage 1c: Zero-Layer Transformer (Subword/BPE-Level) 🧩
## target: Is "dialect" a direction you can find in learned embeddings?

We counted characters (Stage 1a) and word transitions (Stage 1b). Now we take the
last step of the embeddings trilogy: instead of *counting*, we **learn a vector for
every subword** by training a tiny model on our own MSA + Masri text — then we ask one
sharp question about those vectors.

### 💡 What we are showing
1. Train a small **BPE tokenizer** on a mixed MSA+Masri corpus (no giant download — the
   subwords come from *our* data).
2. Train a **zero-layer transformer** — literally `embedding → unembedding → softmax over
   the next token`. This is a subword *bigram* predictor, and its embedding matrix `W_E`
   is the thing we inspect.
3. Ask: **is dialect linearly encoded in `W_E`?** We compare two views of the *same*
   vectors —
   - **Unsupervised PCA**, which surfaces the *loudest* variance (frequency/length) and
     leaves the dialect colours mixed, versus
   - **A supervised probe**, which finds the *one axis* that separates MSA from Masri and
     reports an accuracy number.

> ### ⚠️ Up front — what this is *not*
> These are **toy embeddings** from a one-matrix model trained on a small corpus. They
> capture co-occurrence, not deep meaning. Rich contextual semantics only emerge once a
> model has real layers — that is a *later* stage. The lesson here is the method:
> **PCA shows what is biggest; a probe shows what you asked for.**
```

**Cell 2 — code (install):**

```python
# 🛠️ Setup: Install dependencies if running on Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q tokenizers datasets scikit-learn plotly pandas torch
```

**Cell 3 — markdown (name the corpus + BPE step):**

```markdown
## 📚 Step 1 — Build subwords from our own dialect data
We reuse the Stage 1b corpus: **Modern Standard Arabic** from Wikipedia and **Egyptian
(Masri)** from real street tweets. We train a small **BPE** tokenizer on the two combined,
then encode each stream into subword ids. Keeping the two streams separate matters — the
per-stream counts are exactly how we will label each subword's dialect later.
```

**Cell 4 — code (corpus + BPE + encode):**

```python
# 📦 Fetch MSA + Masri text, then learn a BPE vocabulary from it
from datasets import load_dataset
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

print("Fetching MSA from Wikipedia and Masri from Tweets...")
msa_stream = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
tweets_ds = load_dataset("amgadhasan/arabic_tweets_dialects", split="train")
eg_tweets = tweets_ds.filter(lambda x: x['dialect'] == 'EG')

def clean_msa(stream, max_chars=500000):
    text = ""
    for article in stream:
        cleaned = re.sub(r'\s+', ' ', article['text'])
        cleaned = re.sub(r'[^\s\u0621-\u064A]', '', cleaned)
        text += cleaned + " "
        if len(text) >= max_chars:
            break
    return text[:max_chars]

def clean_masri(tweets, max_chars=500000):
    text = ""
    for t in tweets:
        cleaned = re.sub(r'\s+', ' ', t['text'])
        cleaned = re.sub(r'[a-zA-Z0-9_@]+', '', cleaned)
        cleaned = re.sub(r'[^\s\u0621-\u064A]', '', cleaned)
        text += cleaned + " "
        if len(text) >= max_chars:
            break
    return text[:max_chars]

print("Collecting MSA...")
msa_text = clean_msa(msa_stream)
print("Collecting Masri...")
masri_text = clean_masri(eg_tweets)

# Train a small BPE vocabulary on the two dialects combined.
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=3000, special_tokens=["[UNK]"])
tokenizer.train_from_iterator([msa_text, masri_text], trainer)

# Encode each stream to subword ids. Shapes: msa_ids/masri_ids -> [n_subwords]
msa_ids = tokenizer.encode(msa_text).ids
masri_ids = tokenizer.encode(masri_text).ids
id_to_token = {i: t for t, i in tokenizer.get_vocab().items()}

print(f"Vocab size: {tokenizer.get_vocab_size()} subwords")
print(f"Encoded {len(msa_ids)} MSA / {len(masri_ids)} Masri subword tokens.")
# Tokenizer-aware logging: a few example fracturings (RTL words shown left-to-right
# as token lists for inspection).
for w in ["الذي", "اللي", "دلوقتي", "الآن", "العربية"]:
    print(f"  {w} -> {tokenizer.encode(w).tokens}")
```

**Cell 5 — markdown (name the zero-layer model):**

```markdown
## 🧠 Step 2 — Train the zero-layer transformer
A **zero-layer transformer** has no attention and no MLP: a token's embedding goes
straight to the unembedding to predict the next token. That makes it a **subword bigram
model**. We train it with cross-entropy on `(current → next)` pairs pooled from both
streams; the learned embedding matrix `W_E` (shape `[vocab, 64]`) is our artefact.
```

**Cell 6 — code (train the model → W_E):**

```python
# 🏋️ Train embedding (W_E) + unembedding (W_U) to predict the next subword
import torch
import numpy as np
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

EMBED_DIM = 64
EPOCHS = 3
BATCH = 4096
LR = 1e-2

def make_pairs(ids):
    # consecutive (current, next) subword ids within one stream -> [n-1, 2]
    return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]

pairs = make_pairs(msa_ids) + make_pairs(masri_ids)  # don't bridge the two streams
pairs = torch.tensor(pairs, dtype=torch.long)

vocab_size = tokenizer.get_vocab_size()
embed = torch.nn.Embedding(vocab_size, EMBED_DIM)
unembed = torch.nn.Linear(EMBED_DIM, vocab_size, bias=False)
opt = torch.optim.Adam(list(embed.parameters()) + list(unembed.parameters()), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    perm = torch.randperm(pairs.shape[0])
    running = 0.0
    n_batches = 0
    for b in range(0, pairs.shape[0], BATCH):
        idx = perm[b:b + BATCH]
        cur, nxt = pairs[idx, 0], pairs[idx, 1]
        logits = unembed(embed(cur))           # [batch, vocab]
        loss = loss_fn(logits, nxt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
        n_batches += 1
    print(f"epoch {epoch}  mean loss {running / n_batches:.3f}")

# The learned embeddings. Shape: [vocab, EMBED_DIM]
W_E = embed.weight.detach().numpy()
print(f"Trained W_E shape: {W_E.shape}")
```

**Cell 7 — markdown (explain labels + PCA vs probe):**

```markdown
## 🔍 Step 3 — PCA vs. a dialect probe
**Free labels.** Every subword's dialect comes from the data itself: count how often it
appears in the MSA vs Masri stream. Mostly-MSA → `MSA`, mostly-Masri → `Masri`, roughly
even → `Shared` (rare tokens are dropped).

**Two views of the same `W_E`:**
- **PCA** (unsupervised) plots the two directions of largest variance. It does not know
  about dialect, so the colours tend to stay mixed — it shows what is *biggest*.
- **A linear probe** (logistic regression) is *told* the labels and finds the single axis
  that best separates MSA from Masri. We project every subword onto that axis and read off
  a held-out **accuracy**. `Shared` tokens are held out of training and should land in the
  middle.
```

**Cell 8 — code (analysis: labels + PCA + probe + side-by-side plot):**

```python
# 🎨 Dialect labels, PCA, and a supervised probe — side by side
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go

COLORS = {"MSA": "red", "Masri": "blue", "Shared": "green"}

def dialect_labels(msa_ids, masri_ids, min_count=5, hi=0.7, lo=0.3):
    """Label each subword by which stream it favours. -> {id: 'MSA'|'Masri'|'Shared'}"""
    cm, cs = Counter(msa_ids), Counter(masri_ids)
    labels = {}
    for tok in set(cm) | set(cs):
        m, s = cm.get(tok, 0), cs.get(tok, 0)
        total = m + s
        if total < min_count:
            continue
        share = m / total           # fraction of appearances in the MSA stream
        if share >= hi:
            labels[tok] = "MSA"
        elif share <= lo:
            labels[tok] = "Masri"
        else:
            labels[tok] = "Shared"
    return labels

def probe_dialect(W_E, labels, seed=0, test_size=0.3):
    """Logistic probe: can a single linear axis separate MSA from Masri?"""
    bin_ids = [t for t, l in labels.items() if l in ("MSA", "Masri")]
    X = W_E[bin_ids]
    y = np.array([1 if labels[t] == "MSA" else 0 for t in bin_ids])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    direction = clf.coef_[0]                       # the dialect axis, shape [dim]
    projections = {t: float(W_E[t] @ direction) for t in labels}
    boundary = -float(clf.intercept_[0])           # w·x = -b at the decision boundary
    return {
        "accuracy": clf.score(X_te, y_te),
        "direction": direction,
        "projections": projections,
        "boundary": boundary,
    }

def run_pca(W_E, ids, seed=0):
    """2-component PCA of the selected rows. -> [len(ids), 2]"""
    return PCA(n_components=2, random_state=seed).fit_transform(W_E[ids])

def plot_pca_vs_probe(W_E, labels, probe, id_to_token):
    ids = list(labels.keys())
    coords = run_pca(W_E, ids)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("PCA (unsupervised) — mixed",
                        f"Dialect probe axis — acc {probe['accuracy']:.2f}"),
    )
    # Left: PCA scatter, coloured by dialect.
    for dia in ("MSA", "Masri", "Shared"):
        rows = [k for k, t in enumerate(ids) if labels[t] == dia]
        if not rows:
            continue
        fig.add_trace(
            go.Scatter(
                x=coords[rows, 0], y=coords[rows, 1], mode="markers",
                marker=dict(color=COLORS[dia], size=6), name=dia,
                text=[id_to_token.get(ids[k], str(ids[k])) for k in rows],
            ),
            row=1, col=1,
        )
    # Right: histogram of probe-axis projections, per dialect.
    for dia in ("MSA", "Masri", "Shared"):
        vals = [probe["projections"][t] for t in ids if labels[t] == dia]
        if not vals:
            continue
        fig.add_trace(
            go.Histogram(x=vals, marker=dict(color=COLORS[dia]), opacity=0.6,
                         name=dia, showlegend=False),
            row=1, col=2,
        )
    fig.add_vline(x=probe["boundary"], line=dict(color="white", dash="dash"),
                  row=1, col=2)
    fig.update_layout(barmode="overlay", width=950, height=480,
                      title="Learned subword embeddings: PCA can't see dialect, a probe can")
    return fig

# Driver
labels = dialect_labels(msa_ids, masri_ids)
probe = probe_dialect(W_E, labels)
print(f"Held-out dialect-probe accuracy: {probe['accuracy']:.2f}  "
      f"({sum(v=='MSA' for v in labels.values())} MSA / "
      f"{sum(v=='Masri' for v in labels.values())} Masri / "
      f"{sum(v=='Shared' for v in labels.values())} Shared subwords)")
fig = plot_pca_vs_probe(W_E, labels, probe, id_to_token)
fig.show()
```

**Cell 9 — markdown (recap + handoff):**

```markdown
## ✅ Recap & hand-off
- We **learned** a vector per subword by training a one-matrix (zero-layer) model on our
  own MSA + Masri text — completing the trilogy: *count chars → count words → learn vectors*.
- **PCA** of those vectors leaves the dialects mixed: the loudest variance is frequency and
  length, not dialect.
- A **linear probe**, told the labels, recovers a dialect axis with clear held-out
  accuracy — so **dialect is linearly encoded even in a zero-layer model trained on mixed
  data**. The takeaway is the method: *PCA shows what is biggest; a probe shows what you
  asked for.*

**Next:** with real attention + MLP layers, embeddings start carrying *meaning*, not just
co-occurrence. That is where contextual semantics — and the MSA↔Masri story inside the
residual stream — pick up in the next stage.
```

- [ ] **Step 2: Run the unit tests to verify they pass**

Run: `pytest tests/education/test_stage1c_probe.py -v`
Expected: PASS — all five tests green (the analysis cell now defines `dialect_labels`, `probe_dialect`, `run_pca`, `plot_pca_vs_probe`, and builds `fig` from the injected synthetic globals).

- [ ] **Step 3: Sanity-check the notebook JSON is valid**

Run: `python -c "import json; json.load(open('notebooks/education/stage1_c_subword_reference.ipynb')); print('valid json')"`
Expected: prints `valid json`.

- [ ] **Step 4: Confirm outputs are cleared**

Run: `python -c "import json; nb=json.load(open('notebooks/education/stage1_c_subword_reference.ipynb')); print('all clear' if all((c.get('outputs',[])==[] and c.get('execution_count') is None) for c in nb['cells'] if c['cell_type']=='code') else 'has outputs')"`
Expected: prints `all clear`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/education/stage1_c_subword_reference.ipynb tests/education/test_stage1c_probe.py
git commit -m "feat(stage1c): rewrite reference notebook as learned-embedding dialect probe"
```

---

## Task 3: Rewrite the experiment notebook (scaffolded)

**Files:**
- Modify (full overwrite): `notebooks/education/stage1_c_subword_experiment.ipynb`

The experiment notebook mirrors the reference markdown exactly but replaces the three substantive code cells (corpus+BPE, train, analysis) with TODO scaffolds. The install cell stays identical.

- [ ] **Step 1: Write the experiment notebook**

Write `notebooks/education/stage1_c_subword_experiment.ipynb` as notebook JSON with the same 10-cell structure as the reference, with `outputs: []` / `execution_count: null`, **identical** cells 0, 1, 2, 3, 5, 7, 9 (badge, intro, install, and the three step markdown cells, recap), and these scaffolded code cells:

**Cell 4 — code (corpus + BPE scaffold):**

```python
# 📦 Fetch MSA + Masri text, then learn a BPE vocabulary from it
# TODO:
#   1. load_dataset("wikimedia/wikipedia", "20231101.ar", streaming=True) for MSA and
#      load_dataset("amgadhasan/arabic_tweets_dialects") filtered to dialect == "EG".
#   2. Clean each: collapse whitespace, strip non-Arabic with the regex range
#      [^\s\u0621-\u064A]; build msa_text and masri_text (~500k chars each).
#   3. Train a BPE tokenizer (tokenizers.Tokenizer + BpeTrainer, vocab_size=3000,
#      Whitespace pre-tokenizer) on [msa_text, masri_text] combined.
#   4. Produce msa_ids, masri_ids (tokenizer.encode(...).ids) and
#      id_to_token = {i: t for t, i in tokenizer.get_vocab().items()}.
#   5. Print the vocab size and a few example tokenizations (الذي, اللي, دلوقتي).
```

**Cell 6 — code (train scaffold):**

```python
# 🏋️ Train embedding (W_E) + unembedding (W_U) to predict the next subword
# TODO:
#   1. Seed torch / numpy / random for reproducibility.
#   2. Build (current, next) subword pairs *within* each stream (don't bridge streams).
#   3. Define a zero-layer model: torch.nn.Embedding(vocab, 64) -> torch.nn.Linear(64,
#      vocab, bias=False). No attention, no MLP — that's the whole model.
#   4. Train a few epochs with Adam + CrossEntropyLoss; print the falling loss.
#   5. Expose W_E = embed.weight.detach().numpy()  # shape [vocab, 64]
```

**Cell 8 — code (analysis scaffold):**

```python
# 🎨 Dialect labels, PCA, and a supervised probe — side by side
# TODO:
#   1. dialect_labels(msa_ids, masri_ids): count each id per stream; label MSA / Masri /
#      Shared by the MSA share (e.g. >=0.7 -> MSA, <=0.3 -> Masri), drop rare ids.
#   2. PCA: sklearn PCA(n_components=2) on the labelled rows of W_E -> scatter, coloured
#      by dialect. Notice the colours stay MIXED (unsupervised shows the biggest variance).
#   3. Probe: train sklearn LogisticRegression on W_E rows (MSA vs Masri; hold out Shared),
#      report held-out accuracy, take clf.coef_[0] as the dialect axis, and project every
#      token onto it -> histogram per dialect with the decision boundary.
#   4. Render both with plotly make_subplots(rows=1, cols=2) and fig.show().
#   The point: PCA can't see dialect, but the probe can.
```

- [ ] **Step 2: Validate JSON + cleared outputs**

Run: `python -c "import json; nb=json.load(open('notebooks/education/stage1_c_subword_experiment.ipynb')); print('valid'); assert all((c.get('outputs',[])==[] and c.get('execution_count') is None) for c in nb['cells'] if c['cell_type']=='code'); print('all clear')"`
Expected: prints `valid` then `all clear`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/education/stage1_c_subword_experiment.ipynb
git commit -m "feat(stage1c): rewrite experiment notebook with dialect-probe scaffolds"
```

---

## Task 4: Update the end-to-end verify harness

**Files:**
- Modify: `notebooks/education/verify_notebooks.py:59-82` (`mock_stage1_c`)

The current `mock_stage1_c` mocks `transformers.AutoModel` and `plotly.express.scatter_3d`, which the new notebook no longer uses. Replace it with a `Figure.show` no-op like `mock_stage1_b`.

- [ ] **Step 1: Replace the mock function**

In `notebooks/education/verify_notebooks.py`, replace the entire `mock_stage1_c` function (the block from `def mock_stage1_c(ctx):` through the `transformers.AutoModel.from_pretrained = ...` line) with:

```python
def mock_stage1_c(ctx):
    # New Stage 1c trains a tiny zero-layer model locally and renders with
    # plotly graph_objects; only the figure display needs mocking.
    import plotly.graph_objects as go
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")
```

- [ ] **Step 2: Run the unit tests (regression guard)**

Run: `pytest tests/education/ -v`
Expected: PASS — both `test_stage1b_graph.py` and `test_stage1c_probe.py` green.

- [ ] **Step 3: Run the notebook end-to-end on real data**

Run: `cd notebooks/education && python verify_notebooks.py c`
Expected: downloads the corpus, trains BPE + the zero-layer model (loss prints fall across 3 epochs), prints a non-trivial held-out probe accuracy, `[Mock] plotly.Figure.show() called.`, then `🎉 REFERENCE NOTEBOOKS (c) VERIFIED SUCCESSFULLY!` and exit 0. (Network + a couple of minutes of CPU training expected.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/education/verify_notebooks.py
git commit -m "test(stage1c): update verify harness mock for learned-embedding notebook"
```

---

## Self-Review

**1. Spec coverage:**
- Self-contained BPE on combined corpus, no model download → Task 2 cell 4. ✓
- Reuse 1b corpus/loaders/cleaning with `ء-ي` escapes → Task 2 cell 4. ✓
- Zero-layer `embedding → unembedding → softmax` bigram, seeded, dim 64 → Task 2 cell 6. ✓
- Free dialect labels from per-stream frequency (MSA/Masri/Shared, freq floor, named thresholds) → Task 2 cell 8 `dialect_labels`. ✓
- PCA (unsupervised, coloured, mixed) → cell 8 `run_pca` + left panel. ✓
- Probe (logistic, MSA vs Masri, Shared held out & shown middle, held-out accuracy, axis projection + boundary) → cell 8 `probe_dialect` + right panel. ✓
- Side-by-side render → cell 8 `plot_pca_vs_probe` (`make_subplots` 1×2). ✓
- Pedagogical patterns (name-then-experiment markdown before each code step, upfront limits, shape spine comments, recap+handoff) → cells 1,3,5,7,9. ✓
- Experiment scaffold mirror → Task 3. ✓
- Fast unit tests (synthetic, no network/training) → Task 1. ✓
- Manual `verify_notebooks.py c` end-to-end + harness update → Task 4. ✓
- Colab badge unchanged/correct → Task 2 cell 0. ✓

**2. Placeholder scan:** No "TBD"/"implement later" in the *reference* code; the only TODOs are intentional scaffolds in the *experiment* notebook (Task 3), which is by design. ✓

**3. Type consistency:** `dialect_labels` returns `{id: str}`; `probe_dialect` consumes that dict and returns `{accuracy, direction, projections, boundary}`; `plot_pca_vs_probe(W_E, labels, probe, id_to_token)` consumes both and `run_pca(W_E, ids)`. Test injects exactly `W_E, msa_ids, masri_ids, id_to_token` and reads `dialect_labels, probe_dialect, run_pca, fig`. Names align across Tasks 1, 2, 4. ✓
```
