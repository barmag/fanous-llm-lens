# fanous-llm-lens

**A lantern (فانوس) for looking inside small language models — mechanistic interpretability tooling with a focus on Egyptian Arabic (Masri).**

> *Status: pre-alpha. Phase 0 (foundation) closed 2026-04-30; Phase 1 (probing on Arabic) in flight — first artefacts in [`notebooks/`](notebooks/).*

## Why this exists

Mechanistic interpretability — the project of reverse-engineering what circuits, features, and computations live inside trained neural networks — has produced rich tooling for English-language models. Coverage of Arabic is thin. Coverage of **dialectal** Arabic, including Egyptian Arabic (*Masri*, مصري), is thinner still.

`fanous-llm-lens` is a small, deliberate attempt to close that gap, on a substrate that fits the work:

- **Small models** (Pythia-70m through ~1.5B params, GPT-2 family, TinyStories-class) — the regime where mech interp is tractable
- **A consumer iGPU** (AMD Strix Halo, ROCm) — proves the work doesn't require a datacentre
- **Egyptian Arabic eval inputs** — distinct from Modern Standard Arabic (Fusha, الفصحى); the dialect distinction is the experimental moat

## Project layout

```
fanous-llm-lens/
├── src/fanous_lens/      # Importable package (`import fanous_lens as fl`)
├── notebooks/            # Exploratory + reproducible analysis notebooks
├── experiments/          # Standalone experiment scripts (one hypothesis per file)
├── eval/                 # Eval harnesses + prompt sets (MSA + Masri)
├── data/                 # Datasets (gitignored)
├── docs/                 # Long-form notes, glossary, paper summaries
├── .cursorrules          # Cursor agent rules
├── AGENTS.md             # Operating context for AI assistants (Claude Code / Cursor)
└── pyproject.toml        # Package metadata + ruff + basedpyright config
```

## Setup

### Prerequisites

- Linux, Python 3.11+
- AMD ROCm 6.4 (or compatible) — Strix Halo iGPU users have a working setup script at `~/code/strix-halo-ml-setup.sh`
- [`uv`](https://github.com/astral-sh/uv) for fast Python package management

### Install

```bash
git clone https://github.com/barmag/fanous-llm-lens.git
cd fanous-llm-lens

# Create virtualenv
uv venv
source .venv/bin/activate

# Install torch from ROCm nightly (separate index)
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm6.4

# Install package + dev tools
uv pip install -e ".[dev]"
```

### Verify

Open `notebooks/00-rocm-sanity-check.ipynb` and run all cells. If torch sees the iGPU and a 70M-param model loads cleanly, you're set.

## Quickstart

The package itself is still a skeleton — there's no API surface to import yet.
The work to date lives in `notebooks/` (see below); start there.

## Headline finding so far

> **`gpt2`'s tokenizer charges a Masri or MSA speaker 2.59× more tokens than an English speaker for the same meaning. `pythia-160m`'s charges 1.99×. `mGPT` — the same BPE algorithm trained on multilingual data — closes the gap to ~1.0×.**

The "tokenizer tax" is an inference-cost, latency, and context-window tax on
every minority-language user, every request. It is not a property of Arabic;
it is a property of which corpora the tokenizer's vocabulary was carved out
of. Within Arabic, the same prestige-vs-colloquial dynamic recurses one level
deeper: mGPT spends ~7.4% more tokens on Masri than on MSA. Full numbers, plot,
and method in [`notebooks/01-tokenizer-comparison-msa-masri.ipynb`](notebooks/01-tokenizer-comparison-msa-masri.ipynb);
beginner-friendly bilingual walk-through of the algorithm in [`notebooks/02-tokenization-101-masri.ipynb`](notebooks/02-tokenization-101-masri.ipynb).

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 00 | [`00-rocm-sanity-check`](notebooks/00-rocm-sanity-check.ipynb) | Environment contract — torch + ROCm + Transformers + TransformerLens all green on Strix Halo. |
| 01 | [`01-tokenizer-comparison-msa-masri`](notebooks/01-tokenizer-comparison-msa-masri.ipynb) | The tokenizer-tax measurement (English / MSA / Masri × `pythia-160m` / `gpt2` / `mGPT`). Tax table, normalised bar chart, per-triple zoom. |
| 02 | [`02-tokenization-101-masri`](notebooks/02-tokenization-101-masri.ipynb) | Bilingual (Arabic + English, RTL) walk-through of how subword tokenizers work, for a Masri-reading audience with Python but no ML background. BPE-by-hand on 6 Masri words; live BPE training on the v1 pair set; head-to-head vs `gpt2`. |
| 03 | [`03-whats-inside-the-box-masri`](notebooks/03-whats-inside-the-box-masri.ipynb) | Bilingual visual tour of a small transformer (`pythia-160m`). A Masri prompt is followed end-to-end: token IDs → embedding → residual stream → 12 layers of attention + MLP → unembedding → next-token probability. Names the parts; runs no experiments. The on-ramp from nb02 to anything interpretability-shaped. |

## Roadmap

**Phase 0 — Foundation** *(complete 2026-04-30)*
- [x] Repo skeleton, config, sanity-check notebook
- [x] Smoke-test transformer-lens loading a small model on ROCm *(covered by Cell 5 of [`notebooks/00-rocm-sanity-check.ipynb`](notebooks/00-rocm-sanity-check.ipynb), green 2026-04-30)*
- [x] Glossary in [`docs/glossary.md`](docs/glossary.md) (circuits, features, heads, residual stream — defined once, linked from everywhere)

The roadmap is organised as a **curriculum** for a Masri-reading Python user with no
prior ML background. Each phase ends where the next one needs to begin — no notebook
assumes vocabulary the reader has not already met. The interpretability findings are
real; the *path to them* is the point.

**Phase 1 — On-ramp: from text to a transformer, in Masri** *(current)*

*By the end of this phase, the reader knows what a tokenizer is, what a small
transformer is made of, and what its layers actually do — using Masri prompts at
every step. No interpretability experiments yet; this phase builds the vocabulary
the next phase needs.*

- [x] **Tokenizer comparison.** How do Pythia / GPT-2 / mGPT tokenise English / MSA / Masri? *([`notebooks/01-tokenizer-comparison-msa-masri.ipynb`](notebooks/01-tokenizer-comparison-msa-masri.ipynb), 2026-05-02 — gpt2 charges Arabic 2.59× the English rate, pythia 1.99×, mGPT ~1.0×; within Arabic, mGPT spends ~7.4% more on Masri than MSA.)*
- [x] **Tokenization-101, bilingual.** What is a subword tokenizer, by hand, for a Masri-reading audience. *([`notebooks/02-tokenization-101-masri.ipynb`](notebooks/02-tokenization-101-masri.ipynb).)*
- [x] **Reproducible prompt set.** English + MSA + Masri triples in `eval/`. *([`eval/prompts/msa-masri-pairs-v1.json`](eval/prompts/msa-masri-pairs-v1.json) — 30 hand-crafted minimal triples across 8 categories; schema v1.1 added an English baseline so tokenizer comparisons can disentangle "Arabic is hard" from "Masri is hard for an MSA-trained vocab".)*
- [ ] **nb03 — "What's inside the box?"** Bilingual visual tour of a small transformer: a Masri prompt enters as token IDs, becomes embedding vectors, flows through attention + MLP layers as a residual stream, exits as a probability distribution over the next token. Names the parts; runs no experiments.
- [ ] **nb04 — "What does the model see at each layer?"** Logit lens on a Masri prompt: at every layer, project the residual stream to vocab space and watch the model's running best-guess evolve. Reader gets a felt sense that layers *do* something, before any formal interpretability vocabulary lands.

**Phase 2 — Looking inside: probes, circuits, features**

*With Phase 1's vocabulary in hand, the reader is ready for actual interpretability
methods. Each method gets a Masri-grounded application after an English / MSA
baseline the reader can sanity-check against the literature.*

- [ ] **nb05 — Does the model know it's reading Masri?** Train a linear probe on residual-stream activations at every layer to classify MSA vs Masri. Output: one plot of probe accuracy vs depth. The first real interpretability finding on dialect.
- [ ] **Circuit reproduction.** Reproduce one published circuit (e.g., IOI-style on Pythia-160m) on English first, as a baseline.
- [ ] **Apply the same circuit lens to Masri inputs.** Does the circuit fire? Misfire? Degrade gracefully? This is the dialect-aware contribution.

**Phase 3 — User-facing surface**
- [ ] Streamlit/Gradio mini-UI for non-researcher exploration of one experiment
- [ ] CLI (`fanous`) for headless use

## Substrate notes

Throughput baseline on Strix Halo iGPU (ROCm):

| Model size | Precision | Approx. tok/s |
|---|---|---|
| 70M-410M | fp16 | very fast (interactive) |
| 1.5B | fp16 | ~30 tok/s |
| 7B | fp16 | interactive ceiling |
| 24B | fp16 | ~30 tok/s (per memory) |
| 70B | Q5 | ~5-10 tok/s |

Mech interp work lives mostly in the top two rows. This is the *natural* fit for the substrate, not a compromise.

## Naming

*Fanous* (فانوس) is the Ramadan lantern, a Cairo / Egyptian tradition. Lanterns light dark places — which is the mech-interp project description, near-literally. The cultural rooting is deliberate: a project targeting Egyptian Arabic users named in colloquial Egyptian Arabic, not Latinised academic English.

## Licence

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — the canonical mech interp library
- [Anthropic Interpretability research](https://transformer-circuits.pub/) — the tradition this work participates in
- Strix Halo ROCm tooling — proves consumer hardware can host this work
