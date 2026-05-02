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

## Roadmap

**Phase 0 — Foundation** *(complete 2026-04-30)*
- [x] Repo skeleton, config, sanity-check notebook
- [x] Smoke-test transformer-lens loading a small model on ROCm *(covered by Cell 5 of [`notebooks/00-rocm-sanity-check.ipynb`](notebooks/00-rocm-sanity-check.ipynb), green 2026-04-30)*
- [x] Glossary in [`docs/glossary.md`](docs/glossary.md) (circuits, features, heads, residual stream — defined once, linked from everywhere)

**Phase 1 — Probing on Arabic** *(current)*
- [x] Tokenizer comparison: how do Pythia / GPT-2 / mGPT tokenise English / MSA / Masri? *([`notebooks/01-tokenizer-comparison-msa-masri.ipynb`](notebooks/01-tokenizer-comparison-msa-masri.ipynb), 2026-05-02 — gpt2 charges Arabic 2.59× the English rate, pythia 1.99×, mGPT ~1.0×; within Arabic, mGPT spends ~7.4% more on Masri than MSA. Beginner companion: [`notebooks/02-tokenization-101-masri.ipynb`](notebooks/02-tokenization-101-masri.ipynb).)*
- [ ] First probing experiment: where does dialect signal live in the residual stream?
- [x] Reproducible prompt set (English + MSA + Masri triples) checked into `eval/` *([`eval/prompts/msa-masri-pairs-v1.json`](eval/prompts/msa-masri-pairs-v1.json) — 30 hand-crafted minimal triples across 8 categories; schema v1.1 added an English baseline so tokenizer comparisons can disentangle "Arabic is hard" from "Masri is hard for an MSA-trained vocab".)*

**Phase 2 — Circuit / feature work**
- [ ] Reproduce one published circuit on a small model (e.g., IOI-style on Pythia-160m) as a baseline
- [ ] Apply the same lens to Masri inputs

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
