# fanous-llm-lens

**A lantern (فانوس) for looking inside small language models — mechanistic interpretability tooling with a focus on Egyptian Arabic (Masri).**

> *Status: pre-alpha. Skeleton up; first experiments forthcoming.*

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
# Clone (when published)
# git clone https://github.com/barmag/fanous-llm-lens.git
# cd fanous-llm-lens

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

```python
import fanous_lens as fl

# (API surface forthcoming — currently a skeleton.)
```

## Roadmap

**Phase 0 — Foundation** *(current)*
- [x] Repo skeleton, config, sanity-check notebook
- [ ] Smoke-test transformer-lens loading a small model on ROCm
- [ ] Glossary in `docs/glossary.md` (circuits, features, heads, residual stream — defined once, linked from everywhere)

**Phase 1 — Probing on Arabic**
- [ ] Tokenizer comparison: how do Pythia / GPT-2 / mGPT tokenise MSA vs Masri?
- [ ] First probing experiment: where does dialect signal live in the residual stream?
- [ ] Reproducible prompt set (MSA + Masri pairs) checked into `eval/`

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

MIT. See `LICENSE` (forthcoming).

## Acknowledgements

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — the canonical mech interp library
- [Anthropic Interpretability research](https://transformer-circuits.pub/) — the tradition this work participates in
- Strix Halo ROCm tooling — proves consumer hardware can host this work
