# AGENTS.md — Operating context for AI coding assistants

This file is read by Claude Code, Cursor, and any other agent loaded into this repo.
Keep it short, opinionated, and current. When the project changes, update this file
**first**, before changing code.

## North Star

**fanous-llm-lens** is mechanistic interpretability tooling for small language models,
with a deliberate focus on **Egyptian Arabic (Masri)**. The name *fanous* (فانوس) is the
Ramadan lantern — a light for looking into dark places. The project is what it sounds like.

The goal is twofold:

1. **Credential** — produce hands-on, public artefacts that travel with job applications
   (Anthropic Interpretability team, Anthropic Applied AI / FDE, Meta IC6 ML/AI infra).
2. **Contribution** — interpretability work on under-studied dialects is a real open
   problem. Even modest reproducible experiments on Masri-trained or Masri-evaluated
   models add to a thin literature.

## Scope (what's in)

- Probing, circuit discovery, and SAE-style feature analysis on **small** models
  (Pythia-70m through ~1.5B params; GPT-2 family; TinyStories-class models)
- Comparative experiments: how does a model handle MSA vs Masri inputs? Where does
  the dialect signal live in the residual stream?
- Tooling: notebooks, CLI commands, and (eventually) a small Streamlit/Gradio UI
  that lets non-researchers click through model internals on Arabic prompts
- Reproducible eval harnesses with seeds + commit SHAs logged

## Scope (what's out — at least for now)

- Training new frontier models from scratch (wrong scale)
- Serving infrastructure (vLLM, TensorRT-LLM) — that was the *previous* framing of this
  project; substrate fits mech interp better
- Generic Arabic NLP without dialect awareness — if dialect doesn't matter, this isn't
  the right project
- Multi-GPU distributed work — single-node Strix Halo iGPU only

## Substrate constraints

- AMD Strix Halo iGPU. ROCm 6.4 nightly. **No CUDA-only libraries.**
- ~24-32GB unified memory. Plan for memory-tight experiments.
- Throughput reference: 70M-1.5B fp16 runs cleanly; 7B fp16 is interactive ceiling;
  larger needs quantisation. Mech interp norms (small models for tractability) are
  the *natural* fit here, not a compromise.

## Conventions

- **Package manager:** `uv` (`uv pip install …`, `uv venv`)
- **Formatting + linting:** `ruff format` + `ruff check --fix`
- **Type checking:** `basedpyright` (basic mode initially, strict per-module as code stabilises)
- **Notebooks:** clear all outputs before commit (`jupyter nbconvert --clear-output`)
  to keep diffs sane; use `nbstripout` if it's installed
- **Imports:** `import fanous_lens as fl` is the conventional alias
- **Config:** dataclasses or pydantic models, not loose dicts
- **Logging:** standard `logging` module; structured logs only if a use case demands it
- **Tests:** pytest, kept fast (CPU-only by default; mark GPU-required tests with `@pytest.mark.gpu`)

## Working with Arabic

- **Always distinguish MSA from Masri.** They are different enough that a model trained
  primarily on MSA will fail on colloquial Egyptian inputs in ways that matter for
  interpretability claims. Label datasets accordingly.
- **Tokenizer-aware:** different models tokenise Arabic differently. When reporting
  results, log the tokenizer + a few example tokenisations of representative inputs.
- **RTL display:** Arabic renders right-to-left. In notebooks, use `display(HTML(...))`
  with `dir="rtl"` for Arabic text blocks. Mixed-direction strings need explicit Unicode
  bidi marks if alignment matters.
- **Transliteration:** when storing prompts, keep both Arabic script and Buckwalter or
  ISO-233 transliteration. Researchers without Arabic proficiency should still be able
  to read commit messages and diffs.

## What "good" looks like for an experiment

- Notebook or script that runs end-to-end on a single seed in <10 min on the iGPU
- Clear hypothesis stated at the top
- Result is either a plot, a table, or a one-line numeric claim — not a vibes paragraph
- Commit message names the result, not just the change ("circuit X reproduces in Pythia-160m"
  not "add notebook")

## What to ask the user about

- New external dependencies (don't add them unilaterally)
- Anything that touches `data/` and could be sensitive
- Anything that costs money (paid API keys, cloud inference)
- Architectural pivots (rename the package, change the substrate, etc.)

## Cross-tool consistency

`CLAUDE.md` is a symlink to this file. `.cursorrules` carries a slightly more compact
version of the same context. Keep them aligned when editing — the Single Source is here.
