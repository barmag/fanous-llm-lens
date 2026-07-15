# Induction heads in the wild — design

**Date:** 2026-07-15 · **Status:** design approved
**Branch:** `induction-heads-in-the-wild`

## North star

The blog post *"I Didn't Understand QKV, So I Hand-Crafted an Induction Head"*
(barmag.github.io, 2026-07-15) ends on an open question:

> If I take a small open-weights model like Pythia and go looking for its
> previous-token head and its induction head, will the weights look anything like
> the shift and projection I just wrote by hand? Or does training find some smeared
> version, spread across heads, that only approximates this behavior?

This project answers that question with **one reference notebook** that hunts the
hand-crafted circuit's four jobs inside two pre-trained models and reports, job by
job, whether training found the same solution.

## Deliverable

`notebooks/in_context_learning/induction_heads_in_the_wild.ipynb` — a single
fully-worked reference notebook (reference-only, per the 2026-07-13 decision; no
`*_experiment` twin), sitting next to `icl_from_scratch.ipynb`. Same genre: one
concept per cell block, paper-hooked sections, honest numbers.

**Papers hooked:** Olsson et al. 2022 (*In-context Learning and Induction Heads*)
for the behavioral scores; Elhage et al. 2021 (*A Mathematical Framework for
Transformer Circuits*) for QK/OV circuits and composition scores.

**No Arabic/Masri content this time** — this is a blog-companion anatomy notebook;
the dialect track is unaffected.

## Hypothesis (stated at the top of the notebook)

The hand-crafted two-head circuit — a shift-QK previous-token head feeding a
K-composed, copying induction head — is reproducible in pre-trained models:

1. GPT-2 small's previous-token head implements `shift` visibly in its
   **positional QK circuit** (subdiagonal stripe).
2. Both models' induction heads show **token-identity QK matching** (through
   composition with the prev-token head), **copying OV**, and a **K-composition
   score** that singles out the prev-token head.
3. **Falsifiable twist:** in Pythia-160m the `shift` matrix should *not exist as a
   weight-space object*, because rotary embeddings (RoPE) never write position into
   the residual stream — the same behavior, implemented without the matrix the toy
   used.

## Models

| Model | Why |
|---|---|
| **Pythia-160m** | The blog's own framing; proven on the ROCm stack in nb06; 12L×12H — big enough for crisp induction heads, small enough to sweep every head in minutes. RoPE makes it the contrast case for job 2a. |
| **GPT-2 small** | Cross-family universality check; learned absolute positional embeddings make the `shift` question answerable matrix-by-matrix; best-documented model in the induction literature, so published head labels sanity-check our detection. |

Published GPT-2 head labels are verified against the primary source at build time,
not quoted from memory.

## Notebook structure — five acts

**Act 0 — The toy, re-run.** The blog's ~50-line NumPy toy pasted in and executed
(instant, no download). Re-establishes the object of comparison: 8 matrices, two
jobs, 87% prediction. Everything after is "find each piece in a trained model."

**Act 1 — Find the heads (behavioral).** A batch of repeated random-token
sequences (fixed seed; ~64 sequences, a ~50-token block repeated twice). For every
head in both models, from attention patterns:

- **Prev-token score** — mean attention to position `i−1`.
- **Induction score** — mean attention to the token that followed the previous
  occurrence of the current token (offset `i − T + 1` for period `T`).

Output: two layers×heads heatmaps per model. The **top-1 head per job per model**
(one prev-token head, one induction head, ×2 models = 4 specimens) carries Act 2;
runners-up stay visible in the heatmaps and feed the "smearing" discussion in
Act 3. GPT-2 candidates checked against published labels.

**Act 2 — Open the weights, one toy job at a time.** Each section opens with "in
the toy, this matrix did X — where does that job live in the trained model?"

| | Toy matrix | Learned counterpart | Metric |
|---|---|---|---|
| 2a | `W_Q1`/`W_K1` = `shift×3` | GPT-2: positional QK circuit `pos_embed · W_Q · W_Kᵀ · pos_embedᵀ` of the prev-token head → subdiagonal stripe | Fraction of positional-QK attention mass on offset −1. Pythia: no matrix to compute (RoPE) — demonstrated behaviorally instead: the prev-token attention pattern survives token shuffling, so it is position-driven with no positional weight object. **This contrast is the notebook's twist finding.** |
| 2b | `W_Q2`/`W_K2` = `I×2` token match | QK circuit composed through the prev-token head's OV: `W_Eᵀ · W_Q^ind · W_K^indᵀ · W_OV^prev · W_E` | Diagonal dominance on a fixed-seed vocab sample of ~1,000 tokens (the induction head asks "who has *my* token in their before-me slot"). |
| 2c | K-composition (L1 output → L2 key) | Elhage et al. composition score between the induction head's QK and *every* earlier head's OV | Bar chart per earlier head; the prev-token head should tower. RoPE caveat noted honestly for Pythia (raw-weights approximation, standard practice). |
| 2d | `W_V2`/`W_O2` = `I×4` copy + write-back | Full OV circuit `W_U · W_O^ind · W_V^ind · W_E` | Copying score = fraction of positive eigenvalues. The 50k×50k matrix has rank ≤ `d_head`, so its nonzero eigenvalues equal those of a `d_head×d_head` matrix (eigs of AB = eigs of BA) — computed exactly, no sampling. The toy's `I×4` scores 1.0 by construction. |

**Act 3 — Verdict.** Summary table: rows = the toy's 4 jobs, columns = GPT-2 /
Pythia-160m, cells = the numeric evidence plus "found as a matrix / found as
behavior only". Honest-gaps paragraph: what smeared (several heads sharing each
job, fractional scores vs the toy's exact ones), what didn't exist as a matrix at
all (Pythia's shift). Recap + handoff: **causal ablation** (knock out the found
heads, watch induction collapse) named as the deferred next step.

## Infrastructure & conventions

- **Stack:** TransformerLens `HookedTransformer` for both models (proven in nb06
  on gfx1151/ROCm). No new dependencies. `uv run --no-sync`.
- **Memory/runtime:** both models together ~1.2GB — trivial for the iGPU. The
  behavioral sweep caches attention patterns only (`names_filter`) to bound
  activation memory. Well under the <10-min bar after first model download; no
  checkpoint-caching machinery needed. Fixed seed + commit SHA logged in the
  header cell.
- **Honest negatives:** no pass/fail gates. Weak induction scores or a muddy
  composed-QK diagonal are reported as the numbers they are, with a named reason,
  and feed the verdict table either way.
- **Outputs:** baked into the committed notebook (follows the newer
  save-outputs precedent from `9b7685c`, diverging deliberately from the older
  clear-outputs convention — results are the point of a reference notebook).
- **Branching:** new branch `induction-heads-in-the-wild` off `main`; commits in
  logical chunks; no merge without user confirmation.
- **Verification:** end-to-end `jupyter nbconvert --execute` before commit; no
  notebook edits while an execute run is in flight.

## Out of scope

- Causal ablation / activation patching (named as the next rung, not built here).
- Arabic/Masri prompts (dialect track unaffected; this is a blog-companion
  anatomy notebook).
- Models beyond the two named (no Pythia-410m, no scale sweep).
- A blog post — this deliverable is the repo notebook only; a post may follow
  separately.
