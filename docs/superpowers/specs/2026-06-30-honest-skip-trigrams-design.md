# Honest skip-trigrams — BPE & Unigram experiments

**Date:** 2026-06-30
**Branch:** `honest-skip-trigrams`
**Status:** design, pending user review

## Problem

The Stage 2dash reference notebook (`stage2_dash_skip_trigram_reference.ipynb`) claims
one-layer skip-trigrams are "surprisingly expressive," but its worked example is
**degenerate**. The `skip` cell anchors the source on the OV diagonal:

```python
s = argmax(OV.diagonal()[:FREQ])   # strongest SELF-copy source
o = argmax(OV[s, :FREQ])           # what attending to s promotes  -> ~= s by construction
```

So the "triple" `[s … d → s]` is forced to be a self-copy (`output == source`). It cannot
surface a real `[A … B] → C` skip-trigram (A ≠ B ≠ C) — the linguistically interesting case
from *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021). The notebook
also omits the section's richest result: the **skip-trigram bug** (heads cannot condition
jointly on source and destination, so a head doing `keep…in→mind` and `keep…at→bay` is forced
to also raise `keep…in→bay`).

## Goals

1. Replace the degenerate example with an **honest method** that surfaces real skip-trigrams,
   verified on held-out text, with empty results reported as findings.
2. Organize discovery around **Arabic-relevant categories** (the paper's code/URL/LaTeX
   categories do not transfer).
3. Compare **BPE vs Unigram** tokenizers (trained on the same corpus) for skip-trigram
   legibility — requires a Unigram model retrain (the model is welded to its tokenizer's
   vocab; embeddings/unembeddings are indexed by token id).

## Non-goals

- No HF Hub push of the new Unigram checkpoint unless the user later asks (local-first).
- No change to the trained BPE model or its checkpoint — only the notebook around it.
- No new model architecture; same 1-layer attn-only, LN-free, shortformer config.

## Key facts established

- BPE checkpoint: `checkpoints/stage2dash/` — 1L, 8 heads, d_model=512, vocab=12k,
  trained on ~338M Arabic tokens, ~62 min on the iGPU (`metrics.json`).
- `corpus.py::train_tokenizer` currently trains BPE only (NFKC normalizer + Whitespace
  pre-tokenizer). A Unigram path must be added.
- The model is **tokenizer-welded**: a Unigram experiment = a full retrain (~62 min).

## Files

| File | Change |
|---|---|
| `notebooks/education/stage2_dash_skip_trigram_reference.ipynb` | Upgraded in place → tokenizer-agnostic answer key with the honest method. Reads checkpoint dir from a variable. |
| `notebooks/education/stage2_dash_skip_trigram_bpe_experiment.ipynb` | New scaffold; points at `checkpoints/stage2dash/`. |
| `notebooks/education/stage2_dash_skip_trigram_unigram_experiment.ipynb` | New scaffold; points at `checkpoints/stage2dash_unigram/`. |
| `notebooks/education/corpus.py` | Add `kind: "bpe" \| "unigram"` to `train_tokenizer`; Unigram path uses `models.Unigram` + `UnigramTrainer`, same NFKC normalizer + Whitespace pre-tokenizer, vocab 12k. |
| `notebooks/education/train_stage2dash.py` | Add `--tokenizer {bpe,unigram}`; Unigram writes to `checkpoints/stage2dash_unigram/`. Same seed/corpus/config. |

## Method (reference notebook content)

Replaces the single degenerate `skip` cell with a sequence:

1. **All-8-heads attention diagnostic.** Small-multiples heatmap on a real Arabic sentence.
   Classify each head: positional (BOS / prev-token) vs content-based long-range. This is the
   reality check — if heads are mostly positional, legible non-copy skip-trigrams may not
   exist and the bug demo becomes the headline (stated honestly).

2. **Category-driven candidate generation (the ~100-per-category pool).**
   - Score every candidate triple `(source, dest, output)` by a **composite**:
     `OV[source, output]` (copy/promote strength, `output ≠ source` for non-baseline)
     × `QK[dest, source]` (routing strength) × a frequency guard (restrict to frequent
     tokens so rare high-norm embeddings don't dominate).
   - **Bucketing = hybrid:**
     - **Seeded pools** for the four categories: for each category's seed token set, pull the
       top-N (target 100) triples touching those seeds.
     - **One unsupervised "what else is in here" table:** top triples model-wide, unlabelled,
       to catch categories we didn't anticipate.
   - Each pool is a ranked table (`source · dest · output · OV_score · QK_score · freq`),
     rendered RTL. Full pool saved to a CSV artifact; top ~15 shown inline.
   - **Honesty guardrail:** target is *up to* 100. Plot the score distribution per category
     and draw a **noise cutoff**; report the real count above the line. Do not pad to 100.
     Some categories (esp. Masri contrast) will have far fewer real candidates — that is a
     finding about data/representation, not a failure.

3. **Held-out verification (top-K only).** For the top ~20 of each pool, construct a real
   sequence containing `[source … dest]` and confirm the forward pass actually raises
   `P(output)` vs the bigram-only baseline. A high matrix entry that does **not** move the
   real distribution is reported as an **artifact, not a finding**. Only verified triples are
   eligible to be featured.

4. **Feature the most representative.** From the verified top of each table, pick a small N to
   highlight. In the reference (answer key) these are frozen after an actual run; in the
   experiment scaffolds the learner picks. Always show "chose N of <real count>".

5. **The skip-trigram bug.**
   - *Promised (structural):* pick a source, show its top-2 OV outputs are both promoted
     regardless of which destination attends → joint conditioning is structurally impossible.
   - *Attempted (linguistic):* find an Arabic idiom pair analogous to keep…in-mind /
     keep…at-bay and show the forced cross-term error. May fail; report honestly.

6. **Keep** the existing quantitative cell (skip-trigram reshape / top-1-changed %) — it was
   never broken; only the qualitative example was.

## Categories & draft seed lexicons (USER TO REVIEW — Arabic judgement)

Self-copy / induction-lite baseline is always included as the control.

| Category | Draft seed tokens (edit me) |
|---|---|
| MSA fixed expressions | الرغم · بالإضافة · حين · بسبب · أجل · الرغم من |
| Religious / formulaic | الله · صلى · رضي · شاء · عليه · سبحانه |
| Definite-article / morphology | ال (clitic) · أل- prefixed nouns |
| MSA-vs-Masri contrast | اللي · عايز · دلوقتي · إزاي (Masri) vs الذي · يريد · الآن (MSA) |

Seeds are starting points; the search expands from them by score. The user (Arabic reader)
finalizes these before the search code is locked.

## BPE vs Unigram — controlled comparison

To isolate the **algorithm**, both tokenizers share: vocab=12k, NFKC normalizer, Whitespace
pre-tokenizer, same corpus, same seed. Only `models.BPE` vs `models.Unigram` differs.

- Decision: force Whitespace on both (clean algorithmic comparison). Unigram's natural pairing
  is Metaspace (SentencePiece-style); we call this out in the notebook as the road-not-taken —
  itself a tokenizer-aware teaching point.
- Unigram retrain: ~62 min on the iGPU, run via
  `train_stage2dash.py --tokenizer unigram`. Checkpoint → `checkpoints/stage2dash_unigram/`,
  local-first.

## Pedagogy / structure conventions (per repo memory)

- Reference = worked answer key; experiment = scaffold the learner fills in.
- RTL display for Arabic blocks; bilingual (Arabic + English) markdown as in the existing
  notebook.
- Notebooks run end-to-end in <10 min on the iGPU (the interpretability is fast; training is
  the offline script).
- Clear outputs before commit.

## Honesty guardrails (summary)

1. No fixed example count; we feature what survives held-out verification.
2. Empty / thin categories are reported with their real above-noise count, not padded.
3. If heads are mostly positional, the bug demo becomes the headline and we say so.
4. Featured triples must move the real next-token distribution, not just the matrix.

## Open items decided (unless user objects)

- Unigram vocab matched to 12k.
- Whitespace pre-tokenizer forced on both tokenizers.
- Order: BPE method first (free), then Unigram retrain + same analysis.

## Verification plan

- `corpus.py` / `train_stage2dash.py`: smoke via `--calibrate` (short run) before the full
  Unigram retrain; confirm a Unigram `tokenizer.json` + `model.pt` + config are written.
- Notebooks: run end-to-end on the real checkpoint(s); `verify_notebooks.py` if it covers
  these; confirm decomposition residual ~0 still holds (sanity that the loaded model is intact).
