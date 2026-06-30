# Handover — semantic-tokenization probe line

**Last updated:** 2026-06-30 · **Branch state at handover:** `depth-probes` merged to `main`.

This note is the entry point for the next session working on the tokenizer-interpretability
probe line. It assumes the three reports are the source of truth:
[phase-a-probes.md](phase-a-probes.md) (zero-layer verdict),
[phase-a-depth.md](phase-a-depth.md) (depth null),
[tokenization-lit-note.md](tokenization-lit-note.md) (literature framing).

---

## Where we are (one paragraph)

We compared 5 Arabic tokenizers (bpe, unigram, wordpiece, morfessor, morphological) for
**interpretability fitness** via linear probes on camel-tools features (definiteness, number),
MSA vs Masri, type-split, controls. Two settled results: **(1)** the zero-layer pooled-mean probe
is *insensitive to training* — it scores the **tokenization**, not the learned representation
(every trained−untrained Δ within ±0.014). **(2)** Adding 2 transformer layers does **not**
reconstruct morphology the tokenizer fused: `number` stays an embedding-level property the layers
can't add to, and the Phase-A ranking (unigram ▸ subwords ▸ morphological) survives depth
unchanged. The only real training increment is **definiteness alignment at the embedding** for
tokenizers that spread the article across many tokens (bpe/wordpiece). Headline: **morpheme
alignment is not a universal interpretability win — it is feature-specific, fixed by where the
tokenizer cuts.** Independently corroborated by the literature (Why-morph-complex; Morphemes-
without-borders; Unigram-dominance).

---

## The open confound that gates the next result

**Byte-premium / fertility.** Every model trained on the *same 30k sentences* but, because
tokenizers differ in fertility, **not the same number of tokens** → different effective training
signal per tokenizer. So our cross-tokenizer AUC gaps **conflate feature-exposure with
effective-data**. We have flagged this in all three reports but **not controlled it**. This is the
single most important thing to fix before any AUC ranking is read as a pure tokenization property.

---

## "Do we need more Masri data?" — the verdict from this session

Asked and answered (2026-06-30). The phrase bundles three hypotheses with different verdicts:

1. **Lift the Masri-below-MSA probe gap** → *weak.* The probe is insensitive to training, so a
   data-quantity lever can't move it. The Masri gap is more plausibly **Masri fertility** (MSA-
   dominated tokenizers segment Masri into more tokens) + **partial Masri gold coverage**.
2. **Fix the `number` fused-morphology null** → *no.* Structural: the tokenizer fused `ون`/`ات`
   into lexically-unique stems and the type-split blocks memorization. No data volume gives the
   feature a shared carrier — the tokenizer must cut differently (Inequities: segmentation is
   design, not corpus size).
3. **Genuinely better downstream models** → *yes, but orthogonal.* Literature-supported (Why-
   morph-complex: byte-adjusted data dominates downstream) but our probe apparatus doesn't test
   downstream quality.

**The legitimate, testable core of the intuition:** Masri *composition* changes tokenizer
fertility. More Masri in the tokenizer's training → more Masri merges → lower Masri fertility →
less byte-premium penalty on the Masri leg. This is **not "more data" but "data composition,"** and
it folds directly into the fertility-matched rerun below.

---

## Next steps (priority order)

1. **Fertility-matched rerun (TOP).** Re-run Phase A/depth equalizing training *tokens* (not
   sentences) per tokenizer. Separately log **Masri-vs-MSA fertility** per tokenizer. This
   decomposes the Masri gap into *effective-data (token budget)* vs *segmentation (where it cuts)*
   — the one clean data-vs-algorithm split nobody has done for Egyptian Arabic. Strips the
   byte-premium confound; tests whether the unigram>morphological `number` gap survives.
2. **Intervention test** for the §4 mechanism: force a shared plural token into `morphological`,
   re-probe `number`; and try a **last-token (not mean-pool) readout** to de-risk the §2 number
   erosion (the mean-pool may dilute a post-mixing signal).
3. **Non-concatenative target.** Add a root-and-pattern feature (broken plurals, root identity) —
   the Arabic-specific failure mode BPE is structurally unsuited to, untouched by our concatenative
   probes. Plus negation / tense / **dialect** for a fuller sweep.
4. **Masri-aware gold (CALIMA-EGY)** to fix partial Masri coverage (tripwire:
   [morphological-gold-standard-analysis.md](morphological-gold-standard-analysis.md) §4).

---

## Env / repro gotcha

GPU needs the `rocm` extra — plain `uv run` reverts torch to a non-ROCm build:

```bash
env HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev --extra tokenizers \
    python experiments/embedding-probes/run_probes_depth.py
```

camel-tools lives in the `tokenizers` extra; morfessor declared there too. Checkpoints (~50MB)
are gitignored. `gfx1151→gfx1100` override is the load-bearing line. `origin/main` is behind
local `main` (not pushed) — confirm with the user before pushing.
