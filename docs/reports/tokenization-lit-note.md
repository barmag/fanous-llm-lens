# Lit-note — our tokenizer findings against the published literature

**Date:** 2026-06-30 · **Companion to:** [phase-a-probes.md](phase-a-probes.md),
[phase-a-depth.md](phase-a-depth.md), [tokenizer-comparison.md](tokenizer-comparison.md)

Purpose: situate our small-model probe results (Phase A zero-layer, Phase A depth) against the
tokenization literature, mark where we are **corroborated**, and import one **caveat we should
have stated** — the byte-premium / fertility confound.

---

## 1. The question we keep asking: "does BPE do worse on Arabic, and is it data or algorithm?"

The literature answers **"yes, worse"** — but the data-vs-algorithm verdict splits by *what metric
you mean by "worse,"* and by *which "data."*

| "perform worse" = | dominant cause | source |
|---|---|---|
| **compression / fertility** (tokens per Arabic word) | intrinsic script + morphology **×** tokenizer **design** (pre-tokenization, vocab allocation) — **not** the tokenizer's corpus size | Crosslingual Tokenizer Inequities (2025) |
| **downstream LM quality** (loss, task acc.) | **data** — the *model's* effective, byte-premium-adjusted pretraining amount | Why-morph-complex (2024) |
| **algorithm head-to-head** (BPE vs Unigram) | **algorithm** — BPE's greedy merge heuristic; rescued by morphological pre-segmentation | Unigram-dominance (2025) |

The two "data" claims are not in conflict: **Inequities** holds the *tokenizer* corpus equal and
still sees fertility vary (so fertility ≠ tokenizer-corpus-size); **Why-morph-complex** says the
*model's* pretraining quantity, once byte-adjusted, closes the downstream gap.

### Key specifics worth keeping
- **Inequities (arXiv 2510.21909):** monolingual tokenizers trained on **identical 300 MB**, 7 vocab
  sizes, BPE *and* Unigram, **97 languages incl. MSA** — token premium still varies widely. Regression
  on corpus-token-count: data-similarity R²=0.239, mean-token-length R²=0.168, whitespace-proportion
  R²=0.157. Whitespace pre-tokenization and per-language vocab allocation drive much of the gap.
  **Crucial limitation: they trained no language models** — the claim is about *compression*, not model
  performance.
- **Why-morph-complex (arXiv 2411.14198):** "no language is harder… on the basis of its morphological
  typology. Differences… can be attributed to disparities in dataset size." Byte-premium-scaled data
  erases the gap; **morphological alignment did not correlate with performance.** (Turkish-centric;
  not Arabic.)
- **Unigram-dominance (arXiv 2508.08424):** "Unigram-based tokenizers consistently outperforming BPE";
  morphological alignment is "moderate, positive… secondary to the tokenizer algorithm"; morphological
  **pre-segmentation boosts BPE but not Unigram** → the merge heuristic is the weak link. (Telugu/Hindi/
  English; not Arabic.)
- **Morphemes-without-borders (arXiv 2603.15773, Arabic):** "tokenizer morphological alignment is **not
  necessary nor sufficient** for morphological generation" — alignment quality and model competence are
  decoupled.

---

## 2. Where our results are corroborated

- **"Alignment is not a universal interpretability win"** (our Phase A headline) is the small-model,
  intrinsic-probe echo of **Why-morph-complex** ("alignment doesn't correlate with performance") and
  **Morphemes-without-borders** ("neither necessary nor sufficient"). Three methods, three scales, same
  shape — this is a robust convergence, not a fluke of our setup.
- **"unigram is the best all-rounder"** (Phase A) aligns with **Unigram-dominance** (Unigram > BPE).
  We reached it from probe separability; they from downstream tasks. Independent corroboration.
- **Our depth null** ("2 layers don't reconstruct fused morphology; the feature must be present in the
  tokenization") is consistent with the literature's split: the *downstream* gap is closable with data,
  but the *feature presence* is a property of the tokenization that learning does not synthesize from
  nothing.

## 3. The caveat we should import — byte premium / fertility confound

Our probes train **one model per tokenizer on "the same 30 k sentences."** The literature's central
lesson is that **"same corpus" ≠ "same training signal"** across tokenizers:

- Different tokenizers have different **fertility**, so the same sentences yield **different token
  counts** → different effective sequence lengths and different amounts of next-token signal per step.
  Our `max_tokens` cap + fixed steps only partly control this; per-step *content* still differs.
- Therefore a tokenizer's probe AUC conflates **"is the feature exposed in the inventory"** (what we
  intend to measure) with **"how much effective signal that tokenizer's fertility afforded."** At our
  scale the per-tokenizer AUC gaps are large vs seed noise, so the ranking likely survives — but we have
  **not** controlled byte premium, and should say so.
- We already do the right thing in one place — **we refuse to compare LM loss across tokenizers**
  (different vocabularies/sequence stats; [phase-a-probes.md](phase-a-probes.md) §5). The byte-premium
  point generalizes that caution to the *training-signal* axis, not just the loss axis.

**Concrete text to add** to the caveats of [phase-a-probes.md](phase-a-probes.md) and
[phase-a-depth.md](phase-a-depth.md):

> *Byte-premium confound (untested).* Each model is trained on the same sentences but, because
> tokenizers differ in fertility, **not** the same number of tokens or the same per-step signal. Cross-
> tokenizer AUC gaps therefore conflate feature-exposure with effective-data effects. The published
> decompositions (Inequities 2025; Why-morph-complex 2024) find data/fertility dominate at scale, so a
> fertility-matched rerun (equalize *tokens*, not sentences) is needed before the ranking is read as a
> pure tokenization property.

## 4. The gap fanous-llm-lens is positioned to fill

The cleanest causal data-vs-algorithm decompositions are **not Arabic** (Turkish; 97-language sweeps),
and the Arabic-specific papers measure downstream tasks / generation, not residual-stream features. **No
one has run a seed-controlled, register-split (MSA vs Masri), feature-level probe that isolates
data-vs-algorithm for Egyptian Arabic.** That is exactly our apparatus. A credible contribution:

1. **Fertility-matched probe** — re-run Phase A/depth equalizing *training tokens* (not sentences) per
   tokenizer, to strip the byte-premium confound and test whether the unigram>morphological *number*
   gap survives.
2. **Non-concatenative target** — add a root-and-pattern feature (broken plurals, root identity), the
   Arabic-specific failure mode BPE is structurally unsuited to and our concatenative probes don't yet
   touch.
3. **Masri vs MSA as the controlled axis** — the dialect dimension is under-studied and is the project's
   stated north star.

---

## Sources

- Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages*, NeurIPS 2023 — arXiv 2305.15425
- *Why do language models perform worse for morphologically complex languages?* (2024) — arXiv 2411.14198
- *Explaining and Mitigating Crosslingual Tokenizer Inequities* (2025) — arXiv 2510.21909
- *Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE…* (2025) — arXiv 2508.08424
- *Morphemes Without Borders: Root-Pattern Morphology in Arabic Tokenizers and LLMs* (2026) — arXiv 2603.15773
- *Exploring Tokenization Strategies and Vocabulary Sizes for Enhanced Arabic LMs* (2024) — arXiv 2403.11130
- Alyafeai et al., *Evaluating Various Tokenizers for Arabic Text Classification* (2021) — arXiv 2106.07540
