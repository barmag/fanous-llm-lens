# Tokenizer Comparison — five approaches on MSA vs Masri

**Driver:** [`docs/reports/compare_tokenizers.py`](compare_tokenizers.py)
**Metrics:** [`src/fanous_lens/tokenizers/evaluate.py`](../../src/fanous_lens/tokenizers/evaluate.py)
**Gold:** [`src/fanous_lens/tokenizers/morphological.py`](../../src/fanous_lens/tokenizers/morphological.py) — see [the gold-standard analysis](morphological-gold-standard-analysis.md)
**Date:** 2026-06-29 · **Commit:** `84eff21`
**Stack:** HuggingFace `tokenizers` · `morfessor` 2.0.6 · camel-tools 1.6.0 (`calima-msa-r13`, `d3tok`)

> **Supersedes an earlier draft of this report** that ranked the tokenizers by a
> morpheme-alignment **F1**. That F1 was unsound (§2): against an incomplete gold you
> cannot measure precision, so the F1 rewarded agreement with the gold's blind spots and the
> `morphological` tokenizer scored 1.0 by tautology. This version reports only what the gold
> can honestly support — **clitic recall paired with fertility** — plus a **gold-free
> consistency** signal, and reframes morpheme alignment as a *hypothesis to be tested in
> Phase A*, not a verdict.

**Setup:** trained on **6 000** sentences (3 000 MSA Wikipedia + 3 000 Egyptian tweets),
evaluated on a **disjoint** held-out set of **200 MSA + 200 Masri** sentences, `vocab_size=8 000`.
All five tokenizers realize the full 8 000-entry vocab, so fertility is compared at parity.

> Arabic renders right-to-left; offsets are in *logical* (reading) order, so index 0 is the
> rightmost glyph. Metrics are reported **per register and never averaged** — the two
> registers are measured against golds of different completeness (§5).

---

## 1. What "fitness for interpretability" means here

The project's goal is mechanistic interpretability of MSA-vs-Masri structure in small
models. So the question is **not** "does this tokenizer match a linguist's segmentation" but:

> Under this tokenization, can linguistic structure be **recovered**, **localized** to few
> stable tokens, and **compared across registers** — *inside the model*?

That terminal question is answered by **probing a trained model** (Phase A), not by comparing
token boundaries to a reference. Boundary comparison is a cheap *proxy*, and a leaky one —
which is the whole point of §2. This report is the **cheap CPU tier** that runs before the
probe; it deliberately avoids any metric that pretends to be the verdict.

---

## 2. Why there is no precision or F1 here

The gold ([analysis](morphological-gold-standard-analysis.md)) runs on an **MSA** database
and marks **clitic** boundaries only. It does **not** mark inflection — it leaves `يذهبون`
whole rather than `يذهب`+`ون` — and it is weak on Masri.

Against such an **incomplete** gold, **precision is unmeasurable**: a boundary a tokenizer
places where the gold is silent is *indistinguishable from a true boundary the gold simply
missed*. `يذهب·ون` is exactly that — a tokenizer that correctly splits the plural suffix is
scored as a **false positive**. So:

- any **F1** built on that precision rewards *agreement with the gold's blind spots*, and is
  **anti-aligned with quality** precisely in the morphologically-rich cases interpretability
  cares about;
- `morphological`'s perfect score is a **tautology** — its vocab *is* the gold's segmenter,
  so it agrees with the gold by construction. It is the gold in a mirror, not a ceiling.

The one directionally-honest boundary number is **recall**: *of the clitic boundaries the
gold is sure exist, how many did the tokenizer place?* It never penalises a tokenizer for
splitting inflection or Masri morphology the gold cannot see. But **recall alone is gamed by
over-segmentation** (a char-level splitter scores 1.0 by cutting everywhere), so it is honest
**only read next to fertility**.

---

## 3. Results

**Clitic recall — read across the row with fertility, never alone.**

| Approach | reg | vocab | fertility | UNK % | **clitic recall** | beyond-gold % | gold seams | coverage |
|----------|-----|------:|----------:|------:|------------------:|--------------:|-----------:|---------:|
| bpe | MSA | 8000 | 1.68 | 0.1 | 0.333 | 70 | 3845 | 0.919 |
| bpe | Masri | 8000 | 1.81 | 0.0 | 0.354 | 82 | 1312 | 0.857 |
| unigram | MSA | 8000 | 1.92 | 0.1 | **0.485** | 69 | 3845 | 0.919 |
| unigram | Masri | 8000 | 2.07 | 0.0 | **0.508** | 80 | 1312 | 0.857 |
| wordpiece | MSA | 8000 | 1.72 | 0.1 | 0.280 | 76 | 3845 | 0.919 |
| wordpiece | Masri | 8000 | 1.88 | 0.1 | 0.319 | 85 | 1312 | 0.857 |
| morfessor | MSA | 8000 | 1.56 | 8.0 | 0.380 | 56 | 3845 | 0.919 |
| morfessor | Masri | 8000 | 1.58 | 10.6 | 0.406 | 70 | 1312 | 0.857 |
| _morphological_ ⁰ | MSA | 8000 | 1.52 | 15.6 | _1.000_ | _0_ | 3845 | 0.919 |
| _morphological_ ⁰ | Masri | 8000 | 1.48 | 20.6 | _1.000_ | _0_ | 1312 | 0.857 |

⁰ **Tautology, not a result.** `morphological` cuts only where the gold cuts, so recall =
1.0 and beyond-gold = 0 by construction. Excluded from the live comparison; shown only to
make the tautology explicit. `beyond-gold %` = share of a tokenizer's intra-word cuts that
land where the gold is silent — **descriptive, not error** (much of it is inflection/dialect
splitting the gold can't judge).

**Consistency (gold-free) — does a fixed morpheme tokenize the same across host words?**

| Approach | top-share ↑ | entropy (bits) ↓ |
|----------|------------:|-----------------:|
| unigram | 0.663 | 1.10 |
| morfessor | 0.503 | 1.61 |
| bpe | 0.323 | 2.04 |
| wordpiece | 0.323 | 2.04 |
| _morphological_ ⁰ | _1.000_ | _0.00_ |

(Five shared morphemes: `ال`, `و`, `ب`, `كتاب`, `بيت`, across curated host words. `top-share`
= fraction of hosts where the morpheme takes its single most-common tokenization; `entropy` =
spread of its tokenizations. Stable = localizable feature. `morphological` is 1.0/0.0 because
it is morpheme-based — a real property, but expected, so it is bracketed too.)

---

## 4. Reading the results: there is no single winner

The two honest diagnostics **point in different directions**, and that is the finding:

- **Recall at fertility.** unigram has the highest raw recall (0.49 MSA) but pays the most
  fertility (1.92) and cuts 69 % beyond the gold — it catches clitics largely by cutting a
  lot. morfessor recovers nearly as many clitics (0.38) at the **lowest fertility** (1.56)
  and the **lowest beyond-gold rate** (56 %), so its cuts coincide with real clitics more
  often per cut. bpe is middling; wordpiece is weakest. On *efficiency of clitic recovery*,
  morfessor looks best of the learned four.
- **Consistency.** But unigram is the **most stable** learned tokenizer (top-share 0.66,
  entropy 1.10) — a shared morpheme lands on the same token more often than under morfessor
  (0.50 / 1.61) or bpe/wordpiece (0.32 / 2.04). On *localizability*, unigram looks best.

So **recall-at-fertility favours morfessor; consistency favours unigram.** Nothing here
adjudicates which property matters more for interpretability — that is precisely the
hypothesis the Phase A probe exists to settle (§6). Reporting a single ranking would
manufacture a verdict the evidence does not support.

What *is* robust: **bpe and wordpiece are dominated** — lower recall-per-fertility *and*
lower consistency than the alternatives, in both registers.

---

## 5. The MSA → Masri gap is real *and* partly an artifact of the gold

Every tokenizer recovers fewer Masri clitics, and cuts more **beyond** the gold on Masri
(80–85 % vs 69–76 %). Two effects compound — keep them separate:

1. **Genuine difficulty.** Egyptian tweets carry Masri-specific morphology (progressive `بـ`,
   future `هـ`, possessive `بتاع`) and noisier orthography.
2. **A partial gold.** The gold is MSA-only; on Masri it covers just **0.857** of words (vs
   0.919) and exposes 1312 seams (vs 3845). So Masri recall is scored against fewer, easier
   boundaries — it is a **lower bound**, not a like-for-like number, and the high beyond-gold
   rate means the tokenizers are doing a lot the gold simply cannot judge (the appendix shows
   them splitting `بيكتب`/`هيروح`/`بتاعنا` — arguably correctly — for zero credit).

The honest cross-register statement is the **dominance of bpe/wordpiece**, which holds in
both registers, not any absolute Masri number. A Masri-aware gold (CALIMA-EGY) is the fix;
the [gold analysis §8](morphological-gold-standard-analysis.md) tracks it.

---

## 6. The verdict comes from Phase A, not from this table

This report is **Tier 1**: cheap, CPU-only, honest about its own limits. It can *rule things
out* (bpe/wordpiece dominated) and surface *tensions* (recall vs consistency), but it cannot
declare a fitness winner, because **morpheme alignment is a hypothesis, not a definition** —
BPE may never split `ال` yet leave a model perfectly probeable for definiteness.

**Tier 2 (Phase A, GPU)** settles it: train an embeddings-only model per tokenization, then
probe for definiteness / tense / number / dialect and measure how *localized* each feature is
(how few tokens carry it). The tokenizer whose representations make linguistic features most
recoverable and most localized is the most interpretability-fit — *regardless* of its recall
or consistency here. That step needs explicit GPU go-ahead (it crashes the window manager on
this box) and ~5 model trainings, which is why it is staged after this cheap tier.

---

## 7. Caveats

- **No precision / F1** — unmeasurable against an incomplete gold (§2). Do not reintroduce it.
- **Recall is meaningless without fertility** — always read the two together (§2).
- **`morphological` is tautological** on recall *and* (by construction) consistency — bracketed
  in both tables, excluded from the live comparison.
- **Masri recall is a floor**, scored against a partial gold (§5); always cite it with coverage.
- **Clitic-level, not inflectional.** "Recall" is clitic-boundary recall; the gold does not
  mark stem-internal inflection, so neither does this metric.
- **morfessor is mildly non-deterministic** (±0.01 on recall between runs); does not change §4.
- **Consistency uses a curated morpheme set**, so it is indicative, not exhaustive.
- **Scale.** 6 000 train / 200 eval-per-register — enough to saturate vocab and stabilise the
  *dominance* claim, not a production benchmark.

---

## 8. Reproduce

```bash
uv run python docs/reports/compare_tokenizers.py
```

Datasets are cached under `~/.cache/huggingface`; the run trains all five tokenizers and
prints the two tables in §3, the appendix, and a JSON dump. HF trainers and camel-tools are
deterministic; morfessor varies by ≈±0.01 (§7).

---

## Appendix — representative segmentations

Each cell is one tokenizer's split of the word, pieces joined by `·`. Same trained tokenizers
as §3; the driver prints this verbatim under "APPENDIX". Pieces read right-to-left. `gold` is
the camel-tools clitic target — a whole word (no `·`) means the gold marks no intra-word seam.

**MSA — the gold's native register**

| word · gloss | gold | bpe | unigram | wordpiece | morfessor | morphological |
|---|---|---|---|---|---|---|
| وسيذهبون · and they will go | و·س·يذهبون | وسي·ذهب·ون | و·سي·ذه·ب·ون | وسي·ذهب·ون | وس·يذهب·ون | و·س·يذهبون |
| بالقلم · with the pen | ب·ال·قلم | بالق·لم | بال·ق·لم | بالق·لم | بالقلم | ب·ال·قلم |
| كتبها · he wrote it | كتب·ها | كتبها | كتب·ها | كتب·ها | كتبها | كتب·ها |
| المدرسة · the school | ال·مدرسة | المدرسة | المدرس·ة | المدرسة | المدرسة | ال·مدرسة |

**Masri — Egyptian**

| word · gloss | gold | bpe | unigram | wordpiece | morfessor | morphological |
|---|---|---|---|---|---|---|
| كتابه · his book (shared enclitic) | كتاب·ه | كتابه | كتابه | كتابه | كتابه | كتاب·ه |
| بالعربية · by car (shared proclitics) | ب·ال·عربية | ب·العربية | ب·العربية | بالع·ربية | ب·العربية | ب·ال·عربية |
| بيكتب · he is writing (progressive بـ) | بيكتب | بي·كتب | ب·يكتب | بيك·ت·ب | بيكتب | بيكتب |
| هيروح · he will go (future هـ) | هيروح | هير·وح | ه·يروح | هير·وح | ه·يروح | هيروح |
| بتاعنا · ours (analytic possessive) | بتاعنا | بتاع·نا | بتاع·نا | بتاع·نا | بتاعنا | بتاعنا |

**What these show (and how they back §3–§5):**

1. **`morphological` reproduces the gold exactly** on every MSA row — the visual form of its
   recall-1.0 / consistency-0-entropy tautology.
2. **morfessor recovers clitics efficiently but not exhaustively.** It keeps short clitic
   words whole (`بالقلم`, `المدرسة`, `كتبها`) yet on the long `وسيذهبون` it lands closest of
   the learned four (`وس·يذهب·ون`) — the low-fertility, low-beyond-gold profile from §3/§4.
3. **Partial alignment on stacked proclitics.** On `بالعربية` (gold `ب·ال·عربية`),
   bpe/unigram/morfessor peel `ب` but miss `ال`; only `morphological` recovers both — the
   reason learned recall sits well below 1.0.
4. **unigram visibly over-segments** (`و·سي·ذه·ب·ون`, `بيك·ت·ب`) — the high-fertility,
   high-recall, 69 % beyond-gold profile.
5. **The partial-gold penalty, made visible (the §5 point).** On `بيكتب`, `هيروح`, `بتاعنا`
   the gold marks **no seam** — the MSA database cannot analyse the Masri progressive `بـ`,
   future `هـ`, or possessive `بتاع`. Yet bpe splits `بي·كتب`, morfessor splits `ه·يروح`, and
   three tokenizers split `بتاع·نا` — arguably the *correct* Masri cuts. They earn **zero
   credit** and inflate the beyond-gold rate. This is the visual proof that Masri recall is a
   **lower bound**, and the concrete reason a precision-style metric would have been actively
   misleading.
