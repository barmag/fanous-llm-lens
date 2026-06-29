# Tokenizer Comparison — five approaches on MSA vs Masri

**Driver:** [`docs/reports/compare_tokenizers.py`](compare_tokenizers.py)
**Module:** [`src/fanous_lens/tokenizers/train.py`](../../src/fanous_lens/tokenizers/train.py)
**Gold:** [`src/fanous_lens/tokenizers/morphological.py`](../../src/fanous_lens/tokenizers/morphological.py) — see [the gold-standard analysis](morphological-gold-standard-analysis.md)
**Date:** 2026-06-29 · **Commit:** `d0bc898`
**Stack:** HuggingFace `tokenizers` · `morfessor` 2.0.6 · camel-tools 1.6.0 (`calima-msa-r13`, `d3tok`)

**Setup:** trained on **6 000** sentences (3 000 MSA Wikipedia + 3 000 Egyptian tweets),
evaluated on a **disjoint** held-out set of **200 MSA + 200 Masri** sentences, `vocab_size=8 000`.
All five tokenizers realize the full 8 000-entry vocab, so fertility is compared at parity.

> Arabic renders right-to-left; seam offsets are in *logical* (reading) order, so index 0
> is the rightmost glyph. Metrics are reported **per register and never averaged** — the
> two registers are measured against golds of different completeness (see §4).

---

## 1. Results

| Approach | reg | vocab | fertility | UNK % | precision | recall | **F1** | gold seams | coverage |
|----------|-----|------:|----------:|------:|----------:|-------:|-------:|-----------:|---------:|
| bpe | MSA | 8000 | 1.68 | 0.09 | 0.302 | 0.333 | **0.317** | 3845 | 0.919 |
| bpe | Masri | 8000 | 1.81 | 0.02 | 0.178 | 0.354 | **0.237** | 1312 | 0.857 |
| unigram | MSA | 8000 | 1.92 | 0.08 | 0.306 | 0.485 | **0.376** | 3845 | 0.919 |
| unigram | Masri | 8000 | 2.07 | 0.02 | 0.200 | 0.508 | **0.287** | 1312 | 0.857 |
| wordpiece | MSA | 8000 | 1.72 | 0.09 | 0.235 | 0.280 | **0.256** | 3845 | 0.919 |
| wordpiece | Masri | 8000 | 1.88 | 0.08 | 0.146 | 0.319 | **0.200** | 1312 | 0.857 |
| **morfessor** | MSA | 8000 | 1.56 | 8.0 | 0.444 | 0.383 | **0.411** | 3845 | 0.919 |
| **morfessor** | Masri | 8000 | 1.58 | 10.6 | 0.296 | 0.399 | **0.340** | 1312 | 0.857 |
| _morphological_ ⁰ | MSA | 8000 | 1.52 | 15.6 | 1.000 | 1.000 | _1.000_ | 3845 | 0.919 |
| _morphological_ ⁰ | Masri | 8000 | 1.48 | 20.6 | 1.000 | 1.000 | _1.000_ | 1312 | 0.857 |

⁰ **Oracle — excluded from the alignment ranking.** Its vocab is built from the *same*
camel-tools segmentation as the gold, so it scores F1 = 1.0 by construction. It is shown
only as the ceiling, and remains a fair comparand on fertility / UNK (§3).

**Morpheme-alignment ranking (learned tokenizers, both registers):**
**morfessor ▸ unigram ▸ bpe ▸ wordpiece** — stable across MSA and Masri.

---

## 2. Headline: unsupervised morphology beats raw subword tokenization

Among the four *learned* tokenizers, **morfessor wins morpheme alignment in both
registers** (MSA F1 0.41, Masri 0.34) and does so with the **highest precision** — when it
places a seam, it lands on a true morpheme boundary far more often than BPE/unigram/wordpiece.
This is the expected result, and it is the point of the comparison: a tokenizer that models
morphology, even unsupervised and MDL-driven, segments Arabic closer to its morphemes than
frequency-only subword merging.

The three statistical subword tokenizers split on corpus frequency, not morphology, so they
hit morpheme seams only incidentally:

- **unigram** has the **highest recall** (MSA 0.49) but low precision (0.31) — it
  over-segments (fertility 1.9–2.1, the highest of all five), so it catches many true seams
  simply by cutting often, while also producing many spurious cuts.
- **bpe** sits in the middle on every axis.
- **wordpiece** is weakest on alignment in both registers (MSA F1 0.26, Masri 0.20).

---

## 3. Read fertility and UNK together

Fertility (tokens/word) is only meaningful alongside the UNK rate, because a tokenizer that
emits `[UNK]` for a whole word looks artificially efficient:

- **Subword tokenizers** have near-zero UNK (≤0.1 %) — byte/char fallback means they can
  encode anything — so their fertility is honest. unigram's low fertility-efficiency claim
  is real but bought with over-segmentation, not coverage loss.
- **morfessor** carries **8–11 % UNK**: its vocabulary is 8 000 learned morphs, and an
  out-of-vocabulary morph becomes `[UNK]`. Its fertility (≈1.56) is therefore a *slight*
  under-count of true granularity.
- **morphological** carries the **most UNK (16–21 %)** — its vocab is clitic-level surface
  pieces, many of which are rare stems. Its low fertility (1.48–1.52) must be read with that
  caveat: it is efficient partly because it drops information into `[UNK]`. This is a real
  cost of a closed morph vocabulary at this size, not a free lunch.

**Implication for Phase A.** If a tokenizer feeds an embeddings-only probe model, morfessor
and morphological would need a larger vocab (or subword backoff) to bring UNK down before
their alignment advantage translates into a usable model.

---

## 4. The MSA → Masri gap is real *and* partly an artifact of the gold

**Every** tokenizer scores lower on Masri than MSA. Two effects compound, and the report
must not collapse them:

1. **Genuine difficulty.** Egyptian tweets carry Masri-specific morphology (progressive
   `بـ`, future `هـ`, analytic possessive `بتاع`) and noisier orthography; the subword
   tokenizers, trained on a corpus that is half MSA, model it less well.
2. **A partial gold.** The gold runs on `calima-msa-r13`, an **MSA** database. On Masri it
   segments only the clitics it shares with MSA and **skips more words** — coverage drops
   to **0.857** (vs 0.919 for MSA) and the seam denominator falls from 3845 to 1312. So the
   Masri F1 is scored against fewer, easier seams: it is a **lower bound** on true Masri
   alignment quality, not a like-for-like number.

Because of (2), the *gap* between an approach's MSA and Masri F1 is not a clean measure of
dialect robustness. The honest cross-register claim is the **ranking**, which is identical
in both registers, not the absolute Masri scores. Closing this properly needs a Masri-aware
gold (CALIMA-EGY); the [gold-standard analysis](morphological-gold-standard-analysis.md) §8
tracks that as the next step, with a characterization test as the tripwire.

---

## 5. Caveats

- **Oracle excluded.** `morphological` is the gold's own segmenter; its F1 = 1.0 is a
  tautology and it is excluded from the alignment ranking (a binding callout in the plan).
- **Masri F1 is a floor**, scored against a partial gold (§4). Always cite it with coverage.
- **Clitic-level, not inflectional.** The gold marks clitic boundaries (article,
  conjunctions, prepositions, attached pronouns, tense proclitics), not stem-internal
  inflection (`معلم`+`ون`). "Alignment" here means clitic-boundary alignment; a tokenizer is
  neither rewarded nor penalised for splitting inflection.
- **morfessor is mildly non-deterministic.** Its batch-training tie-breaks shift F1 by
  ≈±0.01 between runs; treat its numbers as ±0.01, which does not change the ranking.
- **Scale.** 6 000 training sentences and 200 eval sentences/register — enough to saturate
  the vocab and stabilise the ranking, not a production-scale benchmark.

---

## 6. Reproduce

```bash
uv run python docs/reports/compare_tokenizers.py
```

Datasets are cached under `~/.cache/huggingface`; the run trains all five tokenizers and
prints the table in §1 plus a JSON dump. The HF trainers and camel-tools are deterministic;
morfessor varies by ≈±0.01 F1 (§5).
