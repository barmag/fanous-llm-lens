# Qualitative Analysis — Morphological Gold Standard

**Module:** [`src/fanous_lens/tokenizers/morphological.py`](../../src/fanous_lens/tokenizers/morphological.py)
**Tests:** [`tests/test_tokenizers/test_morphological.py`](../../tests/test_tokenizers/test_morphological.py) (16 cases)
**Date:** 2026-06-29 · **Commit:** `af63e56`
**Stack:** camel-tools 1.6.0 · disambiguator `calima-msa-r13` · scheme `d3tok`, `split=True`

> Arabic below is shown with a readable romanization and an English gloss so the
> findings are legible without Arabic proficiency. Arabic renders right-to-left; the
> character **indices** used for seam offsets are in *logical* (reading) order, so
> index 0 is the **rightmost** glyph on screen.

---

## 1. What this module is

`morphological.py` provides the **gold standard** against which every candidate
tokenizer is scored for morpheme alignment (Phase B, Task 4). It does not tokenise
for a model — it answers one question: *where are the true morpheme seams inside a
word?* A "good" tokenizer is one whose token boundaries land on these seams.

The public contract is `morpheme_boundaries(text) -> list[int]`: the surface
character offsets where one morpheme begins **inside a word**. Word boundaries
(whitespace) are deliberately excluded — every whitespace pre-tokenizer shares them,
so they carry no ranking signal. `morpheme_boundaries_with_coverage` additionally
returns `(n_words, n_skipped)` for benchmark reporting.

### Pipeline

```
text ──simple_word_tokenize──▶ words
 each word ──MorphologicalTokenizer(d3tok, split=True)──▶ marked segments  e.g. ('ال+', 'كتاب')
 strip clitic markers (+, _) ──▶ surface pieces  e.g. ('ال', 'كتاب')
 reconstruction guard: pieces concatenate back to the word?  yes ─▶ emit seams · no ─▶ skip word
 cumulative piece lengths (minus the last) + word_start ──▶ intra-word seam offsets
```

The **reconstruction guard** is the safety mechanism: d3tok normalises orthography
and clitics can assimilate in writing, so the stripped pieces sometimes do not
reconcatenate to the surface word. Rather than emit misaligned offsets, the guard
**drops that word** and counts it, so coverage loss is visible instead of silent.

---

## 2. Headline finding: the gold standard is high-quality on MSA, partial on Masri

The disambiguator is `calima-msa-r13` — a **Modern Standard Arabic** morphological
database. Empirically this produces a sharp register asymmetry that the test suite
now pins as executable assertions.

| Register | Clitics segmented | Coverage on test batch | Verdict |
|----------|-------------------|------------------------|---------|
| **MSA** (native domain) | article, conjunctions, prepositions, **future `سـ`**, all enclitic pronouns | abundant seams, **0** skips | High quality |
| **Masri** (Egyptian) | only the clitics it *shares* with MSA | sparse; Masri-specific affixes invisible | Partial |

This asymmetry is not a bug in the module — it is a property of the MSA gold
resource, and it directly shapes how Phase B results must be read (§5).

---

## 3. MSA — rich, correct segmentation

On its native register the analyzer peels the full range of clitics and stacks them
correctly. Seams are exact and surface-aligned.

| Surface | Romanization · gloss | Segmentation | Seams |
|---------|----------------------|--------------|-------|
| الكتاب على الطاولة | al-kitāb ʿalā aṭ-ṭāwila · "the book on the table" | `ال+`\|`كتاب` … `ال+`\|`طاولة` | `[2, 13]` |
| وسيذهبون إلى المدرسة | wa-sa-yaḏhabūn… · "and they will go to school" | `و+`\|`س+`\|`يذهبون` ; `ال+`\|`مدرسة` | `[1, 2, 15]` |
| سنكتب الدرس | sa-naktub… · "we will write the lesson" | `س+`\|`نكتب` ; `ال+`\|`درس` | `[1, 8]` |
| بيتهم كبير | bayt-hum… · "their house is big" | `بيت`\|`+هم` | `[3]` |
| كتبها الطالب | katab-hā… · "the student wrote it" | `كتب`\|`+ها` ; `ال+`\|`طالب` | `[3, 8]` |
| بالقلم | bi-l-qalam · "with the pen" | `ب+`\|`ال+`\|`قلم` | `[1, 3]` |
| وبكتابه | wa-bi-kitāb-i-h · "and with his book" | `و+`\|`ب+`\|`كتاب`\|`+ه` | `[1, 2, 6]` |

**Worked example — `وسيذهبون` (three stacked proclitics).** Logical indices:
`و`(0) `س`(1) `ي`(2) `ذ`(3) … The conjunction `و` occupies index 0, so the first seam
is at **1**; the future `س` occupies index 1, so the second seam is at **2**; the verb
stem `يذهبون` begins there. Result `[1, 2, …]` — read right-to-left on screen, the two
proclitics sit on the far right and peel off in order.

**Notable strength:** the **future proclitic `سـ` is segmented** (`سنكتب` → `س`\|`نكتب`).
This is the precise feature Masri's future marker fails on (§4), making it a clean
diagnostic of register.

---

## 4. Masri — shared clitics work, Egyptian-specific morphology is invisible

Masri shares much of its clitic inventory with MSA, and **those seams are found
correctly**:

| Surface | Romanization · gloss | Segmentation | Seams |
|---------|----------------------|--------------|-------|
| كتابه كبير | kitāb-u… · "his book is big" | `كتاب`\|`+ه` | `[4]` |
| وعايزين نروح | wi-ʿayzīn… · "and we want to go" | `و+`\|`عايزين` | `[1]` |
| بالعربية | bi-l-ʿarabiyya · "by car" | `ب+`\|`ال+`\|`عربية` | `[1, 3]` |
| البيت بتاعنا | il-bēt bitāʿ-na · "our house" | `ال+`\|`بيت` ; `بتاعنا` (whole) | `[2]` |

But **Masri-specific morphology is not segmented** — the MSA database has no analysis
for it, so the whole word comes back as a single token:

| Surface | Romanization · gloss | Masri feature | Segmentation | Seams |
|---------|----------------------|---------------|--------------|-------|
| بتاعنا | bitāʿ-na · "ours" | analytic possessive `بتاع` | `بتاعنا` (whole) | `[]` |
| بيكتب | bi-yiktib · "he is writing" | progressive `بـ` | `بيكتب` (whole) | `[]` |
| هيروح | ha-yrūḥ · "he will go" | future `هـ` | `هيروح` (whole) | `[]` |
| الواد ده | il-wād da · "this boy" | Egyptian lexis `واد` | `الواد` (whole) | `[]` |

These are encoded in
`test_masri_specific_morphology_undersegmented_by_msa_gold` as a **characterization
test**: it asserts `[] ` today, so adopting a Masri-aware analyzer (e.g. CALIMA-EGY)
would make it fail — the intended signal to revisit the gold standard.

---

## 5. The reconstruction guard in action

When clitics assimilate orthographically, stripped pieces no longer reconstruct the
surface word, and the guard skips rather than mis-align — in **both** registers:

| Surface | Why it fails to reconstruct | Result |
|---------|-----------------------------|--------|
| للطلاب (li-ṭ-ṭullāb, "for the students") | `لـ`+`الـ` → written `لل`; the article's alef drops | `skipped: 1` |
| ومدرستهم (wa-madrasit-hum, "and their school") | `مدرسة`+`هم` turns ة→ت under suffixation | `skipped: 1` |
| عربيتها (ʿarabiyyit-ha, "her car") | same ة→ت shift | `skipped: 1` |

This is correct, conservative behaviour: a skipped word contributes **no** gold seams
rather than wrong ones. The cost is coverage, which is why
`morpheme_boundaries_with_coverage` surfaces the skip count for the benchmark to
report (§6).

---

## 6. Scope boundary: clitics, not inflection

d3tok segments **clitics** (articles, conjunctions, prepositions, attached pronouns,
tense proclitics). It does **not** split **inflectional** affixes that are fused into
the stem:

| Surface | Segmentation | Note |
|---------|--------------|------|
| المعلمون | `ال+`\|`معلمون` | masculine plural `ون` stays in the stem |
| الطالبات | `ال+`\|`طالبات` | feminine plural `ات` stays in the stem |

So "morpheme alignment" here means **clitic-boundary alignment**, not full
morphological decomposition. Tokenizers will not be rewarded for splitting `معلم`+`ون`,
because the gold does not mark that seam. This is a deliberate, defensible scope, but
it must be stated when reporting (a tokenizer that *does* split inflection is neither
credited nor penalised).

---

## 7. Implications for the Phase B benchmark (Task 4)

1. **Report morpheme alignment stratified by register.** A single averaged F1 hides
   the asymmetry: MSA scores are measured against a rich gold, Masri scores against a
   partial one. Masri numbers are a **lower bound** on true alignment quality.
2. **Surface the skip/coverage rate per register.** If a large fraction of Masri words
   skip (assimilation + unknown lexis), the Masri alignment metric saw little signal
   and its ranking is weak. Do not hide this behind a clean-looking F1.
3. **Exclude the `morphological` tokenizer from the alignment ranking.** Its vocab is
   built from the same camel-tools segmentations as the gold — it wins by construction.
   The live comparison is bpe / unigram / wordpiece / morfessor (already a binding
   callout in the plan).
4. **Inflectional morphology is out of scope** — frame the metric as clitic-boundary
   alignment.

---

## 8. Recommendations

- **Short term:** ship as-is for MSA; treat Masri morpheme-alignment as a floor and
  always pair it with the coverage number.
- **Medium term:** evaluate **CALIMA-EGY** (the Egyptian analyzer) as the Masri gold,
  and run the gold register-matched (MSA gold for MSA text, EGY gold for Masri text).
  The characterization test in §4 is the tripwire that will flag this change.
- **Complementary signal:** because the morphological gold is partial on Masri, the
  Phase A probe results (does the embedding space *separate* linguistic features?)
  become the more trustworthy Masri-side evidence — they do not depend on the MSA gold.

---

## 9. Reproducing this analysis

```bash
uv run pytest tests/test_tokenizers/test_morphological.py -v   # 16 cases, the assertions behind every table here
uv run python -c "
from fanous_lens.tokenizers.morphological import morpheme_boundaries_with_coverage as mb
for s in ['وسيذهبون إلى المدرسة', 'البيت بتاعنا', 'للطلاب']:
    print(s, mb(s))
"
```

Every seam list in this report is asserted in the test suite, so the report and the
code cannot silently drift apart.
</content>
</invoke>
