# MSA / Masri minimal pairs — v1.1

Human-readable mirror of [`msa-masri-pairs-v1.json`](msa-masri-pairs-v1.json). Edits go to the JSON first; this file is regenerated.

**Languages:** Modern Standard Arabic (MSA / Fusha, الفصحى), Egyptian Arabic (Masri, مصري), and English (control baseline).  
**Triples:** 30, hand-crafted by an Egyptian Arabic native speaker. No machine translation.  
**Use:** controlled inputs for tokenizer comparison and residual-stream probing experiments.

> Arabic renders right-to-left; if columns look misaligned in your Markdown viewer, open the raw file or the JSON.

## Construction notes

- Every pair was hand-crafted by an Egyptian Arabic native speaker. No machine translation.
- Pairs aim to be natural in their register — not word-by-word transpositions. A Masri speaker should not feel a Masri line is 'Fusha with one word swapped'.
- Where the MSA form admits diacritics, they are omitted (no tashkeel). Both forms are stored as a tokenizer would actually receive them from a web-scraped corpus.
- Pair count: 30. Categories: pronoun, negation, future, verb, question, demonstrative, lexical, aspect.
- v1.1 (2026-05-02): added an English baseline per pair so tokenizer comparisons can disentangle 'Arabic is hard' from 'Masri is hard for an MSA-trained vocab'.

## Triples

| id | category | English (baseline) | MSA (الفصحى) | Masri (مصري) |
|----|----------|--------------------|--------------|--------------|
| p01 | pronoun | We are students. | <span dir="rtl">نحن طلاب</span> | <span dir="rtl">احنا طلبة</span> |
| p02 | pronoun | You are welcome here. | <span dir="rtl">أنتم مرحب بكم هنا</span> | <span dir="rtl">انتو مرحب بيكم هنا</span> |
| p03 | negation | I don't know. | <span dir="rtl">لا أعرف</span> | <span dir="rtl">مش عارف</span> |
| p04 | negation | He didn't come. | <span dir="rtl">لم يأتِ</span> | <span dir="rtl">ما جاش</span> |
| p05 | negation | We won't go. | <span dir="rtl">لن نذهب</span> | <span dir="rtl">مش هنروح</span> |
| p06 | negation | This is not true. | <span dir="rtl">هذا ليس صحيحا</span> | <span dir="rtl">ده مش صح</span> |
| p07 | future | I will travel tomorrow. | <span dir="rtl">سأسافر غدا</span> | <span dir="rtl">هاسافر بكرة</span> |
| p08 | future | They will arrive at noon. | <span dir="rtl">سيصلون عند الظهر</span> | <span dir="rtl">هيوصلوا الضهر</span> |
| p09 | verb | I want some water. | <span dir="rtl">أريد بعض الماء</span> | <span dir="rtl">عايز شوية ميه</span> |
| p10 | verb | What do you want? | <span dir="rtl">ماذا تريد؟</span> | <span dir="rtl">عايز إيه؟</span> |
| p11 | verb | What are you doing now? | <span dir="rtl">ماذا تفعل الآن؟</span> | <span dir="rtl">بتعمل إيه دلوقتي؟</span> |
| p12 | verb | I saw him yesterday. | <span dir="rtl">رأيته أمس</span> | <span dir="rtl">شفته إمبارح</span> |
| p13 | verb | Where are you going? | <span dir="rtl">إلى أين تذهب؟</span> | <span dir="rtl">رايح فين؟</span> |
| p14 | verb | Come here, please. | <span dir="rtl">تعال هنا من فضلك</span> | <span dir="rtl">تعالى هنا لو سمحت</span> |
| p15 | verb | Give me the book. | <span dir="rtl">أعطني الكتاب</span> | <span dir="rtl">إديني الكتاب</span> |
| p16 | verb | I haven't eaten anything today. | <span dir="rtl">لم آكل شيئا اليوم</span> | <span dir="rtl">ما كلتش حاجة النهاردة</span> |
| p17 | question | How did you arrive? | <span dir="rtl">كيف وصلت؟</span> | <span dir="rtl">وصلت ازاي؟</span> |
| p18 | question | What is your name? | <span dir="rtl">ما اسمك؟</span> | <span dir="rtl">اسمك إيه؟</span> |
| p19 | question | When will you return? | <span dir="rtl">متى ستعود؟</span> | <span dir="rtl">هترجع إمتى؟</span> |
| p20 | question | Why are you late? | <span dir="rtl">لماذا تأخرت؟</span> | <span dir="rtl">اتأخرت ليه؟</span> |
| p21 | question | Where is the house? | <span dir="rtl">أين البيت؟</span> | <span dir="rtl">البيت فين؟</span> |
| p22 | question | Who is at the door? | <span dir="rtl">من على الباب؟</span> | <span dir="rtl">مين على الباب؟</span> |
| p23 | question | How much is this? | <span dir="rtl">كم ثمن هذا؟</span> | <span dir="rtl">ده بكام؟</span> |
| p24 | question | Do you speak Arabic? | <span dir="rtl">هل تتكلم العربية؟</span> | <span dir="rtl">بتتكلم عربي؟</span> |
| p25 | demonstrative | This man is my friend. | <span dir="rtl">هذا الرجل صديقي</span> | <span dir="rtl">الراجل ده صاحبي</span> |
| p26 | demonstrative | This car is new. | <span dir="rtl">هذه السيارة جديدة</span> | <span dir="rtl">العربية دي جديدة</span> |
| p27 | demonstrative | Those people are kind. | <span dir="rtl">هؤلاء الناس طيبون</span> | <span dir="rtl">الناس دول طيبين</span> |
| p28 | lexical | I'm leaving now. | <span dir="rtl">أنا ذاهب الآن</span> | <span dir="rtl">أنا ماشي دلوقتي</span> |
| p29 | lexical | I want only one cup. | <span dir="rtl">أريد كوبا واحدا فقط</span> | <span dir="rtl">عايز كوباية واحدة بس</span> |
| p30 | aspect | She is studying right now. | <span dir="rtl">هي تدرس الآن</span> | <span dir="rtl">هي بتذاكر دلوقتي</span> |

## Per-triple notes

### p01 — we are students *(pronoun)*

- **English:** We are students.
- **MSA:** <span dir="rtl">نحن طلاب</span>
- **Masri:** <span dir="rtl">احنا طلبة</span>
- 1pl pronoun: نحن (MSA) → احنا (Masri). Plural of 'student' shifts طلاب → طلبة.

### p02 — you (pl) are welcome here *(pronoun)*

- **English:** You are welcome here.
- **MSA:** <span dir="rtl">أنتم مرحب بكم هنا</span>
- **Masri:** <span dir="rtl">انتو مرحب بيكم هنا</span>
- 2pl pronoun أنتم → انتو; preposition+suffix بكم → بيكم.

### p03 — I don't know *(negation)*

- **English:** I don't know.
- **MSA:** <span dir="rtl">لا أعرف</span>
- **Masri:** <span dir="rtl">مش عارف</span>
- Present-tense negation: MSA preverbal لا + imperfect → Masri مش + active participle. Different syntactic strategy entirely.

### p04 — he didn't come *(negation)*

- **English:** He didn't come.
- **MSA:** <span dir="rtl">لم يأتِ</span>
- **Masri:** <span dir="rtl">ما جاش</span>
- Past negation: MSA لم + jussive → Masri circumfix ما...ش on the perfect. Verb root also differs (أتى → جا).

### p05 — we won't go *(negation)*

- **English:** We won't go.
- **MSA:** <span dir="rtl">لن نذهب</span>
- **Masri:** <span dir="rtl">مش هنروح</span>
- Future negation: MSA لن + subjunctive → Masri مش + future-prefix هـ + imperfect. Verb ذهب → راح.

### p06 — this is not true *(negation)*

- **English:** This is not true.
- **MSA:** <span dir="rtl">هذا ليس صحيحا</span>
- **Masri:** <span dir="rtl">ده مش صح</span>
- Nominal negation: MSA defective verb ليس → Masri invariant مش. Demonstrative هذا → ده.

### p07 — I will travel tomorrow *(future)*

- **English:** I will travel tomorrow.
- **MSA:** <span dir="rtl">سأسافر غدا</span>
- **Masri:** <span dir="rtl">هاسافر بكرة</span>
- Future marker: MSA prefix سـ → Masri prefix هـ/حـ. Time adverb غدا → بكرة.

### p08 — they will arrive at noon *(future)*

- **English:** They will arrive at noon.
- **MSA:** <span dir="rtl">سيصلون عند الظهر</span>
- **Masri:** <span dir="rtl">هيوصلوا الضهر</span>
- 3pl future. Phonological shift ظ → ض in الظهر → الضهر is characteristic of Cairene Masri.

### p09 — I want some water *(verb)*

- **English:** I want some water.
- **MSA:** <span dir="rtl">أريد بعض الماء</span>
- **Masri:** <span dir="rtl">عايز شوية ميه</span>
- Quintessential lexical split: أريد (verb) → عايز (active participle). 'water' الماء → ميه (different orthography for the same etymon).

### p10 — what do you want? *(verb)*

- **English:** What do you want?
- **MSA:** <span dir="rtl">ماذا تريد؟</span>
- **Masri:** <span dir="rtl">عايز إيه؟</span>
- Wh-fronting (MSA) vs wh-in-situ (Masri). ماذا → إيه, postposed.

### p11 — what are you doing now? *(verb)*

- **English:** What are you doing now?
- **MSA:** <span dir="rtl">ماذا تفعل الآن؟</span>
- **Masri:** <span dir="rtl">بتعمل إيه دلوقتي؟</span>
- Verb فعل → عمل; aspectual prefix بـ marks habitual/progressive in Masri (no MSA equivalent).

### p12 — I saw him yesterday *(verb)*

- **English:** I saw him yesterday.
- **MSA:** <span dir="rtl">رأيته أمس</span>
- **Masri:** <span dir="rtl">شفته إمبارح</span>
- Verb رأى → شاف (etymologically distinct roots). Adverb أمس → إمبارح.

### p13 — where are you going? *(verb)*

- **English:** Where are you going?
- **MSA:** <span dir="rtl">إلى أين تذهب؟</span>
- **Masri:** <span dir="rtl">رايح فين؟</span>
- MSA preposition+wh إلى أين → bare wh فين in Masri. Verb ذهب → راح (active participle رايح).

### p14 — come here please *(verb)*

- **English:** Come here, please.
- **MSA:** <span dir="rtl">تعال هنا من فضلك</span>
- **Masri:** <span dir="rtl">تعالى هنا لو سمحت</span>
- Imperative form differs slightly (تعال → تعالى); politeness formula من فضلك → لو سمحت ('if you've permitted').

### p15 — give me the book *(verb)*

- **English:** Give me the book.
- **MSA:** <span dir="rtl">أعطني الكتاب</span>
- **Masri:** <span dir="rtl">إديني الكتاب</span>
- Verb أعطى → إدّى — dialectal lexical replacement. Object suffix structure identical.

### p16 — I haven't eaten anything today *(verb)*

- **English:** I haven't eaten anything today.
- **MSA:** <span dir="rtl">لم آكل شيئا اليوم</span>
- **Masri:** <span dir="rtl">ما كلتش حاجة النهاردة</span>
- Compound differences: negation strategy (p04), 'thing' شيء → حاجة, 'today' اليوم → النهاردة (an archetypal Masri lexeme).

### p17 — how did you arrive? *(question)*

- **English:** How did you arrive?
- **MSA:** <span dir="rtl">كيف وصلت؟</span>
- **Masri:** <span dir="rtl">وصلت ازاي؟</span>
- Wh-particle كيف → ازاي; in-situ position.

### p18 — what is your name? *(question)*

- **English:** What is your name?
- **MSA:** <span dir="rtl">ما اسمك؟</span>
- **Masri:** <span dir="rtl">اسمك إيه؟</span>
- ما → إيه; postposed in Masri. Both are short and high-frequency — good probe for early-layer divergence.

### p19 — when will you return? *(question)*

- **English:** When will you return?
- **MSA:** <span dir="rtl">متى ستعود؟</span>
- **Masri:** <span dir="rtl">هترجع إمتى؟</span>
- Wh متى → إمتى; future سـ → هـ; verb عاد → رجع.

### p20 — why are you late? *(question)*

- **English:** Why are you late?
- **MSA:** <span dir="rtl">لماذا تأخرت؟</span>
- **Masri:** <span dir="rtl">اتأخرت ليه؟</span>
- Wh لماذا → ليه; verb prefix تـ → اتـ (Masri Form V/VII reanalysis).

### p21 — where is the house? *(question)*

- **English:** Where is the house?
- **MSA:** <span dir="rtl">أين البيت؟</span>
- **Masri:** <span dir="rtl">البيت فين؟</span>
- Wh أين → فين; in-situ. Verbless copula identical.

### p22 — who is at the door? *(question)*

- **English:** Who is at the door?
- **MSA:** <span dir="rtl">من على الباب؟</span>
- **Masri:** <span dir="rtl">مين على الباب؟</span>
- Wh من → مين. Single-letter difference but on a high-frequency content word — interesting probe.

### p23 — how much is this? *(question)*

- **English:** How much is this?
- **MSA:** <span dir="rtl">كم ثمن هذا؟</span>
- **Masri:** <span dir="rtl">ده بكام؟</span>
- Wh كم → كام (with cliticised preposition بـ); MSA needs lexical ثمن ('price'), Masri folds price into the preposition.

### p24 — do you speak Arabic? *(question)*

- **English:** Do you speak Arabic?
- **MSA:** <span dir="rtl">هل تتكلم العربية؟</span>
- **Masri:** <span dir="rtl">بتتكلم عربي؟</span>
- Yes/no marker هل dropped entirely in Masri (intonation does the work). Aspectual بـ; bare العربية → عربي.

### p25 — this man is my friend *(demonstrative)*

- **English:** This man is my friend.
- **MSA:** <span dir="rtl">هذا الرجل صديقي</span>
- **Masri:** <span dir="rtl">الراجل ده صاحبي</span>
- Demonstrative هذا → ده, postposed (MSA pre-, Masri post-nominal). Lexical: رجل → راجل (vowel insertion), صديق → صاحب.

### p26 — this car is new *(demonstrative)*

- **English:** This car is new.
- **MSA:** <span dir="rtl">هذه السيارة جديدة</span>
- **Masri:** <span dir="rtl">العربية دي جديدة</span>
- Feminine demonstrative هذه → دي. Lexical سيارة → عربية ('Arab one'; same etymon, different specialisation).

### p27 — those people are kind *(demonstrative)*

- **English:** Those people are kind.
- **MSA:** <span dir="rtl">هؤلاء الناس طيبون</span>
- **Masri:** <span dir="rtl">الناس دول طيبين</span>
- Plural demonstrative هؤلاء → دول. Plural adjective sound-masculine -ون → -ين in Masri.

### p28 — I'm leaving now *(lexical)*

- **English:** I'm leaving now.
- **MSA:** <span dir="rtl">أنا ذاهب الآن</span>
- **Masri:** <span dir="rtl">أنا ماشي دلوقتي</span>
- 'going' ذاهب → ماشي ('walking' generalised); 'now' الآن → دلوقتي (lit. 'this-the-time').

### p29 — I want only one cup *(lexical)*

- **English:** I want only one cup.
- **MSA:** <span dir="rtl">أريد كوبا واحدا فقط</span>
- **Masri:** <span dir="rtl">عايز كوباية واحدة بس</span>
- 'cup' كوب → كوباية (feminine extension); 'only' فقط → بس (one of the most frequent Masri-marker words).

### p30 — she is studying right now *(aspect)*

- **English:** She is studying right now.
- **MSA:** <span dir="rtl">هي تدرس الآن</span>
- **Masri:** <span dir="rtl">هي بتذاكر دلوقتي</span>
- Aspectual prefix بـ on the imperfect = progressive in Masri; MSA bare imperfect is aspectually underspecified. Verb درس → ذاكر.
