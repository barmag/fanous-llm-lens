# Replicating *A Mathematical Framework for Transformer Circuits* on Egyptian Arabic

**Report date:** 2026-07-02
**Paper:** Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* (Anthropic)
**Substrate:** AMD Strix Halo iGPU (gfx1151 → gfx1100 override), torch 2.5.1+rocm6.2, TransformerLens, seed 42 throughout
**Data:** Arabic Wikipedia (MSA) + Egyptian-dialect tweets (Masri) — MSA-heavy by necessity; Masri data is scarce

This report consolidates what the fanous-llm-lens replication line reproduced from the
2021 framework paper, what it could not reproduce (or had to correct), and where the
line should go next. The distinguishing move throughout: the paper's claims are
re-run **on Arabic**, with tokenization treated as an experimental variable rather
than a fixed background choice.

---

## 1. Map: notebooks ↔ paper claims

| Paper claim | Where we replicated it | Status |
|---|---|---|
| Zero-layer transformer ⇒ bigram statistics only | `experiments/embedding-probes/` (Phase A, 5 tokenizers) + Stage 2 rung 0 | ✅ Confirmed, and weaponised as a control |
| One-layer attention-only = bigram + skip-trigram ensemble (exact decomposition) | `notebooks/education/stage2_dash_skip_trigram_reference.ipynb` | ✅ Exact (diff ≈ 1e-5) |
| QK/OV circuit factorization per head | same notebook + by-hand QK in `stage2_a_single_block_reference.ipynb` | ✅ |
| Skip-trigrams are real and legible | `stage2_dash_skip_trigram_reference.ipynb` §5–§7 | ⚠️ Partially — a small set of genuine frames; most candidates are artifacts (see §3.1) |
| The "skip-trigram bug" (OV is destination-independent) | same notebook §5c | ✅ Reproduced structurally, promoted to headline |
| Copying / self-attention copy heads | same notebook (head 0 control) | ✅ |
| Two-layer attention-only models grow induction heads | `stage2_dash2_composition_induction_reference.ipynb` + `stage2_c_depth_induction_reference.ipynb` | ✅ (score 0.43, one head) |
| Induction is built by **K-composition** specifically | `stage2_dash2` §4–§8 | ✅ (weight-space evidence; no ablation) |
| Q/K/V composition measurable in weights | `stage2_dash2` §4 | ✅ |
| Virtual attention heads / term importance | `stage2_dash2` §5–§6 | ✅ computed; not causally validated |
| Eigenvalue copying analysis | `stage2_dash2` §7 | ⚠️ Tractable proxy only |
| MLPs are where the clean story stops | `stage2_d_mlp_reference.ipynb` | ✅ boundary demonstrated exactly |

Model line used for the headline results ("2dash" family): attention-only
`HookedTransformer`, `d_model=512`, 8 heads, `d_head=64`, `n_ctx=512`, vocab 12k,
**LayerNorm-free + shortformer positions** — a deliberate deviation so the paper's
path expansion is *exact* rather than approximate. 1-layer: 13.6M params, trained
337.6M tokens, loss 9.393 → 4.964 (~62 min on the iGPU). 2-layer: 14.7M params,
same tokens/steps, loss 9.393 → 4.597. Checkpoints on the HF Hub
(`yassermakram/fanous-stage2dash-attn-only-1l`, `…-unigram`, `…-stage2dash2-attn-only-2l`).

---

## 2. What we replicated — with representative examples

### 2.1 Zero-layer = bigram statistics, turned into a methodological control

The paper's zero-layer theorem (embed → unembed can only express token→token
statistics) is not just confirmed — Phase A uses it as an instrument. Five
zero-layer models (one per tokenizer: morphological/camel-tools, Unigram,
WordPiece, Morfessor, BPE) were trained on the same 30k MSA+Masri sentences, and
morphological features were probed from `W_E`.

The decisive result: **trained-vs-untrained probe deltas are all within ±0.014
AUC** — while final losses ranged 5.46–7.74 (so `W_E` was substantially
rewritten). A zero-layer model's embeddings carry only what the tokenization
makes linearly available; the probe scores the tokenizer, not learning. Two
representative rows (MSA, definiteness / number AUC, trained → untrained):

| tokenizer | definite | number |
|---|---|---|
| morphological | 0.929 → 0.931 | 0.596 → 0.586 |
| unigram | 0.860 → 0.872 | 0.762 → 0.773 |

The morphological tokenizer's definiteness "win" is a one-token artifact (a
dedicated `ال` token hands the label to any probe, trained or not). The number
ranking is the real finding: subword tokenizers keep the sound-plural suffix
(`…ون`, `…ات`) as a shared carrier; the morphological tokenizer fuses it into
unique stems and lands *last* (0.596). **Morpheme alignment is a feature-specific
win, not a universal one.**

### 2.2 One-layer: the decomposition is exact, and the QK/OV story holds

- `logits = direct path (bigram) + Σ heads (skip-trigram)` verified numerically:
  max abs difference from the model's actual logits ≈ **1e-5**. The LN-free +
  shortformer choice makes this exact by construction — the one unconditionally
  solid result.
- The bigram skeleton `W_E·W_U` and per-head `QK = W_E·W_Q·W_Kᵀ·W_Eᵀ`,
  `OV = W_E·W_V·W_O·W_U` factorizations behave as the paper describes. In stage2_a
  the attention pattern is recomputed by hand (scores, causal mask, softmax) and
  matches the model's cached pattern to ~0.
- Aggregate context effect: the skip-trigram term reshapes the next-token
  distribution by 0.12 on average and **flips the top-1 prediction at 89% of
  positions** — attention is doing real work even in one layer.

### 2.3 Genuine Arabic skip-trigrams exist and are legible

Held-out verification (does the full model beat the bigram-only direct path on a
constructed `[source, …, dest]` context?) surfaced a small set of genuine
three-way frames, living on specific heads:

| skip-trigram | lift | head | gloss |
|---|---|---|---|
| الرغم … من → أن | +0.27 | 6 | *ʕala r-raghmi min ʔanna* — "despite the fact that" (MSA concessive) |
| رض … ي → الله | +0.55 | 2 | *raḍiya llāh* — religious formula (BPE splits رضي → رض+ي) |
| تبلغ … من → العمر | +0.98 | 6 | *tablughu min al-ʕumr* — "is X years old" |
| بالإضافة … إلى → ذلك | +0.07 | — | *bil-ʔiḍāfa ʔilā dhālik* — "in addition to that" |

Frames concentrate on heads 2 and 6; morphology promotions on head 1; head 0 is
copy-dominated (self-copy control: هوكي→هوكي +0.93), matching the paper's copying
heads.

### 2.4 The skip-trigram bug reproduces — and turns out to be the story

The paper's observation that a head cannot jointly condition on source *and*
destination (OV[source] is destination-independent, so `keep…in → mind` drags
along `keep…in → bay`) reproduces structurally in Arabic: تبلغ promotes both
العمر and أمريكية regardless of destination. On our model this bug is not a
footnote — it explains most of the "verified" candidates (see §3.1).

### 2.5 Tokenizer-conditioning: an extension beyond the paper

Two models with identical architecture, corpus, and training — differing only in
tokenizer (BPE vs Unigram, both 12k vocab, whitespace pre-tokenization forced on
both) — host **different skip-trigrams**:

- BPE splits رضي → رض+ي, so the lexical frame *رضي الله* can exist. Unigram keeps
  رضي whole — that frame **cannot exist** in the Unigram model at all.
- Unigram fragments toward characters; its frames are sub-lexical: the
  masculine-plural ـون concord chain (عسكريو…ن → أمريكيون) and character
  completion (الرغم…م → ن).

Punchline from the notebook: *"§7's story is not an abstract property of the
model — it is conditioned on tokenization."* The paper's skip-trigram taxonomy
(code, URLs, LaTeX) is itself an English/BPE artifact; the taxonomy that appears
depends on where the tokenizer cuts.

### 2.6 Two-layer: induction head via K-composition

The capstone. The 2-layer twin (same tokens, steps, and seed as the 1-layer
model) grew exactly one induction head:

- **Layer 1, head 4, induction score 0.4325** — a sharp isolated spike; every
  other head in both layers scores < 0.01 (layer-1 row:
  `[6e-5, 5e-5, 9e-5, 8e-3, 0.4325, 5e-7, 4e-6, 7e-7]`). The training gate
  (score ≥ 0.4 required before the checkpoint saves) passed.
- The second layer buys **~0.37 nats** on identical data (4.964 → 4.597 final
  loss), consistent with the new copying-novel-tokens capability that
  skip-trigrams provably lack.
- Attribution follows the paper: Q/K/V composition scores computed in weight
  space (Frobenius-ratio, 8×8 per pair); the layer-0 head that K-composes
  strongest into L1.H4 is confirmed to be a **previous-token head** (mean
  attention to position i−1); Q- and V-composition are shown but explicitly
  *not* credited. Notebook thesis: *"Induction = K-composition specifically —
  not Q-composition, not V-composition — confirmed on real Arabic."*
- The head fires on fresh held-out Masri: النهارده الجو حلو والشمس طالعة
  (*in-naharda ig-gaww ḥilw wiš-šams ṭalʕa* — "today the weather is nice and the
  sun is out"), repeated, shows the induction stripe.
- Independently, stage2_c demonstrates the same circuit on a synthetic
  variable-gap repeat task (the variable gap defeats positional-copy shortcuts,
  forcing genuine content-based composition), with an L0 prev-token diagonal +
  L1 induction stripe, and the score holding on unseen sequences — algorithm,
  not memorisation.

### 2.7 The MLP boundary, demonstrated exactly

stage2_d shows by construction what the paper asserts: the MLP run on one
position in isolation vs in the full sequence differs by **exactly 0** (it moves
no information between positions), while changing a distant token shifts the
attention output at the last position. And the notebook says honestly that this
is where the 2021 framework's clean weights-first story stops — the bridge to
features/SAE work.

---

## 3. What we could not replicate, or had to correct

### 3.1 Most "verified" skip-trigrams are not skip-trigrams (self-correction)

The central honesty finding of the line. Raw held-out verification passed
48/52/49/49 candidates across the four seeded categories — but tracking distinct
`(source → output)` pairs collapses these to **29/23/17/16**. Because OV[source]
is destination-independent, the *same* promotion re-verifies under several
destinations, inflating counts. Two whole categories (definite-article
morphology, MSA↔Masri contrast) are **null as skip-trigrams**: everything that
verified there is a destination-independent OV promotion — real morphology in
the OV circuit, but not a three-way `[A…B]→C` pattern. The notebook explicitly
corrects its own earlier overclaim. The strongest raw lifts (الاكت→ابات +0.98)
are also artifacts: the model completing a BPE-fragmented word.

An earlier version of the notebook was worse — its example anchored the source
on the OV diagonal, forcing output==source (pure self-copy), and omitted the
skip-trigram bug entirely. The honest-skip-trigrams redo replaced it.

### 3.2 The induction head is modest, singular, and not causally verified

- Score **0.4325** clears our 0.4 gate but is far from the near-saturated
  induction heads in the paper's figures. There is exactly one; no
  backup/redundant induction heads and no partial-induction ensemble.
- **No ablation evidence anywhere.** The K-composition attribution rests on
  weight-space composition scores plus attention patterns — the paper's
  knockout-style causal evidence (ablate the prev-token head, watch induction
  die) was not reproduced.
- No separate **prefix-matching vs copying scores** (the criteria Anthropic
  later used to define induction heads); we have the attention-stripe score and
  an OV eigenvalue proxy instead.
- No training-dynamics **induction bump**: only start/end losses are logged, so
  the phase transition is inferred from the 1L-vs-2L final-loss gap, never
  plotted.

### 3.3 The eigenvalue copying analysis is a proxy

We compute the positive-eigenvalue fraction of `W_V·W_O` over the non-zero
spectrum (rank ≤ d_head = 64), not the paper's full vocab-space analysis of
`W_Eᵀ·W_OV·W_U`. The notebook flags this itself.

### 3.4 The dialect signal did not materialise

The project's motivating variable — Masri vs MSA — stayed weak everywhere the
skip-trigram line looked. Masri اللي is absent from the frames; the MSA↔Masri
contrast category surfaced only MSA الذي; Masri bigrams (اللي، عايز، دلوقتي) are
weak or fragmented. This is a data finding (the corpus is MSA-heavy because
Masri data is scarce), not a model bug — but it means the paper-replication has
so far been achieved *on Arabic*, not yet *on the dialect contrast*.

### 3.5 Depth does not reconstruct what the tokenizer fused (falsified prediction)

The depth-probe experiment predicted two transformer layers would reconstruct
the plural suffix that the morphological tokenizer fuses away. **Falsified**: the
number probe *erodes* slightly with depth (unigram 0.80→0.78; morphological
0.62→0.60), the Phase-A tokenizer ranking survives intact through L2, and the
only real training increment (definiteness for BPE/WordPiece, Δ+0.05–0.07) lives
at **L0, in the embedding**, not in the layers. Governing rule: recoverability
depends on whether a feature keeps a shared surface signature, not on depth.
(Caveat that bounds the null: mean-pooled readout may dilute signals attention
relocates to one position.)

### 3.6 Standing methodological caveats

- **Not the paper's literal architecture.** LN-free + shortformer was chosen so
  the decomposition is exact. We reproduce the paper's *results*, not its model;
  "faithful-scale ≠ faithful architecture" (the notebook's own words).
- **Single seed (42) everywhere**; head-kind diagnosis in the 1-layer model rests
  on a single sentence.
- **Unequal token budgets in the BPE/Unigram comparison** (337.6M vs 406.5M
  tokens; final losses 4.96 vs 3.95) — an acknowledged confound in the
  tokenizer-conditioning claim.
- **Byte-premium / fertility confound in Phase A**: models trained on equal
  *sentences*, not equal *tokens*; cross-tokenizer AUC gaps conflate segmentation
  quality with effective data. Flagged in every report, not yet controlled.
- **Notebooks ship without cached outputs** (project convention for sane diffs).
  Hard numbers live in `checkpoints/*/metrics.json`, the §9 recap markdown, and
  `experiments/embedding-probes/phase_a_*.json`; composition-score and
  eigenvalue values are reproducible only by running the notebooks.

---

## 4. Ten directions for future work

1. **Causal ablation of the induction circuit.** Zero-ablate the layer-0
   previous-token head and confirm the L1.H4 induction score collapses (and
   loss on repeated text rises); ablate a Q-composing head as the control. This
   upgrades the K-composition claim from weight-space evidence to causal
   evidence — the paper-replication's biggest missing piece.
2. **Training-dynamics study of the induction bump.** Re-run the 2-layer
   training with periodic checkpoints, plot per-step loss and induction score,
   and check whether the phase transition (bump) appears on Arabic data as it
   does in English — first look at *when* the circuit forms, not just that it
   forms.
3. **Prefix-matching and copying scores + backup heads.** Implement the
   two-score induction-head criteria, scan both models, and test for redundant
   or partial induction heads (the paper's later work found ensembles; we found
   exactly one head — real difference or small-model artifact?).
4. **Fertility-matched rerun of Phase A** (top priority in HANDOVER): equalize
   training *tokens* rather than sentences across the five tokenizers, log
   Masri-vs-MSA fertility per tokenizer, and decompose the Masri probe gap into
   effective-data vs segmentation. Same fix applies to the BPE-vs-Unigram
   skip-trigram comparison (337.6M vs 406.5M tokens).
5. **Faithful-architecture twin.** Train the paper's literal `attn-only-2l`
   (LayerNorm + learned positions) on the same Arabic corpus and measure how
   much the exact-decomposition results degrade — quantifying what the LN-free +
   shortformer idealisation buys and costs.
6. **Masri-balanced corpus rerun.** Rebuild the corpus with a much higher Masri
   share (or Masri-only fine-tuning) and re-run the skip-trigram scan: do
   dialect frames (اللي، عايز، دلوقتي) appear once the data supports them? This
   directly tests whether §3.4 is a data artifact.
7. **Full vocab-space copying analysis and virtual heads.** Replace the
   eigenvalue proxy with the paper's `W_Eᵀ·W_OV·W_U` analysis, and analyse the
   composed (virtual) head `OV₀·OV₁` products explicitly — including whether the
   prev-token→induction virtual head is itself a copying matrix.
8. **Multi-seed robustness.** Re-train the 1L and 2L models at 3–5 seeds: do the
   same genuine frames re-appear on the same heads? Does the induction head
   always emerge, at similar strength, always via K-composition? Everything so
   far is seed-42-only.
9. **Non-concatenative morphology as a probe target.** Add root-and-pattern
   features (root identity, broken plurals — which no tokenizer segments) plus
   the intervention test from HANDOVER: force a shared plural token into the
   morphological tokenizer and re-probe number; try a last-token readout to
   bound the mean-pooling caveat on the depth null.
10. **Cross the framework's boundary, and bridge to real models.** Two natural
    continuations: (a) train a small SAE on the 2-layer+MLP model (stage2_d ends
    exactly where features begin); (b) verify induction heads and skip-trigram
    structure in a pretrained model (Pythia-70m/160m) on Arabic prompts,
    connecting the from-scratch ladder to models people actually use — and
    checking whether Arabic induction in the wild also runs through
    K-composition.

---

## Appendix: where the evidence lives

- 1-layer skip-trigrams: `notebooks/education/stage2_dash_skip_trigram_reference.ipynb`
  (§9 recap holds the frozen numbers), helpers in `notebooks/education/skip_trigrams.py`,
  trainer `train_stage2dash.py`, metrics in `notebooks/education/checkpoints/stage2dash*/metrics.json`
- 2-layer composition/induction: `notebooks/education/stage2_dash2_composition_induction_reference.ipynb`,
  trainer `train_stage2dash2.py`, metrics in `checkpoints/stage2dash2/metrics.json`
- Architecture ladder (pedagogical line): `notebooks/education/stage2_{a,b,c,d}_*_reference.ipynb`, shared model code `tiny.py`
- Zero-layer + depth probes: `experiments/embedding-probes/` (`phase_a_results.json`,
  `phase_a_baseline.json`, `phase_a_depth_results.json`), reports in
  `docs/reports/phase-a-probes.md`, `phase-a-depth.md`, `HANDOVER.md`
- Specs/plans: `docs/superpowers/specs/2026-06-{26,27,28,30}-*.md`
- Checkpoints: HF Hub `yassermakram/fanous-stage2dash-attn-only-1l`, `…-unigram`,
  `…-stage2dash2-attn-only-2l`
