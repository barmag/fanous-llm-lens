"""Tests for morphological segmentation and the gold-standard boundaries.

These tests encode Bug 1 from the 2026-06-29 plan revision: the original
`morpheme_boundaries` summed `len(morph)` over d3tok segments, which carry clitic
markers (``و+``, ``ال+``) and drop inter-word whitespace, so the offsets did not map
to surface character positions. The redesigned contract returns intra-word morpheme
seams as surface character offsets.
"""

from __future__ import annotations

from fanous_lens.tokenizers.morphological import (
    analyze_morphology,
    morpheme_boundaries,
    morpheme_boundaries_with_coverage,
)


def _assert_valid_seams(text: str, bounds: list[int]) -> None:
    """Invariants every seam list must satisfy, regardless of register.

    Indices are in logical (reading) order, which for Arabic is right-to-left: index 0
    is the rightmost glyph on screen. A seam is the surface offset where a morpheme
    begins *inside* a word.
    """
    assert bounds == sorted(bounds)
    for b in bounds:
        assert 0 < b < len(text)  # never 0, never past the end
        assert text[b] != " "  # always inside a word, never on a whitespace gap


def test_morpheme_boundaries_are_intra_word_surface_offsets():
    text = "وبالقلم كتب الولد"  # clitic-heavy; len == 17
    bounds = morpheme_boundaries(text)
    assert all(0 < b < len(text) for b in bounds)  # never overshoots len, never 0
    assert all(text[b] != " " for b in bounds)  # every seam is inside a word
    assert bounds == sorted(bounds)


def test_morpheme_boundaries_no_seam_lands_on_whitespace():
    # "مش عارف" — the old code dropped the space and ended at 6 (< len 7).
    text = "مش عارف"
    bounds = morpheme_boundaries(text)
    assert all(0 < b < len(text) for b in bounds)
    assert all(text[b] != " " for b in bounds)


def test_morpheme_boundaries_catch_clitic_seam():
    # "البيت" = definite article ال + بيت; the seam falls between index 2 and 3.
    text = "البيت"
    bounds = morpheme_boundaries(text)
    # camel-tools should split the ال proclitic; at minimum a seam exists inside the word.
    assert any(0 < b < len(text) for b in bounds)


def test_analyze_morphology_still_returns_tokens():
    morphs = analyze_morphology("البيت الكبير")
    assert isinstance(morphs, list)
    assert all(isinstance(m, str) for m in morphs)
    assert morphs  # non-empty


# ───────────────────────── Masri (Egyptian) coverage ─────────────────────────
#
# The gold standard runs on calima-msa-r13, an MSA morphological database. It
# segments clitics shared by MSA and Masri (conjunction و+, preposition ب+,
# article ال+, enclitic +ه/+ها) but does NOT know Masri-specific morphology
# (بتاع possessives, the progressive بـ, the future هـ, Egyptian vocab). These
# tests pin both the working cases and the known limitation, so an upgrade to a
# Masri-aware analyzer (e.g. CALIMA-EGY) would surface as a test change to review.


def test_masri_shared_clitics_produce_intra_word_seams():
    # Real Masri text whose clitics MSA shares, so seams MUST be non-empty —
    # this is the representativeness fix for the empty "مش عارف" case.
    sentences = [
        "كتابه كبير",  # his book is big (enclitic possessive +ه)
        "وعايزين نروح",  # and we-want to-go (conjunction و+ on a Masri verb)
        "بالعربية",  # by car (preposition ب+ + article ال+)
        "البيت بتاعنا",  # our house (article ال+ peels; بتاعنا stays whole)
    ]
    for s in sentences:
        bounds, _n_words, skipped = morpheme_boundaries_with_coverage(s)
        _assert_valid_seams(s, bounds)
        assert bounds, f"expected at least one intra-word seam in {s!r}"
        assert skipped == 0


def test_masri_enclitic_pronoun_seam_is_exact():
    # كتابه = كتاب + ه ("his book"); the only seam is before the enclitic ه.
    text = "كتابه كبير"
    bounds = morpheme_boundaries(text)
    assert bounds == [4]
    assert text[4] == "ه"


def test_masri_proclitic_stack_preposition_plus_article():
    # بالعربية = ب + ال + عربية; two stacked proclitics → two seams.
    text = "بالعربية"
    bounds = morpheme_boundaries(text)
    assert bounds == [1, 3]
    assert text[1] == "ا"  # start of the ال article
    assert text[3] == "ع"  # start of the stem عربية


def test_reconstruction_guard_skips_orthographic_assimilation():
    # When proclitics assimilate in writing, stripped pieces no longer concatenate
    # to the surface form, so the guard skips the word instead of emitting wrong
    # offsets. لـ+الـ → written لل (alef drops); ة→ت under suffixation in عربيتها.
    for text in ("للمدرسة", "عربيتها"):
        bounds, _n_words, skipped = morpheme_boundaries_with_coverage(text)
        assert skipped >= 1, f"expected {text!r} to be skipped by the guard"
        _assert_valid_seams(text, bounds)  # whatever survives must still be valid


def test_masri_specific_morphology_undersegmented_by_msa_gold():
    # Documented limitation: the MSA gold does not split Masri-specific affixes.
    # These yield no intra-word seam today. If a Masri-aware analyzer is adopted,
    # this test should start failing — that is the signal to revisit the gold.
    for text in ("بيكتب", "هيروح", "بتاعنا"):
        bounds = morpheme_boundaries(text)
        assert bounds == [], f"{text!r} now segments — re-evaluate the Masri gold standard"
