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
)


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
