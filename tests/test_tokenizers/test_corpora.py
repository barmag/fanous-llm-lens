"""Tests for corpus loading utilities."""

from __future__ import annotations

from fanous_lens.tokenizers.corpora import load_corpora


def test_load_corpora_returns_strings():
    msa, masri = load_corpora(max_msa=10, max_masri=10)
    assert isinstance(msa, list)
    assert isinstance(masri, list)
    assert all(isinstance(s, str) for s in msa)
    assert all(isinstance(s, str) for s in masri)


def test_load_corpora_respects_max():
    msa, masri = load_corpora(max_msa=5, max_masri=3)
    assert len(msa) <= 5
    assert len(masri) <= 3
