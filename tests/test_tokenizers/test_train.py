"""Tests for tokenizer training and the offset-capable encoder.

Per the 2026-06-29 plan revision (Bug 2), every tokenizer's encoder returns
``(ids, offsets)`` so the evaluation layer can compare *real* token boundaries
against the morpheme gold standard — not a token-count approximation.
"""

from __future__ import annotations

import pytest

from fanous_lens.tokenizers.train import get_tokenizer, train_tokenizer

# Small mixed MSA + Masri corpus; enough to train tiny tokenizers.
CORPUS = [
    "البيت الكبير",
    "وسيذهبون إلى المدرسة",
    "عايز شوية ميه",
    "احنا طلبة في الجامعة",
    "مش عارف عربي كويس",
    "الكتاب على الطاولة",
    "بالقلم كتب الولد الجواب",
]

ALL_APPROACHES = ["bpe", "unigram", "wordpiece", "morfessor", "morphological"]


@pytest.mark.parametrize("approach", ["bpe", "unigram", "wordpiece"])
def test_train_hf_returns_loadable_config(approach):
    config = train_tokenizer(approach, CORPUS, vocab_size=200)
    # HuggingFace tokenizers serialize the vocab under config["model"].
    assert isinstance(config, dict)
    assert "model" in config
    assert config["model"].get("vocab")


def test_train_morfessor_returns_vocab():
    config = train_tokenizer("morfessor", CORPUS, vocab_size=200)
    assert config["vocab"]


def test_morfessor_actually_segments():
    # Regression guard for the load_data bug: passing tuple(word) instead of the
    # surface string made every word an unsplittable atom, so morfessor NEVER split
    # (fertility 1.0) yet still returned a vocab — invisible to the test above.
    # Two long, frequent units that also appear concatenated must split back apart.
    corpus = (
        ["مدرسة"] * 40
        + ["كبيرة"] * 40
        + ["مدرسةكبيرة"] * 5
        + ["كتاب"] * 40
        + ["صغير"] * 40
        + ["كتابصغير"] * 5
    )
    config = train_tokenizer("morfessor", corpus, vocab_size=200)
    encode = get_tokenizer("morfessor", config)
    ids, _offsets = encode("مدرسةكبيرة")
    assert len(ids) >= 2, (
        "morfessor failed to segment a clearly compound word — load_data regression"
    )


def test_train_morphological_returns_vocab():
    config = train_tokenizer("morphological", CORPUS, vocab_size=200)
    assert config["vocab"]


def test_train_unknown_approach_raises():
    with pytest.raises(ValueError):
        train_tokenizer("nonsense", CORPUS, vocab_size=200)


@pytest.mark.parametrize("approach", ALL_APPROACHES)
def test_encoder_returns_ids_and_offsets(approach):
    text = "البيت الكبير"
    config = train_tokenizer(approach, CORPUS, vocab_size=200)
    encode = get_tokenizer(approach, config)

    ids, offsets = encode(text)

    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(offsets) == len(ids)
    for start, end in offsets:
        assert 0 <= start <= end <= len(text)


@pytest.mark.parametrize("approach", ALL_APPROACHES)
def test_encoder_offsets_are_monotonic_and_in_range(approach):
    text = "بالقلم كتب الولد الجواب"
    config = train_tokenizer(approach, CORPUS, vocab_size=200)
    encode = get_tokenizer(approach, config)
    _ids, offsets = encode(text)
    # offsets should be non-decreasing in start position
    starts = [s for s, _ in offsets]
    assert starts == sorted(starts)
