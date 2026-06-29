"""Tests for the honest tokenizer-fitness diagnostics.

These pin the reframed metric (2026-06-29 design rethink): against an *incomplete* gold we
report clitic **recall paired with fertility** (never precision/F1, which would reward
agreement with the gold's blind spots) plus a gold-free **consistency** signal. The metric
math is tested with controlled fake encoders so the assertions are exact and fast.
"""

from __future__ import annotations

from fanous_lens.tokenizers.evaluate import (
    clitic_recall,
    gold_for,
    greedy_match,
    morpheme_consistency,
    predicted_seams,
    register_separability,
    word_starts,
)


def codepoint_encoder(text: str):
    """Content-based fake encoder: one id per char = its codepoint."""
    return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


def char_encoder(text: str):
    """Split into single characters — the over-segmentation gamer."""
    return list(range(len(text))), [(i, i + 1) for i in range(len(text))]


def whitespace_encoder(text: str):
    """One token per whitespace word — never splits inside a word."""
    ids, offsets, i = [], [], 0
    for k, word in enumerate(text.split(" ")):
        if word:
            offsets.append((i, i + len(word)))
            ids.append(k)
        i += len(word) + 1
    return ids, offsets


def test_word_starts_marks_only_word_initial_offsets():
    assert word_starts("ab cd") == {0, 3}
    assert 1 not in word_starts("ab cd")  # interior char is not a word start


def test_greedy_match_is_one_to_one_under_tolerance():
    # Two gold seams one apart; a single prediction must not satisfy both.
    assert greedy_match([5], [5, 6], tol=1) == 1
    # Each prediction consumes a distinct gold.
    assert greedy_match([5, 6], [5, 6], tol=1) == 2
    # Out of tolerance.
    assert greedy_match([5], [8], tol=1) == 0


def test_char_encoder_maximises_recall_but_betrays_itself_via_fertility():
    # The gaming guard: cutting everywhere finds every gold seam (recall 1.0) but the
    # fertility makes the over-segmentation impossible to miss.
    sentences = ["بالقلم كتب الولد", "الكتاب على الطاولة"]
    golds = [gold_for(s) for s in sentences]
    char = clitic_recall(char_encoder, sentences, golds)
    assert char["recall"] == 1.0
    assert char["fertility"] > 3.0  # ~chars-per-word — the tell


def test_whitespace_encoder_has_zero_recall_and_unit_fertility():
    sentences = ["بالقلم كتب الولد"]
    golds = [gold_for(s) for s in sentences]
    ws = clitic_recall(whitespace_encoder, sentences, golds)
    assert ws["recall"] == 0.0  # never splits inside a word
    assert ws["fertility"] == 1.0


def test_recall_sits_between_the_extremes_for_a_partial_splitter():
    # An encoder that splits only the ال proclitic: catches some gold seams, not all,
    # at fertility well below char-level.
    def peel_al(text: str):
        ids, offsets, cursor = [], [], 0
        for word in text.split(" "):
            start = text.find(word, cursor)
            cursor = start + len(word)
            if word.startswith("ال") and len(word) > 2:
                offsets += [(start, start + 2), (start + 2, start + len(word))]
                ids += [0, 1]
            else:
                offsets.append((start, start + len(word)))
                ids.append(2)
        return ids, offsets

    sentences = ["الكتاب على الطاولة"]
    golds = [gold_for(s) for s in sentences]
    res = clitic_recall(peel_al, sentences, golds)
    assert 0.0 < res["recall"] <= 1.0
    assert 1.0 < res["fertility"] < 3.0


def test_consistency_perfectly_stable_morpheme_has_zero_entropy():
    # char_encoder always splits كتاب into the same four pieces, regardless of host.
    items = [("كتاب", ["الكتاب", "كتابه", "وكتاب", "كتاب"])]
    res = morpheme_consistency(char_encoder, items)
    assert res["mean_top_share"] == 1.0
    assert res["mean_entropy"] == 0.0
    assert res["n_morphemes"] == 1


def test_consistency_smeared_morpheme_has_positive_entropy():
    # This encoder peels ال only, so كتاب's span is a clean "كتاب" in الكتاب/كتاب but is
    # fused with the conjunction in وكتاب — two distinct signatures → entropy > 0.
    def peel_al_word(text: str):
        if text.startswith("ال") and len(text) > 2:
            return [0, 1], [(0, 2), (2, len(text))]
        return [0], [(0, len(text))]

    items = [("كتاب", ["الكتاب", "كتاب", "وكتاب"])]
    res = morpheme_consistency(peel_al_word, items)
    assert res["mean_top_share"] < 1.0
    assert res["mean_entropy"] > 0.0


def test_register_separability_perfect_when_classes_use_disjoint_tokens():
    # MSA sentences contain only 'aaaa…', Masri only 'bbbb…' → linearly separable.
    msa = ["aaaa", "aaa", "aaaaa", "aaaa"]
    masri = ["bbbb", "bbb", "bbbbb", "bbbb"]
    res = register_separability(codepoint_encoder, msa, masri, msa, masri)
    assert res["accuracy"] == 1.0
    assert res["auc"] == 1.0


def test_register_separability_at_chance_when_classes_are_identical():
    # Both registers draw from the same sentences → no signal → ~chance.
    shared = ["abab", "baba", "abba", "baab"]
    res = register_separability(codepoint_encoder, shared, shared, shared, shared)
    assert res["auc"] <= 0.75  # cannot beat chance meaningfully on identical features


def test_predicted_seams_excludes_word_boundaries():
    text = "اب جد"  # two 2-char words
    # token offsets that include a word-initial start (3) and an interior start (1)
    offsets = [(0, 1), (1, 2), (3, 5)]
    _seams_spans = gold_for(text)
    # restrict to a span covering the whole first word so the interior seam survives
    seams = predicted_seams(text, offsets, [(0, 2)])
    assert seams == [1]  # 3 is a word start (excluded); 1 is interior
