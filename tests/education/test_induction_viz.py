"""Unit tests for the stage2c induction visualization helpers.

Everything is tested with numpy arrays and injected fakes — no model,
no tokenizer, no GPU."""

import sys
from pathlib import Path

import numpy as np
import pytest

EDU = Path(__file__).resolve().parents[2] / "notebooks" / "education"
sys.path.insert(0, str(EDU))

from induction_viz import (  # noqa: E402
    attn_heatmap,
    diag_score,
    find_induction_anchor,
    pick_example,
    prefix_matching_score,
)


# ---------------------------------------------------------------- attn_heatmap
def test_attn_heatmap_has_one_real_trace_and_token_ticks():
    pat = np.eye(3)
    tokens = ["Tom", "and", "Tom"]  # duplicates must survive as distinct ticks
    fig = attn_heatmap(pat, tokens)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert list(fig.layout.xaxis.tickvals) == [0, 1, 2]
    assert list(fig.layout.xaxis.ticktext) == tokens
    assert list(fig.layout.yaxis.ticktext) == tokens


def test_attn_heatmap_highlight_draws_one_shape_per_cell():
    fig = attn_heatmap(np.eye(4), list("abcd"), highlight=[(1, 0), (3, 2)])
    assert len(fig.layout.shapes) == 2


def test_attn_heatmap_rejects_token_length_mismatch():
    with pytest.raises(ValueError):
        attn_heatmap(np.eye(3), ["a", "b"])


# ------------------------------------------------------- find_induction_anchor
def test_anchor_finds_first_earlier_occurrence():
    # ids: A B C A  -> last token A at pos 3, first A at pos 0, B1 at pos 1
    assert find_induction_anchor([5, 6, 7, 5]) == (0, 1)


def test_anchor_none_when_last_token_is_new():
    assert find_induction_anchor([5, 6, 7, 8]) is None


def test_anchor_none_when_follower_would_be_the_query_itself():
    # ids: A A -> the token after the first A IS the last position; no target
    assert find_induction_anchor([5, 5]) is None


# ----------------------------------------------------------------- pick_example
def _fake_encode(vocab):
    def encode(text):
        return [vocab[w] for w in text.split()]

    return encode


def _fake_decode(rev):
    def decode_token(tid):
        return rev[tid]

    return decode_token


def test_pick_example_returns_first_passing_candidate():
    vocab = {"<unk>": 0, "Tom": 1, "and": 2, "Lily": 3, "ran": 4}
    rev = {v: k for k, v in vocab.items()}

    def topk5(_ids):
        return [(3, 0.9), (4, 0.05), (2, 0.02), (1, 0.01), (0, 0.01)]

    ex = pick_example(
        ["ran ran ran ran", "Tom and Lily ran Tom and"],
        _fake_encode(vocab),
        _fake_decode(rev),
        topk5,
    )
    # candidate 1: anchor (0,1), target 'ran' (id 4) IS in top-3 -> accepted
    # first, so the picker must return it (first-match semantics).
    assert ex.prompt == "ran ran ran ran"
    assert ex.query_pos == len(ex.ids) - 1
    assert ex.ids[ex.key_pos] == ex.target_id
    assert ex.target_str == rev[ex.target_id]
    assert len(ex.topk) == 5


def test_pick_example_skips_candidate_whose_target_misses_top3():
    vocab = {"Tom": 1, "and": 2, "Lily": 3, "ran": 4, "sat": 5, "dog": 6}
    rev = {v: k for k, v in vocab.items()}

    def topk5(_ids):
        # model always predicts Lily strongly; everything else is noise
        return [(3, 0.9), (6, 0.04), (5, 0.03), (2, 0.02), (1, 0.01)]

    ex = pick_example(
        ["Tom ran sat Tom", "Tom and Lily ran Tom and"],
        _fake_encode(vocab),
        _fake_decode(rev),
        topk5,
    )
    # candidate 1 anchor -> target 'ran' (id 4), not in top-3 -> rejected
    # candidate 2 anchor -> target 'Lily' (id 3), top-1 -> accepted
    assert ex.prompt == "Tom and Lily ran Tom and"
    assert ex.target_str == "Lily"


def test_pick_example_raises_with_all_rejections_named():
    vocab = {"a": 1, "b": 2, "c": 3}
    rev = {v: k for k, v in vocab.items()}

    def topk5(_ids):
        return [(3, 0.9), (3, 0.05), (3, 0.02), (3, 0.01), (3, 0.01)]

    with pytest.raises(ValueError, match="a b c"):
        pick_example(["a b c"], _fake_encode(vocab), _fake_decode(rev), topk5)


# ------------------------------------------------------------------ diag_score
def test_diag_score_is_one_for_perfect_prev_token_head():
    S = 5
    pat = np.zeros((S, S))
    for i in range(1, S):
        pat[i, i - 1] = 1.0
    assert diag_score(pat) == pytest.approx(1.0)


# ------------------------------------------------------ prefix_matching_score
def test_prefix_matching_score_reads_the_stripe():
    # ids: A B A B  -> at t=2 (2nd A), most recent earlier A is s=0, expect
    # attention on s+1=1; at t=3 (2nd B), earlier B at s=1, expect s+1=2.
    ids = [7, 8, 7, 8]
    pat = np.zeros((4, 4))
    pat[2, 1] = 1.0
    pat[3, 2] = 1.0
    assert prefix_matching_score(pat, ids) == pytest.approx(1.0)


def test_prefix_matching_score_zero_when_no_repeats():
    assert prefix_matching_score(np.eye(4), [1, 2, 3, 4]) == 0.0
