"""Unit tests for the honest skip-trigram analysis helpers.

All tests run against a tiny random attn-only model (no checkpoint, no network),
mirroring the FORCE_TINY path in the reference notebook.
"""

import sys
from pathlib import Path

import torch

EDU = Path(__file__).resolve().parents[2] / "notebooks" / "education"
sys.path.insert(0, str(EDU))

import skip_trigrams as st  # noqa: E402
import tiny  # noqa: E402


def _tiny_model(d_vocab=40):
    return tiny.make_tiny_model(
        n_layers=1,
        n_heads=2,
        d_vocab=d_vocab,
        n_ctx=32,
        d_model=64,
        attn_only=True,
        normalization_type=None,
        positional_embedding_type="shortformer",
    )


def test_head_circuits_shapes():
    model = _tiny_model(d_vocab=40)
    QK, OV = st.head_circuits(model, 0)
    assert QK.shape == (40, 40)
    assert OV.shape == (40, 40)


def test_head_attention_kind_detects_bos():
    # all attention on column 0 -> "bos"
    pattern = torch.zeros(5, 5)
    pattern[:, 0] = 1.0
    assert st.head_attention_kind(pattern) == "bos"


def test_head_attention_kind_detects_prev_token():
    # each row attends to the immediately preceding token -> "prev_token"
    pattern = torch.zeros(5, 5)
    for i in range(1, 5):
        pattern[i, i - 1] = 1.0
    pattern[0, 0] = 1.0
    assert st.head_attention_kind(pattern) == "prev_token"


def test_candidate_pool_returns_sorted_scored_triples():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=10)
    assert 0 < len(pool) <= 10
    keys = {"source", "dest", "output", "ov", "qk", "score"}
    assert keys <= set(pool[0])
    scores = [c["score"] for c in pool]
    assert scores == sorted(scores, reverse=True)


def test_candidate_pool_excludes_self_copy_by_default():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=40)
    assert all(c["source"] != c["output"] for c in pool)


def test_seeded_pool_restricts_sources():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=40, sources={3, 7})
    assert {c["source"] for c in pool} <= {3, 7}


def test_seed_ids_resolves_in_vocab_tokens():
    model = _tiny_model(d_vocab=40)  # noqa: F841
    id_to_str = {i: f"t{i}" for i in range(40)}
    encode = lambda s: [int(s[1:])] if s.startswith("t") else []
    ids = st.seed_ids(encode, id_to_str, ["t3", "t7", "zzz"], freq=40)
    assert set(ids) == {3, 7}
