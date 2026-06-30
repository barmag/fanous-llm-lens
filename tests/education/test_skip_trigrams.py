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

    def encode(s):
        return [int(s[1:])] if s.startswith("t") else []

    ids = st.seed_ids(encode, id_to_str, ["t3", "t7", "zzz"], freq=40)
    assert set(ids) == {3, 7}


def test_verify_triple_reports_lift_and_probabilities():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=5)
    v = st.verify_triple(model, pool[0])
    for key in ("p_full", "p_bigram", "lift", "verified"):
        assert key in v
    assert 0.0 <= v["p_full"] <= 1.0
    assert 0.0 <= v["p_bigram"] <= 1.0
    assert v["lift"] == v["p_full"] - v["p_bigram"]
    assert isinstance(v["verified"], bool)
    assert v["verified"] == (v["lift"] > 0)


def test_verify_pool_runs_topk():
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=10)
    verified = st.verify_pool(model, pool, top_k=3)
    assert len(verified) == 3
    assert all("lift" in v for v in verified)


def test_candidate_pool_expands_outputs_and_dests():
    # one seed source, 3 outputs x 2 dests -> up to 6 distinct (s,d,o) candidates
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(
        model,
        head=0,
        freq=40,
        top_n=100,
        sources={5},
        per_source_outputs=3,
        per_source_dests=2,
    )
    assert 1 < len(pool) <= 6
    assert all(c["source"] == 5 for c in pool)
    triples = {(c["source"], c["dest"], c["output"]) for c in pool}
    assert len(triples) == len(pool)  # no duplicates
    assert all(c["output"] != 5 for c in pool)  # self-copy still forbidden


def test_candidate_pool_default_is_one_per_source():
    # defaults (1,1) preserve the old one-candidate-per-source behaviour
    model = _tiny_model(d_vocab=40)
    pool = st.candidate_pool(model, head=0, freq=40, top_n=100, sources={5, 9})
    assert len(pool) == 2


def test_dedup_triples_keeps_highest_score():
    rows = [
        {"source": 1, "dest": 2, "output": 3, "score": 0.4, "head": 0},
        {"source": 1, "dest": 2, "output": 3, "score": 0.9, "head": 1},
        {"source": 4, "dest": 5, "output": 6, "score": 0.1, "head": 0},
    ]
    out = st.dedup_triples(rows)
    assert len(out) == 2
    kept = next(r for r in out if r["source"] == 1)
    assert kept["score"] == 0.9 and kept["head"] == 1


def test_top_per_group_picks_highest_lift_per_group():
    rows = [
        {"group": "A", "lift": 0.1, "source": 1, "dest": 2, "output": 3},
        {"group": "A", "lift": 0.8, "source": 4, "dest": 5, "output": 6},
        {"group": "B", "lift": 0.3, "source": 7, "dest": 8, "output": 9},
    ]
    out = st.top_per_group(rows, key="group", n=1)
    assert len(out) == 2
    a = next(r for r in out if r["group"] == "A")
    assert a["lift"] == 0.8


def test_triple_table_html_renders_tokens_lift_and_gloss():
    id_to_str = {1: "الرغم", 2: "على", 3: "إلا"}
    rows = [
        {
            "source": 1,
            "dest": 2,
            "output": 3,
            "lift": 0.5,
            "head": 3,
            "group": "MSA",
            "gloss": "fixed على الرغم…إلا frame",
        }
    ]
    out = st.triple_table_html(rows, id_to_str, title="Punchline")
    assert "<table" in out
    assert "الرغم" in out and "إلا" in out
    assert "+0.500" in out
    assert "fixed على الرغم" in out
    assert 'dir="ltr"' in out  # triple keeps [src … dst] -> out ordering
    assert out.index("الرغم") < out.index("إلا")  # source before output in the expr


def test_triple_table_html_empty_returns_panel_not_crash():
    out = st.triple_table_html([], {}, title="Empty", empty_msg="none here")
    assert "<table" not in out
    assert "none here" in out and "Empty" in out
