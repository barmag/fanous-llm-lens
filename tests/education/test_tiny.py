"""Fast CPU tests for the Stage 2 shared helper (notebooks/education/tiny.py)."""

import sys
from pathlib import Path

import torch

# tiny.py lives beside the notebooks, not in the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "education"))
import tiny


def test_device_returns_known_string():
    assert tiny.device() in ("cuda", "cpu")


def test_make_tiny_model_attn_only_shapes():
    m = tiny.make_tiny_model(n_layers=1, n_heads=1, d_vocab=50, n_ctx=16, d_model=32)
    assert m.cfg.n_layers == 1 and m.cfg.n_heads == 1 and m.cfg.attn_only is True
    logits = m(torch.randint(0, 50, (2, 16)), return_type="logits")
    assert tuple(logits.shape) == (2, 16, 50)


def test_make_tiny_model_with_mlp():
    m = tiny.make_tiny_model(
        n_layers=2, n_heads=2, d_vocab=40, n_ctx=16, d_model=32, attn_only=False
    )
    assert m.cfg.attn_only is False
    # forward still works with the MLP on
    assert m(torch.randint(0, 40, (1, 16)), return_type="loss").ndim == 0


def test_make_induction_data_second_half_repeats_first():
    data = tiny.make_induction_data(batch=4, seq_len=16, d_vocab=20, seed=0)
    assert tuple(data.shape) == (4, 16)
    half = 16 // 2
    assert torch.equal(data[:, :half], data[:, half:])
    assert int(data.min()) >= 1 and int(data.max()) < 20


def test_make_natural_batches_chunks_and_drops_remainder():
    ids = list(range(35))
    b = tiny.make_natural_batches(ids, n_ctx=16)
    assert tuple(b.shape) == (2, 16)  # 35 // 16 == 2, remainder dropped


class _FakeTokenizer:
    """Minimal stand-in: maps each non-space char to a sparse, non-contiguous id."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002 (matches real signature)
        return [ord(c) for c in text if not c.isspace()]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_make_compact_encoder_remaps_to_dense_small_vocab():
    tok = _FakeTokenizer()
    encode, corpus_ids, id_to_str, d_vocab = tiny.make_compact_encoder(tok, ["abc", "bcd"])
    # 4 distinct chars (a,b,c,d) + [UNK] -> d_vocab 5, dense ids 0..4
    assert d_vocab == 5
    flat = [i for seq in corpus_ids for i in seq]
    assert min(flat) >= 1 and max(flat) < d_vocab
    assert id_to_str[0] == "[UNK]"
    assert id_to_str[encode("a")[0]] == "a"  # round-trips a seen char


def test_make_compact_encoder_maps_unseen_to_unk():
    tok = _FakeTokenizer()
    encode, *_ = tiny.make_compact_encoder(tok, ["abc"])
    assert encode("z") == [0]  # 'z' never appeared in the corpus -> [UNK]


def test_train_reduces_loss():
    m = tiny.make_tiny_model(n_layers=1, n_heads=2, d_vocab=40, n_ctx=16, d_model=32)
    batches = tiny.make_induction_data(batch=8, seq_len=16, d_vocab=40, seed=0)
    losses = tiny.train(m, batches, n_epochs=40, lr=1e-3, seed=0)
    assert len(losses) == 40 and losses[-1] < losses[0]
