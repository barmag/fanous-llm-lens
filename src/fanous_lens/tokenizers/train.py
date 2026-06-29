"""Train the five candidate Arabic tokenizers and build offset-capable encoders.

Approaches: ``bpe``, ``unigram``, ``wordpiece`` (HuggingFace ``tokenizers``),
``morfessor`` (unsupervised morphology), ``morphological`` (camel-tools d3tok).

Every encoder returned by :func:`get_tokenizer` has the signature
``encode(text) -> tuple[list[int], list[tuple[int, int]]]`` — ids plus per-token
surface ``(start, end)`` spans — so the evaluation layer can score real token
boundaries against the morpheme gold standard (plan revision, Bug 2).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.pre_tokenizers import Whitespace

SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

Encoder = Callable[[str], tuple[list[int], list[tuple[int, int]]]]

# ──────────────────────────── HuggingFace models ────────────────────────────


def _train_hf(model, trainer, corpus: list[str]) -> dict[str, Any]:
    """Train a HuggingFace tokenizer and return its serialized JSON config dict."""
    tok = Tokenizer(model)
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(corpus, trainer=trainer)
    return json.loads(tok.to_str())


def train_bpe(corpus: list[str], vocab_size: int = 8_000) -> dict[str, Any]:
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=1
    )
    return _train_hf(BPE(unk_token="[UNK]"), trainer, corpus)


def train_unigram(corpus: list[str], vocab_size: int = 8_000) -> dict[str, Any]:
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, unk_token="[UNK]"
    )
    return _train_hf(Unigram(), trainer, corpus)


def train_wordpiece(corpus: list[str], vocab_size: int = 8_000) -> dict[str, Any]:
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=1
    )
    return _train_hf(WordPiece(unk_token="[UNK]"), trainer, corpus)


# ─────────────────────────────── Morfessor ──────────────────────────────────


def train_morfessor(corpus: list[str], vocab_size: int = 8_000) -> dict[str, Any]:
    """Train a Morfessor Baseline model and build a vocab of surface morphs."""
    import morfessor

    word_counts: Counter[str] = Counter()
    for text in corpus:
        word_counts.update(text.split())

    model = morfessor.BaselineModel()
    # load_data expects (count, atoms) tuples; atoms are the splittable units (chars).
    model.load_data([(count, tuple(word)) for word, count in word_counts.items()])
    model.train_batch()

    morph_counts: Counter[str] = Counter()
    for word in word_counts:
        morphs, _cost = model.viterbi_segment(word)
        morph_counts.update(morphs)

    vocab = _build_vocab(morph_counts, vocab_size)
    return {"type": "morfessor", "vocab": vocab, "model": model}


# ───────────────────────────── Morphological ────────────────────────────────


def _train_morphological(corpus: list[str], vocab_size: int = 8_000) -> dict[str, Any]:
    """Build a vocab from camel-tools surface morpheme pieces (markers stripped)."""
    from fanous_lens.tokenizers.morphological import analyze_with_offsets

    morph_counts: Counter[str] = Counter()
    for text in corpus:
        pieces, _offsets = analyze_with_offsets(text)
        morph_counts.update(pieces)

    vocab = _build_vocab(morph_counts, vocab_size)
    return {"type": "morphological", "vocab": vocab}


def _build_vocab(counts: Counter[str], vocab_size: int) -> dict[str, int]:
    """Specials first, then the most frequent segments up to ``vocab_size``."""
    keep = [seg for seg, _ in counts.most_common(vocab_size - len(SPECIAL_TOKENS))]
    return {tok: i for i, tok in enumerate(SPECIAL_TOKENS + keep)}


# ──────────────────────────────── Dispatch ──────────────────────────────────


def train_tokenizer(approach: str, corpus: list[str], vocab_size: int = 8_000) -> dict[str, Any]:
    """Train a tokenizer with the named approach; returns a serializable config."""
    trainers_by_name = {
        "bpe": train_bpe,
        "unigram": train_unigram,
        "wordpiece": train_wordpiece,
        "morfessor": train_morfessor,
        "morphological": _train_morphological,
    }
    if approach not in trainers_by_name:
        raise ValueError(f"Unknown approach: {approach!r}")
    return trainers_by_name[approach](corpus, vocab_size)


def get_tokenizer(approach: str, config: dict[str, Any]) -> Encoder:
    """Return an ``encode(text) -> (ids, offsets)`` function for the config."""
    if approach in ("bpe", "unigram", "wordpiece"):
        return _hf_encoder(config)
    if approach == "morfessor":
        return _morfessor_encoder(config)
    if approach == "morphological":
        return _morphological_encoder(config)
    raise ValueError(f"Unknown approach: {approach!r}")


def _hf_encoder(config: dict[str, Any]) -> Encoder:
    tok = Tokenizer.from_str(json.dumps(config))

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = tok.encode(text, add_special_tokens=False)
        return list(enc.ids), [(s, e) for s, e in enc.offsets]

    return encode


def _morfessor_encoder(config: dict[str, Any]) -> Encoder:
    model = config["model"]
    vocab: dict[str, int] = config["vocab"]
    unk_id = vocab["[UNK]"]

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in text.split():
            start = text.find(word, cursor)
            cursor = start + len(word)
            morphs, _cost = model.viterbi_segment(word)
            pos = start
            for morph in morphs:
                ids.append(vocab.get(morph, unk_id))
                offsets.append((pos, pos + len(morph)))
                pos += len(morph)
        return ids, offsets

    return encode


def _morphological_encoder(config: dict[str, Any]) -> Encoder:
    from fanous_lens.tokenizers.morphological import analyze_with_offsets

    vocab: dict[str, int] = config["vocab"]
    unk_id = vocab["[UNK]"]

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        pieces, offsets = analyze_with_offsets(text)
        ids = [vocab.get(p, unk_id) for p in pieces]
        return ids, offsets

    return encode
