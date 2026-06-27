"""Shared corpus + tokenizer helpers for the offline Stage 2dash / 2dash² trainers.

Streaming/cleaning Arabic text, training a small unicode BPE, and caching the
tokenised ids. Extracted from train_stage2dash.py so both trainers reuse one copy."""
from __future__ import annotations

import os
import re

import numpy as np

_AR = re.compile(r"[^\sء-ي]")
_NOISE = re.compile(r"[a-zA-Z0-9_@]+")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _NOISE.sub("", text)
    return _AR.sub("", text)


def build_corpus(char_budget: int, cache_path: str) -> str:
    """Stream Arabic Wikipedia (MSA) + all EG tweets (Masri), clean, concat to
    ~char_budget characters. Cached to disk so re-runs are instant."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            text = f.read()
        print(f"[corpus] cache hit: {len(text):,} chars from {cache_path}")
        return text

    from datasets import load_dataset

    print(f"[corpus] streaming up to {char_budget:,} chars (MSA Wikipedia + EG tweets)...")
    parts: list[str] = []
    total = 0

    # Masri first (small, ~few M chars) — keep all of it so the dialect signal is in.
    tweets = load_dataset("amgadhasan/arabic_tweets_dialects", split="train")
    eg = tweets.filter(lambda x: x["dialect"] == "EG")
    for r in eg:
        c = clean(r["text"])
        if c:
            parts.append(c)
            total += len(c) + 1
    masri_chars = total
    print(f"[corpus] Masri (EG tweets): {masri_chars:,} chars")

    # MSA fills the rest of the budget.
    wiki = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
    for r in wiki:
        if total >= char_budget:
            break
        c = clean(r["text"])
        if c:
            parts.append(c)
            total += len(c) + 1
    text = " ".join(parts)
    print(
        f"[corpus] total: {len(text):,} chars (MSA {len(text) - masri_chars:,} + Masri {masri_chars:,})"
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# --------------------------------------------------------------------------- #
# Tokenizer: small unicode-level BPE -> readable Arabic subword tokens.
# --------------------------------------------------------------------------- #
def train_tokenizer(text: str, vocab_size: int, out_path: str):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if os.path.exists(out_path):
        print(f"[tok] cache hit: {out_path}")
        return Tokenizer.from_file(out_path)

    print(f"[tok] training {vocab_size}-vocab BPE...")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["[UNK]"])
    # chunk the text so the trainer iterates instead of holding one giant string
    chunk = 1_000_000
    tok.train_from_iterator(
        (text[i : i + chunk] for i in range(0, len(text), chunk)), trainer=trainer
    )
    tok.save(out_path)
    print(f"[tok] saved {tok.get_vocab_size()} tokens -> {out_path}")
    return tok


def tokenize(text: str, tok, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        ids = np.load(cache_path)
        print(f"[tok] token cache hit: {len(ids):,} ids from {cache_path}")
        return ids
    print("[tok] encoding corpus (chunked)...")
    # chunk so a multi-GB corpus doesn't build one giant Encoding in memory
    chunks, step = [], 5_000_000
    for i in range(0, len(text), step):
        chunks.append(np.asarray(tok.encode(text[i : i + step]).ids, dtype=np.uint16))
    ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    np.save(cache_path, ids)
    print(f"[tok] {len(ids):,} tokens -> {cache_path}")
    return ids
