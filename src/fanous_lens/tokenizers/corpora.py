"""Load MSA and Masri corpora for tokenizer training and evaluation."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

CACHE_DIR = Path.home() / ".cache" / "fanous-lens" / "datasets"


def _load_wikipedia() -> list[str]:
    """Load MSA Wikipedia articles, return a list of sentences."""
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ar",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )
    sentences: list[str] = []
    for i, article in enumerate(ds):
        text = article["text"]
        # Simple sentence splitting on Arabic punctuation
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) > 20:
                sentences.append(stripped)
        if i >= 50_000:
            break
    return sentences


def _load_tweets_dialects(dialect: str = "EG") -> list[str]:
    """Load Egyptian-dialect tweets."""
    ds = load_dataset(
        "amgadhasan/arabic_tweets_dialects",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )
    sentences: list[str] = []
    for i, row in enumerate(ds):
        if row["dialect"] == dialect:
            text = row["text"].strip()
            if text and len(text) > 10:
                sentences.append(text)
        if i >= 100_000:
            break
    return sentences


def load_corpora(
    max_msa: int = 100_000,
    max_masri: int = 50_000,
) -> tuple[list[str], list[str]]:
    """Return (msa_sentences, masri_sentences) lists."""
    msa = _load_wikipedia()[:max_msa]
    masri = _load_tweets_dialects("EG")[:max_masri]
    return msa, masri
