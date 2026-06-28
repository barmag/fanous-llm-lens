"""Morphological segmentation using camel-tools."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer
from camel_tools.tokenizers.word import simple_word_tokenize

# Load once, cache globally
_DISAMBIGUATOR = MLEDisambiguator.pretrained("calima-msa-r13")
_TOKENIZER = MorphologicalTokenizer(
    disambiguator=_DISAMBIGUATOR,
    scheme="d3tok",
    split=True,
)


@functools.lru_cache(maxsize=10_000)
def _analyze_single(text: str) -> tuple[str, ...]:
    """Segment a single string into morphemes using camel-tools."""
    words = simple_word_tokenize(text)
    segments = _TOKENIZER.tokenize(words)
    return tuple(segments)


def analyze_morphology(text: str) -> list[str]:
    """Return a list of morpheme tokens for the given Arabic text.

    Falls back to character-level for strings camel-tools cannot parse.
    """
    try:
        result = _analyze_single(text)
        if result:
            return list(result)
    except Exception:
        pass
    # fallback: character-level
    return list(text)


def analyze_batch(texts: Sequence[str]) -> list[list[str]]:
    """Analyze many texts, returning a list of morpheme lists."""
    return [analyze_morphology(t) for t in texts]


def morpheme_boundaries(text: str) -> list[int]:
    """Return character-offset boundaries of each morpheme in the text.

    Each boundary is the start position of the next morpheme.
    The last boundary equals len(text).
    """
    morphs = analyze_morphology(text)
    offsets: list[int] = []
    pos = 0
    for m in morphs:
        pos += len(m)
        offsets.append(pos)
    return offsets
