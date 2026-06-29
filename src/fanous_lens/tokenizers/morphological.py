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


# d3tok marks proclitics with a trailing '+', enclitics with a leading '+';
# some schemes use '_'. Stripping these recovers the surface morpheme piece.
_CLITIC_MARKERS = "+_"


def _strip_markers(segment: str) -> str:
    """Remove d3tok clitic markers, leaving the surface character content."""
    return segment.strip(_CLITIC_MARKERS)


@functools.lru_cache(maxsize=10_000)
def _analyze_single(text: str) -> tuple[str, ...]:
    """Segment a single string into morphemes using camel-tools."""
    words = simple_word_tokenize(text)
    segments = _TOKENIZER.tokenize(words)
    return tuple(segments)


@functools.lru_cache(maxsize=10_000)
def _segment_word(word: str) -> tuple[str, ...]:
    """Return d3tok morpheme segments for a single surface word (markers kept)."""
    return tuple(_TOKENIZER.tokenize([word]))


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


def morpheme_boundaries_with_coverage(text: str) -> tuple[list[int], int, int]:
    """Return (intra-word morpheme seams, n_words, n_skipped_words).

    A seam is the surface character offset where one morpheme ends and the next
    begins *inside a word*. Word boundaries (whitespace) are intentionally excluded:
    every whitespace pre-tokenizer shares them, so they carry no ranking signal.

    d3tok carries clitic markers and normalizes orthography. For each word we strip
    the markers, then verify the pieces concatenate back to the surface word. If they
    do not (e.g. an alef-hamza was normalized to a bare alef, changing a character),
    we skip that word's seams rather than emit misaligned offsets, and count it toward
    ``n_skipped``.
    """
    words = simple_word_tokenize(text)
    boundaries: list[int] = []
    n_skipped = 0
    search_from = 0
    for word in words:
        idx = text.find(word, search_from)
        if idx < 0:
            # Word not found verbatim in surface text (normalization/tokenization
            # mismatch); cannot align seams safely.
            n_skipped += 1
            continue
        word_start = idx
        search_from = idx + len(word)

        pieces = [p for p in (_strip_markers(s) for s in _segment_word(word)) if p]
        if "".join(pieces) != word:
            # Orthographic normalization changed characters — offsets would be wrong.
            n_skipped += 1
            continue

        pos = word_start
        for piece in pieces[:-1]:  # last piece ends at the word boundary, not a seam
            pos += len(piece)
            boundaries.append(pos)

    return boundaries, len(words), n_skipped


def morpheme_boundaries(text: str) -> list[int]:
    """Return intra-word morpheme seams as surface character offsets.

    See :func:`morpheme_boundaries_with_coverage` for the full contract; this is the
    convenience form that drops the coverage counters.
    """
    boundaries, _, _ = morpheme_boundaries_with_coverage(text)
    return boundaries


def analyze_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Surface morpheme pieces for ``text`` with their ``(start, end)`` char spans.

    This is the encoder-facing companion to :func:`morpheme_boundaries`: it returns
    the stripped surface pieces (markers removed) plus full spans for every piece,
    so a morphological tokenizer can emit ids *and* offsets. A word whose stripped
    segments fail to reconstruct the surface form (orthographic assimilation) is
    emitted as a single whole-word token, keeping every character covered and offsets
    exact for the words that do reconstruct.
    """
    words = simple_word_tokenize(text)
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    search_from = 0
    for word in words:
        idx = text.find(word, search_from)
        if idx < 0:
            continue
        word_start = idx
        search_from = idx + len(word)

        surface = [p for p in (_strip_markers(s) for s in _segment_word(word)) if p]
        if "".join(surface) == word:
            pos = word_start
            for piece in surface:
                pieces.append(piece)
                offsets.append((pos, pos + len(piece)))
                pos += len(piece)
        else:
            # Cannot align sub-word seams; keep the whole word as one token.
            pieces.append(word)
            offsets.append((word_start, word_start + len(word)))

    return pieces, offsets
