# Semantic Tokenization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 5 Arabic tokenizers, benchmark them on morpheme alignment and concept consistency, then train an embeddings-only model on the best one and measure probe accuracy for linguistic features.

**Architecture:** Phase B builds tokenizers (morphological-aware, Unigram, WordPiece, Morfessor, BPE), evaluates them on 4 metrics against a gold-standard morphological segmentation, and selects the best. Phase A trains a zero-layer transformer on the best tokenizer and measures 10 linear probes on its embedding space.

**Tech Stack:** camel-tools, sentencepiece, tokenizers (HuggingFace), morfessor, torch, sklearn

## Global Constraints

- Package manager: `uv`
- Formatter: `ruff format` + `ruff check --fix`
- Type checker: `basedpyright`
- Imports: `import fanous_lens as fl`
- Test framework: pytest, CPU-only by default
- Corpus: Wikipedia 20231101.ar (MSA) + amgadhasan/arabic_tweets_dialects filtered to EG (Masri)
- Hardware: ROCm on Strix Halo iGPU for Phase A

---

## Revision Log — 2026-06-29 (methodology fixes, binding)

> Task 1 is already committed (`e4bddf8`). During a status review, three **silent**
> bugs were found — code that runs green but produces meaningless numbers. Loud bugs
> (wrong `morfessor`/`camel-tools` APIs, missing deps) are intentionally **not** patched
> here: the implementer corrects those against reality at runtime, exactly as happened to
> `morphological.py` in Task 1. The draft code blocks below remain as reference; where a
> **REVISION (binding)** callout appears, the callout overrides the draft.
>
> **Bug 1 — gold standard is broken (verified).** Committed `morpheme_boundaries()`
> sums `len(morph)` over d3tok segments. d3tok emits clitic markers (`و+`, `ال+`) and
> drops inter-word whitespace, so offsets don't map to surface chars:
> `"وبالقلم كتب الولد"` (len 17) → bounds end at 19; `"مش عارف"` (len 7) → bounds end at 6.
> Every downstream metric compares against garbage. **Fix = redesign, see Task 1b.**
>
> **Bug 2 — alignment metric is blind to where tokens split.** Draft
> `compute_morpheme_alignment` derives token boundaries from token *count*
> (`len(text)*i//len(tokens)`), so two tokenizers that split into the same number of
> tokens at totally different positions get identical F1 — the metric cannot rank the
> tokenizers it exists to rank. Real char offsets are available (`enc.offsets`, verified).
> **Fix in Task 2 + Task 3 callouts.**
>
> **Bug 3 — morpheme-alignment ranking is circular.** The `morphological` tokenizer's
> vocab is built from camel-tools; the gold standard is *also* camel-tools, so it wins by
> construction. **Fix = exclude it from the alignment ranking (Task 4 callout); the live
> comparison is bpe / unigram / wordpiece / morfessor.**
>
> **Decision — boundary semantics:** gold and token boundaries are compared as
> **intra-word split points only**. Word boundaries are shared by every whitespace
> pre-tokenizer and would inflate F1 trivially; the question is whether a tokenizer splits
> *inside* a word at morpheme seams. This also sidesteps the dropped-whitespace problem.
>
> **Caveat to verify during Task 1b:** d3tok normalizes orthography (e.g. أ→ا, ى→ي), so
> stripped morphemes may not concatenate char-for-char to the surface word. Task 1b must
> assert reconstruction on a normalized surface form, or fall back to per-word search.
>
> **Minor notes (non-binding):**
> - `_load_wikipedia` iterates 50k streamed articles *before* slicing, so even
>   `load_corpora(max_msa=10)` runs for minutes — the Task 1 test will look hung. Add an
>   early break tied to the requested count when this surfaces.
> - Phase A (Task 5) defaults `device="cuda"`; on this ROCm/Strix Halo box that string is
>   correct (ROCm torch reports `cuda`), but heavy GPU sweeps crash the window manager —
>   confirm with the user before grabbing the iGPU, keep runs small.
> - `morfessor` is now installed locally (`morfessor==2.0.6`); its real API is
>   `BaselineModel()` / `train_batch()` / `viterbi_segment()` — the draft's
>   `morfessor.Morfessor()` will fail loudly and self-correct at runtime (not patched here).

---

### Task 1: Create tokenizers module scaffolding

**Files:**
- Create: `src/fanous_lens/tokenizers/__init__.py`
- Create: `src/fanous_lens/tokenizers/corpora.py`
- Create: `src/fanous_lens/tokenizers/morphological.py`
- Test: `tests/test_tokenizers/test_corpora.py`

**Interfaces:**
- Consumes: HuggingFace datasets (wikipedia, arabic_tweets_dialects)
- Produces: `load_corpora() -> tuple[list[str], list[str]]` (MSA sents, Masri sents), `analyze_morphology(text: str) -> list[str]` (morpheme list)

- [ ] **Step 1: Write `__init__.py`**

```python
"""Tokenization experiment module — building and benchmarking Arabic tokenizers."""

from fanous_lens.tokenizers.corpora import load_corpora
from fanous_lens.tokenizers.morphological import analyze_morphology

__all__ = ["load_corpora", "analyze_morphology"]
```

- [ ] **Step 2: Write `corpora.py`**

```python
"""Load MSA and Masri corpora for tokenizer training and evaluation."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator

from datasets import Dataset, load_dataset

CACHE_DIR = Path.home() / ".cache" / "fanous-lens" / "datasets"

def _load_wikipedia() -> list[str]:
    """Load MSA Wikipedia articles, return a list of sentences."""
    ds = load_dataset(
        "wikipedia", "20231101.ar", split="train", streaming=True,
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
        "amgadhasan/arabic_tweets_dialects", split="train", streaming=True,
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
```

- [ ] **Step 3: Write `morphological.py`**

```python
"""Morphological segmentation using camel-tools."""

from __future__ import annotations

import functools
from typing import Sequence

from camel_tools.tokenizer import Tokenizer as CamelTokenizer

# camel-tools provides morphological analysis + tokenization.
# We use its tokenizer which segments into morphemes by default.

@functools.lru_cache(maxsize=10_000)
def _analyze_single(text: str) -> tuple[str, ...]:
    """Segment a single string into morphemes using camel-tools."""
    tok = CamelTokenizer()
    segments = tok.tokenize(text)
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
```

- [ ] **Step 4: Write `tests/test_tokenizers/test_corpora.py`**

```python
"""Tests for corpus loading utilities."""

from __future__ import annotations

import pytest

from fanous_lens.tokenizers.corpora import load_corpora


def test_load_corpora_returns_strings():
    msa, masri = load_corpora(max_msa=10, max_masri=10)
    assert isinstance(msa, list)
    assert isinstance(masri, list)
    assert all(isinstance(s, str) for s in msa)
    assert all(isinstance(s, str) for s in masri)


def test_load_corpora_respects_max():
    msa, masri = load_corpora(max_msa=5, max_masri=3)
    assert len(msa) <= 5
    assert len(masri) <= 3
```

- [ ] **Step 5: Run tests**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/test_tokenizers/test_corpora.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add src/fanous_lens/tokenizers/ tests/test_tokenizers/ && git commit -m "feat: add tokenizers module scaffolding"
```

---

### Task 1b: Fix the gold-standard morpheme boundaries (REVISION — Bug 1)

The committed `morpheme_boundaries()` in `morphological.py` returns surface-misaligned
offsets (verified: overshoots on clitics, undershoots on whitespace). Redesign it to
return **intra-word morpheme split points as surface character offsets**.

**Files:**
- Modify: `src/fanous_lens/tokenizers/morphological.py`
- Modify: `tests/test_tokenizers/test_corpora.py` (or new `test_morphological.py`)

**New contract for `morpheme_boundaries(text) -> list[int]`:**
- Returns sorted character offsets *inside words* where a morpheme seam falls — NOT word
  boundaries, NOT `len(text)`.
- Algorithm: for each surface word (locate its span in `text` by search to get
  `word_start`), get its d3tok segments, **strip the `+`/`_` clitic markers** from each
  segment, take cumulative lengths of the stripped pieces (excluding the final piece,
  which is the word end), and emit `word_start + cum_len` for each interior seam.
- **Reconstruction guard:** assert the stripped pieces concatenate to the word; if not
  (orthographic normalization, e.g. أ→ا), skip that word's seams and log a counter so the
  benchmark can report coverage. Do not emit wrong offsets silently.

**Required test (must encode the bug):**
```python
def test_morpheme_boundaries_are_intra_word_surface_offsets():
    from fanous_lens.tokenizers.morphological import morpheme_boundaries
    text = "وبالقلم كتب الولد"          # clitic-heavy; len == 17
    bounds = morpheme_boundaries(text)
    assert all(0 < b < len(text) for b in bounds)     # never overshoots len, never 0
    assert all(text[b] != " " for b in bounds)        # every seam is inside a word
    assert bounds == sorted(bounds)
```

**Commit:** `fix: gold-standard morpheme boundaries return surface intra-word offsets`

---

### Task 2: Tokenizer training scripts (all 5 approaches)

**Files:**
- Create: `src/fanous_lens/tokenizers/train.py`
- Test: `tests/test_tokenizers/test_train.py`

**Interfaces:**
- Consumes: `load_corpora() -> tuple[list[str], list[str]]`
- Produces: `train_tokenizer(approach: str, corpus: list[str], vocab_size: int) -> dict` (returns tokenizer config), `get_tokenizer(approach: str, path: str) -> callable`

> **REVISION (binding) — Bug 2.** The draft `get_tokenizer` returns a bare
> `text -> list[int]` lambda, discarding the offset information Task 3 needs to rank
> tokenizers. Change the produced encoder contract to expose character offsets:
> `encode(text) -> tuple[list[int], list[tuple[int, int]]]` (ids, per-token `(start, end)`
> surface spans).
> - HF tokenizers (bpe/unigram/wordpiece): take `enc.ids` and `enc.offsets` from a single
>   `tok.encode(text)` (verified working). Do **not** wrap in a lambda that drops `.offsets`.
> - morphological/morfessor: compute offsets from the surface length of each emitted
>   segment (strip d3tok markers first; for morfessor, segments are surface substrings).
> - Keep the simple ids-only path available if convenient, but the offset-capable encoder
>   is the one Task 3 consumes. Update `test_get_tokenizer_encodes` to assert offsets too.

- [ ] **Step 1: Write failing test**

```python
"""Tests for tokenizer training."""

from __future__ import annotations

import pytest

from fanous_lens.tokenizers.train import train_tokenizer, get_tokenizer


def test_train_bpe_returns_config():
    corpus = ["عايز شوية ميه", "احنا طلبة", "مش عارف"]
    config = train_tokenizer("bpe", corpus, vocab_size=200)
    assert "vocab" in config
    assert "merges" in config
    assert len(config["vocab"]) >= 50


def test_train_unigram_returns_config():
    corpus = ["عايز شوية ميه", "احنا طلبة", "مش عارف"]
    config = train_tokenizer("unigram", corpus, vocab_size=200)
    assert "vocab" in config
    assert len(config["vocab"]) >= 50


def test_train_wordpiece_returns_config():
    corpus = ["عايز شوية ميه", "احنا طلبة", "مش عارف"]
    config = train_tokenizer("wordpiece", corpus, vocab_size=200)
    assert "vocab" in config
    assert len(config["vocab"]) >= 50


def test_train_morfessor_returns_vocab():
    corpus = ["عايز شوية ميه", "احنا طلبة", "مش عارف"]
    config = train_tokenizer("morfessor", corpus, vocab_size=200)
    assert "vocab" in config


def test_get_tokenizer_encodes():
    corpus = ["عايز شوية ميه", "احنا طلبة"]
    config = train_tokenizer("bpe", corpus, vocab_size=200)
    tokenize = get_tokenizer("bpe", config)
    ids = tokenize("عايز شوية ميه")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/test_tokenizers/test_train.py -v
```

Expected: FAIL with "function not defined"

- [ ] **Step 3: Write `train.py`**

```python
"""Train tokenizers using all 5 approaches."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.pre_tokenizers import Whitespace


def train_bpe(
    corpus: list[str],
    vocab_size: int = 8_000,
) -> dict[str, Any]:
    """Train a BPE tokenizer, return its serialized config dict."""
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tok.save(f.name)
        config = json.loads(Path(f.name).read_text())
    return config


def train_unigram(
    corpus: list[str],
    vocab_size: int = 8_000,
) -> dict[str, Any]:
    """Train a Unigram (SentencePiece-style) tokenizer."""
    tok = Tokenizer(Unigram())
    tok.pre_tokenizer = Whitespace()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tok.save(f.name)
        config = json.loads(Path(f.name).read_text())
    return config


def train_wordpiece(
    corpus: list[str],
    vocab_size: int = 8_000,
) -> dict[str, Any]:
    """Train a WordPiece tokenizer."""
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tok.save(f.name)
        config = json.loads(Path(f.name).read_text())
    return config


def train_morfessor(
    corpus: list[str],
    vocab_size: int = 8_000,
) -> dict[str, Any]:
    """Train a Morfessor-based tokenizer.

    Uses unsupervised morphological segmentation to discover morphemes,
    then builds a vocabulary from the most frequent segments.
    """
    import morfessor

    # Train Morfessor model on the corpus
    io = morfessor.MorfessorIO()
    training_data = []
    for text in corpus:
        # Morfessor expects tokenized words; split on whitespace
        for word in text.split():
            training_data.append(list(word))

    model = morfessor.Morfessor()
    # Use a small sample for speed
    sample = training_data[:100_000]
    model.load_data(sample)
    model.train()

    # Collect all unique segmentations
    segment_counts: dict[str, int] = {}
    for word_chars in training_data[:50_000]:
        word = "".join(word_chars)
        segmentation = model.segment(word)
        for seg in segmentation:
            seg_str = "".join(seg)
            segment_counts[seg_str] = segment_counts.get(seg_str, 0) + 1

    # Build vocab from most frequent segments
    sorted_segs = sorted(segment_counts.items(), key=lambda x: -x[1])
    special = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab_list = special + [seg for seg, _ in sorted_segs[: vocab_size - len(special)]]

    # Return a config-like dict
    config: dict[str, Any] = {
        "type": "morfessor",
        "vocab": {tok: i for i, tok in enumerate(vocab_list)},
        "model": model,
    }
    return config


def train_tokenizer(
    approach: str,
    corpus: list[str],
    vocab_size: int = 8_000,
) -> dict[str, Any]:
    """Train a tokenizer using the given approach.

    Supported approaches: 'bpe', 'unigram', 'wordpiece', 'morfessor', 'morphological'.
    For 'morphological', the vocabulary is built from camel-tools segmentations.
    """
    if approach == "bpe":
        return train_bpe(corpus, vocab_size)
    elif approach == "unigram":
        return train_unigram(corpus, vocab_size)
    elif approach == "wordpiece":
        return train_wordpiece(corpus, vocab_size)
    elif approach == "morfessor":
        return train_morfessor(corpus, vocab_size)
    elif approach == "morphological":
        return _train_morphological(corpus, vocab_size)
    else:
        raise ValueError(f"Unknown approach: {approach}")


def _train_morphological(
    corpus: list[str],
    vocab_size: int = 8_000,
) -> dict[str, Any]:
    """Build vocabulary from camel-tools morphological segmentations."""
    from fanous_lens.tokenizers.morphological import analyze_batch

    segment_counts: dict[str, int] = {}
    for text in corpus:
        morphs = analyze_batch([text])[0]
        for m in morphs:
            segment_counts[m] = segment_counts.get(m, 0) + 1

    sorted_segs = sorted(segment_counts.items(), key=lambda x: -x[1])
    special = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab_list = special + [seg for seg, _ in sorted_segs[: vocab_size - len(special)]]

    config: dict[str, Any] = {
        "type": "morphological",
        "vocab": {tok: i for i, tok in enumerate(vocab_list)},
    }
    return config


def get_tokenizer(
    approach: str,
    config: dict[str, Any],
) -> Callable[[str], list[int]]:
    """Return an encode function for the given tokenizer config."""
    if approach in ("bpe", "unigram", "wordpiece"):
        # Re-load from the HuggingFace tokenizers JSON
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_str(json.dumps(config))
        return lambda text: tok.encode(text, add_special_tokens=False).ids
    elif approach == "morfessor":
        model = config["model"]
        vocab = config["vocab"]
        unk_id = vocab.get("[UNK]", 0)

        def encode_morfessor(text: str) -> list[int]:
            ids: list[int] = []
            for word in text.split():
                segmentation = model.segment(word)
                for seg in segmentation:
                    seg_str = "".join(seg)
                    ids.append(vocab.get(seg_str, unk_id))
            return ids

        return encode_morfessor
    elif approach == "morphological":
        from fanous_lens.tokenizers.morphological import analyze_morphology

        vocab = config["vocab"]
        unk_id = vocab.get("[UNK]", 0)

        def encode_morph(text: str) -> list[int]:
            morphs = analyze_morphology(text)
            return [vocab.get(m, unk_id) for m in morphs]

        return encode_morph
    else:
        raise ValueError(f"Unknown approach: {approach}")
```

- [ ] **Step 4: Run tests**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/test_tokenizers/test_train.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add src/fanous_lens/tokenizers/train.py tests/test_tokenizers/test_train.py && git commit -m "feat: add tokenizer training for all 5 approaches"
```

---

### Task 3: Evaluation framework

**Files:**
- Create: `src/fanous_lens/tokenizers/evaluate.py`
- Create: `tests/test_tokenizers/test_evaluate.py`

**Interfaces:**
- Consumes: `train_tokenizer`, `get_tokenizer`, `analyze_morphology`
- Produces: `evaluate_tokenizer(tokenize_fn, test_sentences, gold_morphs) -> dict`, `compute_morpheme_alignment(tokenize_fn, morph_boundaries) -> dict`, `compute_concept_consistency(tokenize_fn, feature_patterns) -> dict`

> **REVISION 2 (binding, 2026-06-29) — drop precision/F1 entirely; this SUPERSEDES the
> precision/F1 parts below.** A deeper flaw than Bug 2: the gold marks **clitics only** (not
> inflection — `يذهبون` stays whole) and is weak on Masri, so against this **incomplete** gold
> **precision is unmeasurable** — a boundary where the gold is silent is indistinguishable from
> a true one it missed. An F1 then rewards agreement with the gold's blind spots, and
> `morphological` scores 1.0 by tautology. Morpheme alignment is a **hypothesis** (does it buy
> interpretability?), settled only by the Phase A probe — not a verdict.
> **As-built `evaluate.py`** (commit `84eff21`):
> - `clitic_recall(encode, sentences, golds)` → recall of confident clitic boundaries **+
>   fertility** (recall alone is gamed by over-segmentation) **+ `beyond_gold_rate`**
>   (descriptive: cuts where the gold is silent, never scored as error) **+ coverage**.
>   **No precision, no F1.**
> - `morpheme_consistency(encode, items)` → gold-free `top-share` / `entropy` localizability.
> - Tests use controlled **fake encoders** for exact metric-math assertions (not mocks that
>   only line up by token count). The mock-based draft tests below are obsolete.
> See `docs/reports/tokenizer-comparison.md` for the reframe and results.
>
> **REVISION (binding) — Bugs 1+2 (mechanism, still applies to recall).** The draft metrics
> derive token boundaries from token *count* (`len(text)*i//len(tokens)`) and proportional
> position mapping. Delete that approach entirely — it makes the metric blind to where tokens
> actually split. Instead:
> - `compute_morpheme_alignment` consumes the **offset-capable encoder** from Task 2.
>   Token boundaries = the `end` (equivalently interior `start`) offsets from
>   `encode(text)`, **filtered to intra-word seams** (drop any boundary that lands on/at a
>   whitespace gap). Gold boundaries = the redesigned `morpheme_boundaries` (Task 1b),
>   already intra-word surface offsets. Precision/recall/F1 with the ±1-char tolerance as
>   drafted, but over these real sets.
> - `compute_concept_consistency` locates each feature substring by `text.index(pat)` and
>   maps it to covering tokens via **real offsets** (`start < pat_end and end > pat_start`),
>   not proportional `token_starts`/`token_ends`. Note d3tok markers mean morphological
>   tokens may not substring-match raw patterns like `مش`; match against surface spans.
> - The draft mock-based tests (`test_compute_morpheme_alignment_perfect`, etc.) pass only
>   because the mock's token *count* lines up — they validate nothing about real
>   tokenization. Rewrite them to pass `(ids, offsets)` pairs and a real gold list.

- [ ] **Step 1: Write failing test**

```python
"""Tests for tokenizer evaluation."""

from __future__ import annotations

import pytest

from fanous_lens.tokenizers.evaluate import (
    compute_morpheme_alignment,
    compute_concept_consistency,
    evaluate_tokenizer,
)


def test_compute_morpheme_alignment_perfect():
    # A mock tokenizer that perfectly aligns with morpheme boundaries
    def tokenize(text):
        # Return one token per morpheme: boundaries at [3, 7, 10]
        return [1, 2, 3]

    boundaries = [3, 7, 10]
    result = compute_morpheme_alignment(tokenize, ["abc def ghi"], [boundaries])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_compute_morpheme_alignment_mismatch():
    def tokenize(text):
        # Tokens at wrong boundaries: [0, 5, 10] instead of [3, 7, 10]
        return [1, 2, 3]

    boundaries = [3, 7, 10]
    result = compute_morpheme_alignment(tokenize, ["abc def ghi"], [boundaries])
    assert result["precision"] < 1.0
    assert result["recall"] < 1.0


def test_concept_consistency_high():
    # If negation always maps to the same token, consistency = 1.0
    def tokenize(text):
        # Always token 42 for negation prefix
        return [42, 1, 2]

    result = compute_concept_consistency(
        tokenize, {"negation": ["مش", "ما"]}, ["مش عارف", "ما جاش"]
    )
    assert result["negation"]["consistency"] > 0.9


def test_evaluate_tokenizer_returns_all_metrics():
    def tokenize(text):
        return [1, 2, 3]

    result = evaluate_tokenizer(
        tokenize, ["abc def"], [[3, 7, 10]], ["abc def"]
    )
    assert "morpheme_alignment" in result
    assert "concept_consistency" in result
    assert "efficiency" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/test_tokenizers/test_evaluate.py -v
```

Expected: FAIL with "function not defined"

- [ ] **Step 3: Write `evaluate.py`**

```python
"""Evaluation framework for tokenizer interpretability alignment."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable

import numpy as np


def token_boundary_offsets(
    text: str, token_ids: list[int], tokenize_fn: Callable[[str], list[int]]
) -> list[int]:
    """Recover character offsets where tokens begin/end.

    Uses the tokenize function to split text, then maps each token
    back to its character span in the original text.
    """
    # Simple approach: approximate boundaries by length of each token
    # This is approximate — for exact boundaries we'd need decode.
    # For evaluation we use the token-level approximation.
    # Actually, we can use the tokenizer's pre-tokenize step:
    from tokenizers import Tokenizer as HFTokenizer

    if hasattr(tokenize_fn, "_tok"):
        tok = tokenize_fn._tok
    else:
        return [len(text)]  # fallback

    # Encode with known offsets — use the Rust tokenizer's offset tracking
    enc = tok.encode(text)
    offsets: list[int] = []
    for start, end in enc.offsets:
        offsets.append(end)
    return offsets


def compute_morpheme_alignment(
    tokenize_fn: Callable[[str], list[int]],
    test_sentences: list[str],
    gold_boundaries: list[list[int]],
) -> dict[str, float]:
    """Compute precision, recall, F1 of token boundaries vs morpheme boundaries.

    A token boundary is 'correct' if it falls within ±1 char of a morpheme boundary.
    """
    tp = 0
    fp = 0
    fn = 0

    for text, gold_bounds in zip(test_sentences, gold_boundaries):
        tokens = tokenize_fn(text)
        # Approximate token boundaries
        if len(tokens) == 0:
            continue
        # Use character-length approximation
        token_bounds: list[int] = []
        for i in range(1, len(tokens)):
            # Approximate: each token gets roughly equal character share
            pos = len(text) * i // len(tokens)
            token_bounds.append(pos)
        token_bounds.append(len(text))

        # Count TP, FP, FN
        gold_set = set(gold_bounds)
        tok_set = set(token_bounds)

        for b in tok_set:
            if any(abs(b - g) <= 1 for g in gold_set):
                tp += 1
            else:
                fp += 1
        for g in gold_set:
            if not any(abs(g - t) <= 1 for t in tok_set):
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_concept_consistency(
    tokenize_fn: Callable[[str], list[int]],
    feature_patterns: dict[str, list[str]],
    test_sentences: list[str],
) -> dict[str, dict[str, float]]:
    """Measure how often a linguistic feature maps to the same token(s).

    For each feature (e.g., negation), find occurrences in test sentences,
    record which token ID(s) encode that span, measure consistency.
    """
    results: dict[str, dict[str, float]] = {}

    for feature, patterns in feature_patterns.items():
        feature_token_ids: list[set[int]] = []

        for text in test_sentences:
            tokens = tokenize_fn(text)
            # Find pattern occurrences in text
            for pat in patterns:
                if pat in text:
                    # The token(s) covering this pattern
                    # Approximate: which token indices overlap with pattern?
                    pat_start = text.index(pat)
                    pat_end = pat_start + len(pat)

                    # Map character positions to token indices
                    n_tokens = len(tokens)
                    token_starts = [
                        len(text) * i // n_tokens for i in range(n_tokens)
                    ]
                    token_ends = [
                        len(text) * (i + 1) // n_tokens for i in range(n_tokens)
                    ]

                    covered_ids: set[int] = set()
                    for i, (s, e) in enumerate(zip(token_starts, token_ends)):
                        if s < pat_end and e > pat_start:
                            covered_ids.add(tokens[i])
                    if covered_ids:
                        feature_token_ids.append(covered_ids)

        if not feature_token_ids:
            results[feature] = {"consistency": 1.0, "n_occurrences": 0}
            continue

        # Consistency = how often the most common token set appears
        # Normalize by collapsing to the most common single token
        all_ids: list[int] = []
        for idset in feature_token_ids:
            all_ids.extend(idset)

        if not all_ids:
            results[feature] = {"consistency": 0.0, "n_occurrences": 0}
            continue

        most_common = max(set(all_ids), key=all_ids.count)
        count_most = all_ids.count(most_common)
        consistency = count_most / len(all_ids)

        results[feature] = {
            "consistency": consistency,
            "n_occurrences": len(feature_token_ids),
            "most_common_token": most_common,
        }

    return results


def compute_efficiency(
    tokenize_fn: Callable[[str], list[int]],
    test_sentences: list[str],
) -> dict[str, float]:
    """Compute token-count efficiency metrics."""
    n_tokens = sum(len(tokenize_fn(s)) for s in test_sentences)
    n_chars = sum(len(s) for s in test_sentences)
    return {
        "mean_tokens_per_sentence": n_tokens / len(test_sentences),
        "tokens_per_char": n_tokens / n_chars if n_chars > 0 else 0.0,
    }


def evaluate_tokenizer(
    tokenize_fn: Callable[[str], list[int]],
    test_sentences: list[str],
    gold_morph_boundaries: list[list[int]],
    concept_test_sentences: list[str],
) -> dict[str, Any]:
    """Run all evaluation metrics on a tokenizer.

    Returns a dict with keys: 'morpheme_alignment', 'concept_consistency', 'efficiency'.
    """
    alignment = compute_morpheme_alignment(
        tokenize_fn, test_sentences, gold_morph_boundaries
    )

    feature_patterns = {
        "negation": ["مش", "ما", "لم", "لن"],
        "future": ["سـ", "س", "ه", "حـ"],
        "past_tense": ["ت", "ات", "ت"],
        "plural": ["ين", "ون", "ات", "وا"],
        "feminine": ["ة", "ها", "ات"],
        "possessive": ["ه", "ها", "هم", "نا"],
        "1sg_pronoun": ["أنا", "انا", "أن", "ا"],
        "wh_question": ["إيه", "إمتى", "ازاي", "ليه", "فين", "مين", "كام"],
        "progressive_aspect": ["ب", "بـ", "بت"],
        "demonstrative": ["ده", "دي", "دول", "هذا", "هذه"],
    }

    concept = compute_concept_consistency(
        tokenize_fn, feature_patterns, concept_test_sentences
    )

    efficiency = compute_efficiency(tokenize_fn, test_sentences)

    return {
        "morpheme_alignment": alignment,
        "concept_consistency": concept,
        "efficiency": efficiency,
    }
```

- [ ] **Step 4: Run tests**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/test_tokenizers/test_evaluate.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add src/fanous_lens/tokenizers/evaluate.py tests/test_tokenizers/test_evaluate.py && git commit -m "feat: add evaluation framework for tokenizers"
```

---

### Task 4: Phase B benchmark runner

**Files:**
- Create: `experiments/tokenization-benchmark/run_benchmark.py`
- Create: `experiments/tokenization-benchmark/__init__.py`
- Create: `experiments/tokenization-benchmark/results.py`

**Interfaces:**
- Consumes: `load_corpora`, `train_tokenizer`, `get_tokenizer`, `evaluate_tokenizer`, `analyze_morphology`
- Produces: comparison table (dict of dicts), per-tokenizer plots

> **REVISION (binding) — Bug 3.** The `morphological` tokenizer's vocab and the gold
> standard are both camel-tools, so it wins morpheme-alignment by construction. When
> selecting "best by morpheme F1" (the `avg_f1.idxmax()` step), **exclude `morphological`
> from the alignment ranking** and treat bpe/unigram/wordpiece/morfessor as the live
> comparison. Still report `morphological`'s F1 in the table, but label it as the
> trivial/oracle upper bound, not a competitor. Concept-consistency and efficiency rankings
> may include all five.
>
> **Note — gold coverage.** Surface the Task 1b reconstruction-skip counter in the
> benchmark output (what fraction of words contributed gold seams). A low number means the
> alignment metric saw little signal and the ranking is weak — do not hide it.

- [ ] **Step 1: Write `__init__.py`**

```python
"""Phase B: tokenizer benchmarking experiment."""
```

- [ ] **Step 2: Write `run_benchmark.py`**

```python
"""Run full tokenizer benchmark: train all 5 approaches, evaluate, compare."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from fanous_lens.tokenizers.corpora import load_corpora
from fanous_lens.tokenizers.morphological import (
    analyze_batch,
    morpheme_boundaries,
)
from fanous_lens.tokenizers.train import train_tokenizer, get_tokenizer
from fanous_lens.tokenizers.evaluate import evaluate_tokenizer

APPROACHES = ["bpe", "unigram", "wordpiece", "morfessor", "morphological"]
VOCAB_SIZE = 8_000


def main():
    print("Loading corpora...")
    msa_sents, masri_sents = load_corpora(max_msa=50_000, max_masri=25_000)
    combined = msa_sents + masri_sents

    # Prepare test sets
    test_msa = msa_sents[:500]
    test_masri = masri_sents[:500]

    print("Generating gold-standard morphological boundaries...")
    gold_msa = [morpheme_boundaries(s) for s in test_msa]
    gold_masri = [morpheme_boundaries(s) for s in test_masri]

    # Concept consistency test sentences: use minimal pairs
    pairs_path = (
        Path(__file__).resolve().parents[2]
        / "eval" / "prompts" / "msa-masri-pairs-v1.json"
    )
    with open(pairs_path) as f:
        pairs_data = json.load(f)
    concept_sentences = []
    for p in pairs_data["pairs"]:
        concept_sentences.append(p["msa"])
        concept_sentences.append(p["masri"])

    results: dict[str, dict] = {}

    for approach in APPROACHES:
        print(f"\n=== {approach} ===")
        print("  Training...")
        config = train_tokenizer(approach, combined, vocab_size=VOCAB_SIZE)
        tokenize_fn = get_tokenizer(approach, config)

        print("  Evaluating on MSA...")
        msa_metrics = evaluate_tokenizer(
            tokenize_fn, test_msa, gold_msa, concept_sentences
        )
        print("  Evaluating on Masri...")
        masri_metrics = evaluate_tokenizer(
            tokenize_fn, test_masri, gold_masri, concept_sentences
        )

        results[approach] = {
            "msa": msa_metrics,
            "masri": masri_metrics,
        }

        print(f"  MSA F1: {msa_metrics['morpheme_alignment']['f1']:.3f}")
        print(f"  Masri F1: {masri_metrics['morpheme_alignment']['f1']:.3f}")

    # Build comparison table
    rows = []
    for approach, metrics in results.items():
        for register in ("msa", "masri"):
            m = metrics[register]
            rows.append({
                "approach": approach,
                "register": register,
                "morpheme_f1": m["morpheme_alignment"]["f1"],
                "morpheme_precision": m["morpheme_alignment"]["precision"],
                "morpheme_recall": m["morpheme_alignment"]["recall"],
                "tokens_per_char": m["efficiency"]["tokens_per_char"],
                "negation_consistency": m["concept_consistency"].get("negation", {}).get("consistency", 0),
                "future_consistency": m["concept_consistency"].get("future", {}).get("consistency", 0),
            })

    df = pd.DataFrame(rows)
    print("\n\n=== COMPARISON TABLE ===")
    print(df.to_string())

    # Save results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    with open(out_dir / "full_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}")

    # Determine best approach
    avg_f1 = df.groupby("approach")["morpheme_f1"].mean()
    best = avg_f1.idxmax()
    print(f"\nBest approach by morpheme F1: {best} ({avg_f1[best]:.3f})")
    print("Recommendation: take", best, "into Phase A")

    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `results.py`**

```python
"""Results visualization for tokenizer benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results() -> pd.DataFrame:
    out_dir = Path(__file__).parent / "results"
    return pd.read_csv(out_dir / "benchmark_results.csv")


def plot_morpheme_f1(df: pd.DataFrame | None = None) -> plt.Figure:
    if df is None:
        df = load_results()
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = df.pivot_table(
        index="approach", columns="register", values="morpheme_f1"
    )
    x = range(len(pivot.index))
    ax.bar([i - 0.15 for i in x], pivot["msa"], 0.3, label="MSA", color="#4C72B0")
    ax.bar([i + 0.15 for i in x], pivot["masri"], 0.3, label="Masri", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Morpheme Alignment F1")
    ax.set_title("Tokenizer Morpheme Alignment by Register")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_consistency(df: pd.DataFrame | None = None) -> plt.Figure:
    if df is None:
        df = load_results()
    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot negation consistency across approaches
    pivot = df.pivot_table(
        index="approach", columns="register", values="negation_consistency"
    )
    x = range(len(pivot.index))
    ax.bar([i - 0.15 for i in x], pivot["msa"], 0.3, label="MSA", color="#4C72B0")
    ax.bar([i + 0.15 for i in x], pivot["masri"], 0.3, label="Masri", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Negation Token Consistency")
    ax.set_title("Negation Feature → Same Token?")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig
```

- [ ] **Step 4: Run the benchmark**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run python experiments/tokenization-benchmark/run_benchmark.py
```

Expected: runs all 5 tokenizers, prints comparison table, saves results.

- [ ] **Step 5: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add experiments/tokenization-benchmark/ && git commit -m "feat: add Phase B benchmark runner and results plotting"
```

---

### Task 5: Phase A — Embeddings-only model training

**Files:**
- Create: `experiments/embedding-probes/train_embedding_model.py`
- Create: `experiments/embedding-probes/__init__.py`

**Interfaces:**
- Consumes: best tokenizer from Phase B, corpus
- Produces: trained embedding weight matrix `W_E`, checkpoint files

- [ ] **Step 1: Write `__init__.py`**

```python
"""Phase A: embedding-space probe experiments."""
```

- [ ] **Step 2: Write `train_embedding_model.py`**

```python
"""Train a zero-layer transformer (embeddings only) on MSA+Masri corpus."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fanous_lens.tokenizers.corpora import load_corpora
from fanous_lens.tokenizers.train import train_tokenizer, get_tokenizer


class ZeroLayerTransformer(nn.Module):
    """A transformer with NO attention layers — only embeddings + positional.

    This lets us measure what the *tokenization* alone contributes to
    feature separability in the embedding space.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, max_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie embedding weights with LM head
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        x = self.embed(input_ids) + self.pos(positions)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


def train(
    approach: str,
    vocab_size: int = 8_000,
    d_model: int = 256,
    max_steps: int = 100_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    """Train a zero-layer transformer and save checkpoint."""
    print(f"Loading corpora (device={device})...")
    msa_sents, masri_sents = load_corpora(max_msa=50_000, max_masri=25_000)
    combined = msa_sents + masri_sents

    print(f"Training {approach} tokenizer...")
    config = train_tokenizer(approach, combined, vocab_size=vocab_size)
    tokenize_fn = get_tokenizer(approach, config)

    print("Tokenizing corpus...")
    all_ids: list[int] = []
    for sent in combined:
        ids = tokenize_fn(sent)
        all_ids.extend(ids)
    all_ids = all_ids[:1_000_000]  # cap for memory

    data = torch.tensor(all_ids, dtype=torch.long, device=device)

    model = ZeroLayerTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    n_tokens = len(data)
    steps_per_epoch = max(1, n_tokens // batch_size)

    print(f"Training for {max_steps} steps ({steps_per_epoch} steps/epoch)...")
    step = 0
    for epoch in range(100):
        for i in range(0, n_tokens - batch_size, batch_size):
            if step >= max_steps:
                break
            batch = data[i : i + batch_size]
            # Predict next token: input = batch[:-1], target = batch[1:]
            inp = batch[:-1].unsqueeze(0)
            tgt = batch[1:].unsqueeze(0)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 1000 == 0:
                print(f"  step {step}/{max_steps}  loss={loss.item():.4f}")
            step += 1

        if step >= max_steps:
            break

    # Save
    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"{approach}_zerolayer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "d_model": d_model,
        "approach": approach,
        "config": config,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # Also save embedding matrix separately for probing
    emb_path = out_dir / f"{approach}_embeddings.pt"
    torch.save(model.embed.weight.detach().cpu(), emb_path)
    print(f"Embeddings saved: {emb_path}")

    return ckpt_path


if __name__ == "__main__":
    import sys
    approach = sys.argv[1] if len(sys.argv) > 1 else "morphological"
    train(approach)
```

- [ ] **Step 3: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add experiments/embedding-probes/ && git commit -m "feat: add Phase A embeddings-only model training"
```

---

### Task 6: Phase A — Probe evaluation

**Files:**
- Create: `experiments/embedding-probes/run_probes.py`
- Create: `tests/test_tokenizers/test_probes.py`

**Interfaces:**
- Consumes: trained embeddings from Task 5, tokenizer config
- Produces: probe accuracy table, PCA visualization

- [ ] **Step 1: Write failing test**

```python
"""Tests for probe evaluation."""

from __future__ import annotations

import pytest
import torch

from fanous_lens.tokenizers.evaluate import compute_concept_consistency


def test_probe_accuracy_basic():
    # Mock embeddings: 10 tokens, 2 features
    embeddings = torch.randn(10, 4)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=100)
    clf.fit(embeddings.numpy(), labels.numpy())
    acc = clf.score(embeddings.numpy(), labels.numpy())
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Write `run_probes.py`**

```python
"""Train linear probes on embedding space to detect linguistic features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fanous_lens.tokenizers.train import get_tokenizer

# Linguistic features and their trigger patterns
FEATURE_PATTERNS: dict[str, list[str]] = {
    "negation": ["مش", "ما", "لم", "لن", "مش", "ما"],
    "future": ["سـ", "س", "ه", "حـ", "هي"],
    "past_tense": ["ت", "ات", "ت"],
    "plural": ["ين", "ون", "ات", "وا"],
    "feminine": ["ة", "ها", "ات"],
    "dialect_msa": [],  # Will use MSA sentences as positive
    "wh_question": ["إيه", "إمتى", "ازاي", "ليه", "فين", "مين", "كام"],
    "progressive_aspect": ["ب", "بـ", "بت"],
    "1sg_pronoun": ["أنا", "انا"],
    "possessive": ["ه", "ها", "هم", "نا"],
}

# Control probes (should be at chance)
CONTROL_PATTERNS: dict[str, list[str]] = {
    "token_length_short": [],  # tokens with <3 chars
    "token_freq_high": [],     # tokens appearing >100 times
}


def load_embeddings(approach: str) -> tuple[torch.Tensor, dict]:
    """Load trained embeddings and tokenizer config."""
    ckpt_dir = Path(__file__).parent / "checkpoints"
    emb_path = ckpt_dir / f"{approach}_embeddings.pt"
    config_path = ckpt_dir / f"{approach}_config.json"

    embeddings = torch.load(emb_path, map_location="cpu")
    with open(config_path) as f:
        config = json.load(f)

    return embeddings, config


def build_probe_dataset(
    embeddings: torch.Tensor,
    tokenize_fn,
    test_sentences: list[str],
    feature_patterns: dict[str, list[str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build (X, y) for each feature probe.

    X: embedding vectors for each token position
    y: binary label (1 = token carries this feature, 0 = not)
    """
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    all_token_embeds: list[torch.Tensor] = []
    all_token_labels: dict[str, list[int]] = {f: [] for f in feature_patterns}

    for text in test_sentences:
        ids = tokenize_fn(text)
        if not ids:
            continue
        token_embeds = embeddings[ids]  # shape: (n_tokens, d_model)

        for i, tok_id in enumerate(ids):
            all_token_embeds.append(token_embeds[i])

            for feat, patterns in feature_patterns.items():
                if feat == "dialect_msa":
                    # Label based on whether text is MSA (has Arabic diacritics/formal patterns)
                    # Simple heuristic: check for MSA markers
                    is_msa = any(p in text for p in ["أ", "إن", "لـ", "هذا", "هذه"])
                    all_token_labels[feat].append(1 if is_msa else 0)
                elif feat in ("token_length_short", "token_freq_high"):
                    # Control: label based on token properties
                    if feat == "token_length_short":
                        is_short = len(text.split()[i if i < len(text.split()) else -1]) < 3
                        all_token_labels[feat].append(1 if is_short else 0)
                    else:
                        all_token_labels[feat].append(0)
                else:
                    # Feature: check if any pattern appears in the text
                    has_feature = any(p in text for p in patterns)
                    all_token_labels[feat].append(1 if has_feature else 0)

    X = torch.stack(all_token_embeds).numpy()
    for feat, labels in all_token_labels.items():
        y = np.array(labels)
        datasets[feat] = (X, y)

    return datasets


def train_probe(X: np.ndarray, y: np.ndarray) -> tuple[float, LogisticRegression]:
    """Train a logistic regression probe, return (ROC-AUC, classifier)."""
    clf = LogisticRegression(max_iter=500, solver="liblinear")
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    return auc, clf


def main(approach: str = "morphological"):
    print(f"Loading embeddings for {approach}...")
    embeddings, config = load_embeddings(approach)
    tokenize_fn = get_tokenizer(approach, config)

    # Load test sentences
    pairs_path = (
        Path(__file__).resolve().parents[2]
        / "eval" / "prompts" / "msa-masri-pairs-v1.json"
    )
    with open(pairs_path) as f:
        pairs_data = json.load(f)

    test_sentences = []
    for p in pairs_data["pairs"]:
        test_sentences.append(p["msa"])
        test_sentences.append(p["masri"])

    print("Building probe datasets...")
    datasets = build_probe_dataset(
        embeddings, tokenize_fn, test_sentences, FEATURE_PATTERNS
    )

    results: dict[str, float] = {}
    for feat, (X, y) in datasets.items():
        if len(np.unique(y)) < 2:
            results[feat] = 0.0
            continue
        auc, clf = train_probe(X, y)
        results[feat] = auc
        print(f"  {feat:>20s}  AUC={auc:.3f}")

    # Control probes
    control_results: dict[str, float] = {}
    for feat, (X, y) in datasets.items():
        if feat in CONTROL_PATTERNS:
            control_results[feat] = results.get(feat, 0.0)

    # Save results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{approach}_probe_results.json", "w") as f:
        json.dump({
            "approach": approach,
            "feature_probes": results,
            "control_probes": control_results,
        }, f, indent=2)

    print(f"\nResults saved to {out_dir}/{approach}_probe_results.json")
    return results


if __name__ == "__main__":
    approach = sys.argv[1] if len(sys.argv) > 1 else "morphological"
    main(approach)
```

- [ ] **Step 3: Run probe tests**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/test_tokenizers/test_probes.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add experiments/embedding-probes/run_probes.py tests/test_tokenizers/test_probes.py && git commit -m "feat: add probe evaluation for embedding-space features"
```

---

### Task 7: Install dependencies and verify

- [ ] **Step 1: Install camel-tools**

```bash
uv pip install camel-tools
```

- [ ] **Step 2: Install morfessor**

```bash
uv pip install morfessor
```

- [ ] **Step 3: Verify imports**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run python -c "
from fanous_lens.tokenizers import load_corpora, analyze_morphology
from fanous_lens.tokenizers.train import train_tokenizer, get_tokenizer
from fanous_lens.tokenizers.evaluate import evaluate_tokenizer
print('All imports OK')
"
```

Expected: no errors

- [ ] **Step 4: Run all tokenizer tests**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run pytest tests/test_tokenizers/ -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add pyproject.toml && git commit -m "chore: add camel-tools and morfessor dependencies"
```

---

### Task 8: Lint and type-check final pass

- [ ] **Step 1: Run ruff format**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run ruff format src/fanous_lens/tokenizers/ experiments/tokenization-benchmark/ experiments/embedding-probes/
```

- [ ] **Step 2: Run ruff check**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run ruff check --fix src/fanous_lens/tokenizers/ experiments/tokenization-benchmark/ experiments/embedding-probes/
```

- [ ] **Step 3: Run basedpyright**

```bash
cd /home/yassermakram/code/fanous-llm-lens && uv run basedpyright src/fanous_lens/tokenizers/
```

- [ ] **Step 4: Commit**

```bash
cd /home/yassermakram/code/fanous-llm-lens && git add -A && git commit -m "chore: lint and type-check tokenization experiment code"
```
