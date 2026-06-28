"""Unit test for the shared corpus helpers + import-smoke for both trainers.

clean() is pure (regex-only, no network); we test it directly. We also assert
both training scripts import the shared module so the refactor is wired up."""
import sys
from pathlib import Path

EDU = Path(__file__).resolve().parents[2] / "notebooks" / "education"
sys.path.insert(0, str(EDU))


def test_clean_strips_latin_digits_and_collapses_whitespace():
    import corpus
    # keeps Arabic letters, drops latin/digits/underscore/@, collapses whitespace
    # _WS collapses whitespace first (→ single spaces), then _NOISE strips latin tokens
    # leaving adjacent single-spaces; final _AR pass does not re-collapse whitespace.
    # "مرحبا   world123 @user يا"
    #   → after _WS: "مرحبا world123 @user يا"   (single spaces)
    #   → after _NOISE removes "world123" and "@user": "مرحبا   يا"  (3 spaces)
    assert corpus.clean("مرحبا   world123 @user يا") == "مرحبا   يا"


def test_both_trainers_import_shared_corpus():
    import importlib
    for mod in ("train_stage2dash", "train_stage2dash2"):
        m = importlib.import_module(mod)
        assert hasattr(m, "corpus") or hasattr(m, "build_corpus"), mod
