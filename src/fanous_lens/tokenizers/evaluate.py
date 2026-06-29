"""Honest, gold-light fitness diagnostics for the candidate tokenizers.

The morpheme gold ([`morphological.py`](morphological.py)) runs on an MSA database and
marks **clitic** boundaries only — not inflection (e.g. it leaves ``يذهبون`` whole rather
than ``يذهب``+``ون``) and it is weak on Masri. Against such an *incomplete* gold you cannot
measure precision: a boundary a tokenizer places where the gold is silent is
indistinguishable from a true boundary the gold simply missed. So an F1 built on that
precision rewards *agreement with the gold's blind spots*, and a ``morphological`` tokenizer
whose vocab IS the gold scores 1.0 by tautology.

This module therefore measures only what the gold can honestly support, plus gold-free
signals:

1. :func:`clitic_recall` — of the clitic boundaries the gold is *confident* about, what
   fraction did the tokenizer place? Reported **with fertility**, because recall alone is
   gamed by over-segmentation (a char-level splitter scores 1.0 by cutting everywhere).
2. :func:`morpheme_consistency` — gold-free: across many host words, does a fixed morpheme
   tokenize the *same* way? A stable morpheme→token mapping is what makes a feature
   localizable, which is the actual interpretability question.

These are **diagnostics, not a verdict.** Whether morpheme-aligned tokenization actually
buys interpretability is a hypothesis only the Phase A probe can settle (see the design
spec); this module is the cheap CPU tier that runs before that.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

from camel_tools.tokenizers.word import simple_word_tokenize

from fanous_lens.tokenizers.morphological import morpheme_boundaries_with_coverage

Encoder = Callable[[str], tuple[list[int], list[tuple[int, int]]]]

# ────────────────────────── shared boundary geometry ────────────────────────


def word_starts(text: str) -> set[int]:
    """Offsets that begin a whitespace-delimited word (never an intra-word seam)."""
    starts: set[int] = set()
    in_word = False
    for i, ch in enumerate(text):
        if ch == " ":
            in_word = False
        else:
            if not in_word:
                starts.add(i)
            in_word = True
    return starts


def gold_for(text: str) -> tuple[list[int], list[tuple[int, int]], int, int]:
    """Confident gold seams for one text, computed once.

    Returns ``(seams, covered_spans, n_words, n_skipped)`` where ``seams`` are intra-word
    clitic boundaries inside words the gold could reconstruct, and ``covered_spans`` are the
    ``(start, end)`` char spans of those words.
    """
    seams, n_words, n_skipped = morpheme_boundaries_with_coverage(text)
    search_from, spans = 0, []
    for w in simple_word_tokenize(text):
        idx = text.find(w, search_from)
        if idx < 0:
            continue
        search_from = idx + len(w)
        _b, _n, sk = morpheme_boundaries_with_coverage(w)
        if sk == 0:
            spans.append((idx, idx + len(w)))
    covered = [s for s in seams if any(a < s < b for a, b in spans)]
    return covered, spans, n_words, n_skipped


def predicted_seams(
    text: str, offsets: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> list[int]:
    """Token-start offsets strictly inside a gold-covered word.

    The Whitespace pre-tokenizer guarantees no token crosses a word boundary, so an
    intra-word seam is exactly a token start that is not a word start.
    """
    ws = word_starts(text)
    cand = {s for s, _e in offsets if s not in ws and 0 < s < len(text)}
    return sorted(s for s in cand if any(a < s < b for a, b in spans))


def greedy_match(pred: list[int], gold: list[int], tol: int = 1) -> int:
    """Greedy one-to-one matches within ±tol; each pred and gold seam consumed once."""
    gold = sorted(gold)
    used = [False] * len(gold)
    matched = 0
    for p in sorted(pred):
        for j, g in enumerate(gold):
            if not used[j] and abs(p - g) <= tol:
                used[j] = True
                matched += 1
                break
    return matched


# ──────────────────────────── clitic recall ─────────────────────────────────


def clitic_recall(
    encode: Encoder,
    sentences: list[str],
    golds: list[tuple[list[int], list[tuple[int, int]], int, int]],
    tol: int = 1,
) -> dict[str, float]:
    """Recall of confident clitic boundaries, reported with fertility.

    ``recall`` is the only directionally-honest boundary score against an incomplete gold:
    *of the boundaries the gold is sure exist, how many did the tokenizer place?* It says
    nothing about boundaries the gold lacks, so it never penalises a tokenizer for splitting
    inflection or Masri morphology the gold cannot see.

    **Must be read with ``fertility``** (tokens/word): recall is trivially maximised by
    cutting everywhere, so a high recall at low fertility is real signal while a high recall
    at high fertility is just over-segmentation. ``beyond_gold_rate`` is the share of a
    tokenizer's intra-word seams that fall where the gold is silent — purely descriptive
    (it may be correct inflectional/dialectal splitting), never scored as error.
    """
    n_tokens = n_words = 0
    sum_matched = sum_gold = sum_pred = 0
    gold_words = gold_skipped = 0
    for text, (gold_seams, spans, nw, sk) in zip(sentences, golds, strict=True):
        _ids, offsets = encode(text)
        n_tokens += len(_ids)
        n_words += len(text.split())
        gold_words += nw
        gold_skipped += sk
        pred = predicted_seams(text, offsets, spans)
        sum_matched += greedy_match(pred, gold_seams, tol)
        sum_gold += len(gold_seams)
        sum_pred += len(pred)
    recall = sum_matched / sum_gold if sum_gold else 0.0
    beyond = (sum_pred - sum_matched) / sum_pred if sum_pred else 0.0
    return {
        "recall": round(recall, 3),
        "fertility": round(n_tokens / n_words, 3) if n_words else 0.0,
        "beyond_gold_rate": round(beyond, 3),
        "gold_seams": sum_gold,
        "coverage": round(1 - gold_skipped / gold_words, 3) if gold_words else 0.0,
    }


# ──────────────────────────── consistency ───────────────────────────────────


def _morph_signature(encode: Encoder, host: str, morpheme: str) -> tuple[str, ...] | None:
    """The token pieces overlapping ``morpheme``'s span inside ``host`` (None if absent)."""
    idx = host.find(morpheme)
    if idx < 0:
        return None
    ms, me = idx, idx + len(morpheme)
    _ids, offsets = encode(host)
    overlap = [host[s:e] for s, e in offsets if s < me and e > ms]
    return tuple(overlap)


def register_separability(
    encode: Encoder,
    msa_train: list[str],
    masri_train: list[str],
    msa_eval: list[str],
    masri_eval: list[str],
) -> dict[str, float]:
    """How accessibly does this tokenization expose the MSA-vs-Masri signal?

    Represents each sentence as a bag of token ids, fits a linear classifier (logistic
    regression) on the train split to predict register, and scores it on the **disjoint**
    eval split. A tokenization under which the dialect signal is more linearly accessible
    scores higher — a zero-GPU proxy for the project's north star (where does the dialect
    signal live).

    **Heavy caveat — a complement, not a verdict.** The MSA corpus is Wikipedia and the Masri
    corpus is tweets, so this conflates *dialect* with *topic / register / lexis*: a high score
    may reflect "Wikipedia vs Twitter vocabulary," not Egyptian morphology. Differences
    *between tokenizers* are the only interpretable signal (same text, same topic confound),
    and even those are weak evidence. The real dialect-localization test is the Phase A probe
    on a balanced feature set.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    def feats(sentences: list[str]) -> list[Counter]:
        return [Counter(str(i) for i in encode(t)[0]) for t in sentences]

    x_train = feats(msa_train) + feats(masri_train)
    y_train = [0] * len(msa_train) + [1] * len(masri_train)
    x_eval = feats(msa_eval) + feats(masri_eval)
    y_eval = [0] * len(msa_eval) + [1] * len(masri_eval)

    vec = DictVectorizer(sparse=True)
    x_train_v = vec.fit_transform(x_train)
    x_eval_v = vec.transform(x_eval)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_v, y_train)
    proba = clf.predict_proba(x_eval_v)[:, 1]
    pred = [1 if p >= 0.5 else 0 for p in proba]
    return {
        "accuracy": round(accuracy_score(y_eval, pred), 3),
        "auc": round(roc_auc_score(y_eval, proba), 3),
    }


def morpheme_consistency(encode: Encoder, items: list[tuple[str, list[str]]]) -> dict[str, float]:
    """Gold-free type-coherence: does a morpheme tokenize the same across host words?

    ``items`` pairs each target morpheme with host words that contain it. For each morpheme
    we collect the token signature of its character span in every host and measure:

    - ``mean_top_share`` — averaged over morphemes, the fraction of hosts in which the
      morpheme takes its single most-common tokenization. 1.0 = perfectly stable.
    - ``mean_entropy`` — averaged Shannon entropy (bits) of the per-morpheme signature
      distribution. 0.0 = one signature everywhere; higher = the morpheme is smeared into
      different pieces depending on its neighbours, so a feature for it cannot be localized.

    **Confounded with fertility.** Like :func:`clitic_recall`, this is maximised by
    over-segmentation: a char-level splitter scores ``top_share=1.0, entropy=0.0`` because it
    cuts everything the same way everywhere. The metric rewards *regularity* and cannot, alone,
    separate "stable because morpheme-aligned" from "stable because finer-grained" — so a high
    score is only meaningful read next to fertility.
    """
    top_shares: list[float] = []
    entropies: list[float] = []
    for morpheme, hosts in items:
        sigs = [s for s in (_morph_signature(encode, h, morpheme) for h in hosts) if s]
        if not sigs:
            continue
        counts = Counter(sigs)
        total = sum(counts.values())
        top_shares.append(max(counts.values()) / total)
        entropies.append(-sum((c / total) * math.log2(c / total) for c in counts.values()))
    n = len(top_shares)
    return {
        "mean_top_share": round(sum(top_shares) / n, 3) if n else 0.0,
        "mean_entropy": round(sum(entropies) / n, 3) if n else 0.0,
        "n_morphemes": n,
    }
