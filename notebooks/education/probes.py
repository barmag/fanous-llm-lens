"""probes.py — shared helper for the probing-methods ladder (rung probe_a onward).

Self-contained companion to tiny.py: where tiny.py builds tiny *models*, this
builds tiny *probes* — the logistic-regression readouts and their controls that
tell you whether a feature is linearly recoverable from a representation.

Depends only on numpy, torch, and scikit-learn (plus `datasets` for the corpus
loader, matching corpus.py). Delivered to Colab by wget of this single file;
imported locally as a sibling module.

The zero-layer model here is the same one Phase A used
(experiments/embedding-probes/train_embedding_model.py): embeddings + positional
+ LayerNorm + tied head, *no attention, no MLP*. All learnable structure lives in
W_E, so a probe on its pooled embeddings measures what the **tokenization** makes
available — which is exactly rung probe_a's lesson.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_SEED = 0


# --------------------------------------------------------------------------- #
# 1. Corpus: register-labelled sentences (MSA vs Masri).
#    Dialect is a *sentence-level* label — no morphological gold needed — so it
#    is the cleanest feature to teach "what is a probe" on.
# --------------------------------------------------------------------------- #
def load_dialect_corpus(
    max_msa: int = 20_000, max_masri: int = 10_000, cache_path: str | None = None
) -> dict:
    """Stream MSA (Arabic Wikipedia) + Masri (EG tweets); return labelled sentences.

    Mirrors src/fanous_lens/tokenizers/corpora.load_corpora but self-contained so
    the ladder notebooks stay wget-portable. Returns::

        {"sentences": [str, ...], "register": np.ndarray[int]  # 1 = Masri, 0 = MSA}

    Masri is label 1 (the register the project cares about). Registers are
    interleaved by the caller if a shuffle is wanted (do it with a fixed seed).

    If ``cache_path`` is given, the (streamed, slow) corpus is written there as JSON
    on the first pass and reloaded instantly on every re-run — the notebook's
    idempotent <10-min bar.
    """
    import json
    import os

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            blob = json.load(f)
        print(f"[corpus] cache hit: {len(blob['sentences']):,} sentences from {cache_path}")
        return {"sentences": blob["sentences"], "register": np.array(blob["register"], dtype=int)}

    from datasets import load_dataset

    msa: list[str] = []
    wiki = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
    for article in wiki:
        for line in article["text"].split("\n"):
            s = line.strip()
            if len(s) > 20:
                msa.append(s)
                if len(msa) >= max_msa:
                    break
        if len(msa) >= max_msa:
            break

    masri: list[str] = []
    tweets = load_dataset("amgadhasan/arabic_tweets_dialects", split="train", streaming=True)
    for row in tweets:
        if row["dialect"] == "EG":
            s = row["text"].strip()
            if len(s) > 10:
                masri.append(s)
                if len(masri) >= max_masri:
                    break

    sentences = msa + masri
    register = np.array([0] * len(msa) + [1] * len(masri), dtype=int)
    if cache_path:
        import json
        import os

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"sentences": sentences, "register": register.tolist()}, f, ensure_ascii=False
            )
        print(f"[corpus] cached {len(sentences):,} sentences -> {cache_path}")
    return {"sentences": sentences, "register": register}


# --------------------------------------------------------------------------- #
# 2. The zero-layer model — embeddings only (no attention, no MLP).
# --------------------------------------------------------------------------- #
class ZeroLayerModel(nn.Module):
    """embed + positional + LayerNorm + tied LM head. All structure lives in W_E."""

    def __init__(self, vocab_size: int, d_model: int = 256, max_len: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions)[None, :, :]
        return self.lm_head(self.ln(x))

    @property
    def w_e(self) -> torch.Tensor:
        """The embedding table [vocab, d_model], detached on CPU — the probe's input."""
        return self.embed.weight.detach().cpu()


@dataclass
class ZeroLayerConfig:
    vocab_size: int
    d_model: int = 256
    seq_len: int = 128
    batch_size: int = 64
    max_steps: int = 3_000
    lr: float = 1e-3
    seed: int = DEFAULT_SEED
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def train_zero_layer(
    cfg: ZeroLayerConfig, token_ids: list[int], log_every: int = 0, cache_path: str | None = None
) -> dict:
    """Train a ZeroLayerModel on a flat id stream; return {model, losses, final_loss}.

    Sampled-batch SGD over next-token prediction. `token_ids` is one long list of
    compact ids (see make_compact_encoder in tiny.py). Fast on the iGPU; the point
    is only to *reshape W_E*, not to reach a strong LM.

    If ``cache_path`` is given, a matching checkpoint (same vocab/d_model) is loaded
    instead of retraining; otherwise training runs and the result is saved there.
    """
    import os

    if cache_path and os.path.exists(cache_path):
        ckpt = torch.load(cache_path, map_location=cfg.device, weights_only=False)
        if ckpt["vocab_size"] == cfg.vocab_size and ckpt["d_model"] == cfg.d_model:
            model = ZeroLayerModel(cfg.vocab_size, cfg.d_model, max_len=cfg.seq_len).to(cfg.device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[model] cache hit: final_loss={ckpt['final_loss']:.3f} from {cache_path}")
            return {"model": model, "losses": ckpt["losses"], "final_loss": ckpt["final_loss"]}
        print(f"[model] cache mismatch (vocab/d_model changed) — retraining {cache_path}")

    torch.manual_seed(cfg.seed)
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    n_windows = ids.shape[0] // cfg.seq_len
    if n_windows < 2:
        raise ValueError(f"corpus too small: {ids.shape[0]} ids < {2 * cfg.seq_len}")
    windows = ids[: n_windows * cfg.seq_len].reshape(n_windows, cfg.seq_len)

    model = ZeroLayerModel(cfg.vocab_size, cfg.d_model, max_len=cfg.seq_len).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    g = torch.Generator().manual_seed(cfg.seed)
    losses: list[float] = []
    model.train()
    for step in range(cfg.max_steps):
        idx = torch.randint(0, n_windows, (cfg.batch_size,), generator=g)
        batch = windows[idx].to(cfg.device)
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and ((step + 1) % log_every == 0 or (step + 1) == cfg.max_steps):
            print(f"  step {step + 1:>5}/{cfg.max_steps}  loss={losses[-1]:.3f}")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab_size": cfg.vocab_size,
                "d_model": cfg.d_model,
                "losses": losses,
                "final_loss": losses[-1],
            },
            cache_path,
        )
        print(f"[model] cached -> {cache_path}")
    return {"model": model, "losses": losses, "final_loss": losses[-1]}


# --------------------------------------------------------------------------- #
# 3. Pooling: turn a text into one vector by averaging its token embeddings.
#    A (near-random at init) linear projection of the text's bag of tokens —
#    which is exactly why the zero-layer probe scores the *tokenization*.
# --------------------------------------------------------------------------- #
def pooled_embeddings(w_e: torch.Tensor, encode, texts: list[str]) -> np.ndarray:
    """Mean token embedding per text -> [n_texts, d_model] float32 (zeros if empty)."""
    d = w_e.shape[1]
    rows = np.zeros((len(texts), d), dtype=np.float32)
    for i, t in enumerate(texts):
        ids = encode(t)
        ids = [j for j in ids if 0 <= j < w_e.shape[0]]
        if ids:
            rows[i] = w_e[ids].mean(dim=0).numpy()
    return rows


# --------------------------------------------------------------------------- #
# 4. The probe itself + its controls.
# --------------------------------------------------------------------------- #
def probe_auc(
    x: np.ndarray, y: np.ndarray, seed: int = DEFAULT_SEED, min_per_class: int = 30
) -> float | None:
    """Held-out logistic-probe ROC-AUC with StandardScaler. None if a class too small.

    StandardScaler is load-bearing once features differ in scale (Phase-A-depth
    §1); logistic regression + a 30% stratified holdout is the standard probe.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    mask = y >= 0
    x, y = x[mask], y[mask]
    pos = int(y.sum())
    if pos < min_per_class or (len(y) - pos) < min_per_class:
        return None
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(scaler.transform(x_tr), y_tr)
    proba = clf.predict_proba(scaler.transform(x_te))[:, 1]
    return round(float(roc_auc_score(y_te, proba)), 3)


def make_controls(texts: list[str], seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """Two control labels every probe run needs:

    - ``random``: coin-flip labels — a valid probe must sit at AUC ~= 0.50 (no leakage).
    - ``length``: is the text longer than the median? — must be *decodable* (AUC well
      above 0.50), proving the probe + representation actually work.
    """
    rng = np.random.default_rng(seed)
    length = np.array([len(t) for t in texts])
    return {
        "random": rng.integers(0, 2, size=len(texts)),
        "length": (length > np.median(length)).astype(int),
    }


# --------------------------------------------------------------------------- #
# 5. Network-free fixtures — used by verify_notebooks.py so the reference runs
#    in seconds in CI without streaming HuggingFace datasets or loading mGPT.
# --------------------------------------------------------------------------- #
_SYNTH_MSA = [
    "الكتاب",
    "المدرسة",
    "الجامعة",
    "العلم",
    "المعرفة",
    "البحث",
    "الدراسة",
    "القراءة",
    "الفكرة",
    "النظرية",
]
_SYNTH_MASRI = ["عايز", "ازيك", "يلا", "خالص", "كده", "اهو", "معلش", "يعني", "بجد", "قوي"]


def synthetic_dialect_corpus(n_per_reg: int = 200, seed: int = DEFAULT_SEED) -> dict:
    """A tiny two-register corpus with **disjoint** vocab (no network).

    Same shape as load_dialect_corpus. The vocab gap makes dialect trivially
    recoverable — which is the point: it reproduces the real MSA/Masri lesson
    (high AUC from tokenization alone) fast enough for CI.
    """
    rng = np.random.default_rng(seed)
    sentences, register = [], []
    for reg, vocab in ((0, _SYNTH_MSA), (1, _SYNTH_MASRI)):
        for _ in range(n_per_reg):
            k = int(rng.integers(4, 10))
            sentences.append(" ".join(rng.choice(vocab, size=k)))
            register.append(reg)
    return {"sentences": sentences, "register": np.array(register, dtype=int)}


def whitespace_encoder(texts: list[str]):
    """Build a whitespace tokenizer over `texts` -> (encode, vocab_size). id 0 = [UNK].

    A dependency-free stand-in for tiny.make_compact_encoder (mGPT) on the
    synthetic corpus, so CI needs neither a HuggingFace download nor `transformers`.
    """
    vocab: dict[str, int] = {}
    for t in texts:
        for w in t.split():
            if w not in vocab:
                vocab[w] = len(vocab) + 1  # 0 reserved for [UNK]

    def encode(text: str) -> list[int]:
        return [vocab.get(w, 0) for w in text.split()]

    return encode, len(vocab) + 1


def definite_labels(words: list[str]) -> np.ndarray:
    """Cheap, gold-free definiteness proxy: does the word start with the article ``ال``?

    A teaching stand-in for camel-tools' morphological gold (Phase A used the gold).
    Good enough to demonstrate a *word-level* probe alongside the sentence-level
    dialect probe; not a research-grade label.
    """
    return np.array([1 if w.startswith("ال") else 0 for w in words], dtype=int)
