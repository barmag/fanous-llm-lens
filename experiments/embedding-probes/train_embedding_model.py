"""Phase A — train a zero-layer transformer (embeddings only) per tokenization.

A zero-layer model has **no attention and no MLP**: a token's representation is just
``embed(token) + pos(position)``, layer-normed, then an (un)tied linear head predicting the
next token. With no context mixing, *all* the structure the model can learn lives in the
embedding table ``W_E`` — so probing ``W_E`` measures what the **tokenization alone** makes
linearly available. That is exactly the Phase A question: does a morpheme-aligned tokenization
expose linguistic features more cleanly than frequency-only subword tokenization?

Run on the iGPU with the ROCm backend:

    HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev \\
        python experiments/embedding-probes/train_embedding_model.py morphological

(plain ``uv run`` reverts torch to a non-ROCm build — always pass ``--extra rocm``.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fanous_lens.tokenizers.corpora import load_corpora
from fanous_lens.tokenizers.train import get_tokenizer, train_tokenizer

CKPT_DIR = Path(__file__).parent / "checkpoints"


class ZeroLayerTransformer(nn.Module):
    """Embeddings + positional embeddings + LayerNorm + tied LM head. No attention/MLP."""

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


@dataclass
class TrainConfig:
    approach: str
    vocab_size: int = 8_000
    d_model: int = 256
    seq_len: int = 128
    batch_size: int = 64
    max_steps: int = 3_000
    lr: float = 1e-3
    max_msa: int = 20_000
    max_masri: int = 10_000
    max_tokens: int = 2_000_000
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 0


def _tokenize_corpus(encode, sentences: list[str], max_tokens: int) -> list[int]:
    """Flatten the corpus to a single id stream (encoder returns (ids, offsets))."""
    out: list[int] = []
    for sent in sentences:
        ids, _offsets = encode(sent)  # NB: encoder returns a tuple; take ids only
        out.extend(ids)
        if len(out) >= max_tokens:
            break
    return out[:max_tokens]


def train(cfg: TrainConfig, corpus: list[str] | None = None, save: bool = True) -> dict[str, Any]:
    """Train one zero-layer model; return ``{W_E, config, losses, ...}`` and optionally save."""
    torch.manual_seed(cfg.seed)
    if corpus is None:
        msa, masri = load_corpora(max_msa=cfg.max_msa, max_masri=cfg.max_masri)
        corpus = msa + masri

    tok_config = train_tokenizer(cfg.approach, corpus, vocab_size=cfg.vocab_size)
    encode = get_tokenizer(cfg.approach, tok_config)
    ids = _tokenize_corpus(encode, corpus, cfg.max_tokens)

    n_blocks = len(ids) // cfg.seq_len
    if n_blocks < 2:
        raise ValueError(f"corpus too small: {len(ids)} tokens < {2 * cfg.seq_len}")
    data = torch.tensor(ids[: n_blocks * cfg.seq_len], dtype=torch.long).view(n_blocks, cfg.seq_len)

    model = ZeroLayerTransformer(cfg.vocab_size, cfg.d_model, max_len=cfg.seq_len).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)

    model.train()
    losses: list[float] = []
    for step in range(cfg.max_steps):
        idx = torch.randint(0, n_blocks, (cfg.batch_size,), generator=gen)
        batch = data[idx].to(cfg.device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % max(1, cfg.max_steps // 20) == 0 or step == cfg.max_steps - 1:
            losses.append(round(loss.item(), 4))

    w_e = model.embed.weight.detach().cpu()
    result = {
        "approach": cfg.approach,
        "W_E": w_e,
        "tok_config": tok_config,
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "losses": losses,
        "final_loss": losses[-1],
    }
    if save:
        CKPT_DIR.mkdir(exist_ok=True)
        path = CKPT_DIR / f"{cfg.approach}_zerolayer.pt"
        torch.save(result, path)
        (CKPT_DIR / f"{cfg.approach}_meta.json").write_text(
            json.dumps({"approach": cfg.approach, "losses": losses, "device": cfg.device}, indent=2)
        )
    return result


if __name__ == "__main__":
    import sys

    approach = sys.argv[1] if len(sys.argv) > 1 else "morphological"
    cfg = TrainConfig(approach=approach)
    print(f"training {approach} on {cfg.device} ...", flush=True)
    res = train(cfg)
    print(f"losses: {res['losses']}")
    print(f"final loss: {res['final_loss']}")
