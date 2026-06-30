"""Phase A (depth) — shallow transformers (attention + MLP) per tokenization.

Phase A's zero-layer probe was **insensitive to training** (trained ≈ untrained, Δ ≤ ±0.014):
with no attention and no MLP a word is just the bag of its token embeddings, so the probe scored
the *tokenization*, not the model. This module adds depth so the **trained − untrained increment**
can become nonzero — the first measurement of what *learning* adds on top of tokenization.

**This is an isolated-word probe (deliberate).** Words are encoded alone, never in a sentence, to
avoid number/definiteness leaking through grammatical agreement. A consequence worth stating up
front: for *number*, an isolated word gives attention no within-word source of plurality to gather
(`ال` carries definiteness, not number; a fused plural is a single token), so the only lever on
number here is **MLP nonlinearity on the (possibly fused) stem token**. Read this experiment as an
"intra-word MLP reconstruction" test, not a full-context one — a null on number×morphological then
means "an MLP can't lift plurality out of one fused token," which is narrower than "alignment is
necessary."

Run on the iGPU with the ROCm backend (see run_probes_depth.py for the full driver):

    HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev --extra tokenizers \\
        python experiments/embedding-probes/run_probes_depth.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from fanous_lens.tokenizers.train import get_tokenizer, train_tokenizer


class Block(nn.Module):
    """Pre-norm transformer block: causal multi-head attention + GELU MLP, residual."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class DepthTransformer(nn.Module):
    """Embeddings + ``n_layers`` pre-norm blocks + final LN + tied head.

    ``n_layers=0`` collapses to the Phase A zero-layer model (embeddings only). The probe reads
    the **residual stream after each block** via :meth:`hidden_states` (index 0 = embeddings+pos).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    @staticmethod
    def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)

    def _stream(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Residual stream after each block; list length ``n_layers+1`` (0 = embeddings+pos)."""
        t = input_ids.shape[1]
        positions = torch.arange(t, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions)[None, :, :]
        mask = self._causal_mask(t, input_ids.device)
        states = [x]
        for blk in self.blocks:
            x = blk(x, mask)
            states.append(x)
        return states

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_f(self._stream(input_ids)[-1]))

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Residual stream after each block (no final LN). For probing at depth 0..n_layers."""
        return self._stream(input_ids)


@dataclass
class DepthConfig:
    approach: str
    n_layers: int = 2
    n_heads: int = 4
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
    """Flatten the corpus to a single id stream (encoder returns ``(ids, offsets)``)."""
    out: list[int] = []
    for sent in sentences:
        ids, _offsets = encode(sent)
        out.extend(ids)
        if len(out) >= max_tokens:
            break
    return out[:max_tokens]


def build_token_stream(approach: str, corpus: list[str], vocab_size: int, max_tokens: int):
    """Build the tokenizer **once** and flatten the corpus. Reused across seeds (deterministic)."""
    tok_config = train_tokenizer(approach, corpus, vocab_size=vocab_size)
    encode = get_tokenizer(approach, tok_config)
    ids = _tokenize_corpus(encode, corpus, max_tokens)
    return encode, tok_config, ids


def make_data(ids: list[int], seq_len: int) -> torch.Tensor:
    n_blocks = len(ids) // seq_len
    if n_blocks < 2:
        raise ValueError(f"corpus too small: {len(ids)} tokens < {2 * seq_len}")
    return torch.tensor(ids[: n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)


def train_inplace(model: DepthTransformer, data: torch.Tensor, cfg: DepthConfig) -> list[float]:
    """Train ``model`` in place on a prebuilt id-block tensor; return the loss trace.

    The model is **constructed by the caller** (so its random init can be captured for the
    untrained baseline before this call). Batch sampling uses ``cfg.seed``.
    """
    model.to(cfg.device).train()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    n_blocks = data.shape[0]
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
    model.eval()
    return losses


if __name__ == "__main__":
    # Smoke test: one tiny model trains and loss falls.
    from fanous_lens.tokenizers.corpora import load_corpora

    cfg = DepthConfig(approach="bpe", max_steps=200, max_msa=2_000, max_masri=1_000)
    msa, masri = load_corpora(max_msa=cfg.max_msa, max_masri=cfg.max_masri)
    _enc, _tc, ids = build_token_stream("bpe", msa + masri, cfg.vocab_size, cfg.max_tokens)
    data = make_data(ids, cfg.seq_len)
    torch.manual_seed(cfg.seed)
    model = DepthTransformer(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.seq_len)
    print(f"device={cfg.device} layers={cfg.n_layers}", flush=True)
    print("losses:", train_inplace(model, data, cfg))
