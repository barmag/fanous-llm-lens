"""tiny.py — shared helper for the Stage 2 architecture-ladder notebooks.

Self-contained: depends only on torch, transformer_lens, numpy. Delivered to
Colab via wget of this single file; imported locally as a sibling module.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

DEFAULT_SEED = 42


def device() -> str:
    """Generic device pick so identical code runs on Colab (CUDA) and Strix Halo."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_tiny_model(
    n_layers, n_heads, d_vocab, n_ctx, d_model=128, attn_only=True, seed=DEFAULT_SEED
):
    """Build a tiny HookedTransformer. Rungs 2a-2c keep attn_only=True; 2d flips it."""
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        d_mlp=(4 * d_model if not attn_only else None),
        attn_only=attn_only,
        act_fn=(None if attn_only else "gelu"),
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        normalization_type="LN",
        seed=seed,
        device=device(),
    )
    return HookedTransformer(cfg)


def make_natural_batches(token_ids, n_ctx, batch_size=None):
    """Chunk a 1D stream of ids into a [N, n_ctx] long tensor (drops the remainder)."""
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    n = ids.shape[0] // n_ctx
    ids = ids[: n * n_ctx].reshape(n, n_ctx)
    if batch_size is not None:
        ids = ids[:batch_size]
    return ids


def make_induction_data(batch, seq_len, d_vocab, seed=DEFAULT_SEED):
    """Sequences whose second half repeats their first half -> rewards induction.

    Returns a [batch, seq_len] long tensor; seq_len must be even; ids in [1, d_vocab).
    """
    assert seq_len % 2 == 0, "seq_len must be even"
    g = torch.Generator().manual_seed(seed)
    half = seq_len // 2
    first = torch.randint(1, d_vocab, (batch, half), generator=g)
    return torch.cat([first, first], dim=1)


def train(model, batches, n_epochs=10, lr=1e-3, seed=DEFAULT_SEED):
    """Full-batch train on a [N, n_ctx] long tensor. Returns per-epoch loss list."""
    torch.manual_seed(seed)
    batches = batches.to(model.cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = model(batches, return_type="loss")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses
