"""tiny.py — shared helper for the Stage 2 architecture-ladder notebooks.

Self-contained: depends only on torch and transformer_lens. Delivered to
Colab via wget of this single file; imported locally as a sibling module.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

DEFAULT_SEED = 42


def device() -> str:
    """Pick the GPU only if it can actually run a kernel, else CPU.

    Colab's CUDA passes; some ROCm builds (e.g. Strix Halo gfx1151 on the
    official wheels) report a GPU as "available" but have no runnable kernel for
    it, so a plain is_available() check would send work to a device that then
    crashes. We probe with a tiny op and fall back to CPU on failure.
    """
    if torch.cuda.is_available():
        try:
            torch.ones(1, device="cuda").add_(1)
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


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


def make_compact_encoder(tokenizer, texts):
    """Use a pretrained Arabic tokenizer but keep the vocab small enough to train.

    A from-scratch tiny model can't carry a 100k-row embedding, so we tokenize
    `texts`, keep only the subword ids that actually appear, and remap them to a
    compact 0..K space (id 0 reserved for [UNK]). Returns:
      encode(text) -> list[int]   # maps any text into the compact id space
      corpus_ids   -> list[list[int]]  # one compact id list per input text
      id_to_str    -> dict[int, str]   # compact id -> readable token (for plots)
      d_vocab      -> int              # size of the compact vocab (== K + 1)
    """
    raw = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    used = sorted({i for seq in raw for i in seq})
    remap = {old: j + 1 for j, old in enumerate(used)}  # 0 reserved for [UNK]
    id_to_str = {0: "[UNK]"}
    for old, j in remap.items():
        id_to_str[j] = tokenizer.decode([old]).strip() or "▁"
    corpus_ids = [[remap[i] for i in seq] for seq in raw]
    d_vocab = len(used) + 1

    def encode(text):
        return [remap.get(i, 0) for i in tokenizer.encode(text, add_special_tokens=False)]

    return encode, corpus_ids, id_to_str, d_vocab


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
    NB: the repeat is at a FIXED offset, so a single layer can copy it positionally.
    For a model that must build a real prev-token -> induction *composition*, use
    `make_repeated_with_gap` (variable offset defeats the positional shortcut).
    """
    assert seq_len % 2 == 0, "seq_len must be even"
    g = torch.Generator().manual_seed(seed)
    half = seq_len // 2
    first = torch.randint(1, d_vocab, (batch, half), generator=g)
    return torch.cat([first, first], dim=1)


def make_repeated_with_gap(batch, block_len, gap_max, d_vocab, seed=DEFAULT_SEED):
    """[BOS, A, <random gap>, A] — a random block A that repeats after a VARIABLE
    gap, so the repeat sits at a different offset each sequence. A fixed positional
    rule can't copy it; the model must find "where did this token appear before"
    (content) and look one past it -> a genuine prev-token(L0) -> induction(L1)
    composition forms. n_ctx = 1 + 2*block_len + gap_max.

    Returns (tokens, src), both [batch, n_ctx] long. Token id 1 is BOS; content ids
    are in [2, d_vocab). src[b, t] is the first-occurrence index of the token
    repeated at position t (else -1) — the known copy source for scoring induction.
    """
    g = torch.Generator().manual_seed(seed)
    n_ctx = 1 + 2 * block_len + gap_max
    tokens = torch.randint(2, d_vocab, (batch, n_ctx), generator=g)
    tokens[:, 0] = 1  # BOS
    src = torch.full((batch, n_ctx), -1, dtype=torch.long)
    for b in range(batch):
        gap = int(torch.randint(1, gap_max + 1, (1,), generator=g))
        s2 = 1 + block_len + gap
        tokens[b, s2 : s2 + block_len] = tokens[b, 1 : 1 + block_len].clone()
        for j in range(block_len):
            src[b, s2 + j] = 1 + j
    return tokens, src


def train(model, batches, n_epochs=10, lr=1e-3, seed=DEFAULT_SEED, log_every=0):
    """Full-batch train on a [N, n_ctx] long tensor. Returns per-epoch loss list.

    If log_every > 0, print progress every `log_every` epochs (and on the last),
    so long runs don't sit silent.
    """
    torch.manual_seed(seed)
    batches = batches.to(model.cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for e in range(n_epochs):
        opt.zero_grad()
        loss = model(batches, return_type="loss")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and ((e + 1) % log_every == 0 or (e + 1) == n_epochs):
            print(f"  epoch {e + 1:>4}/{n_epochs}  loss={losses[-1]:.3f}")
    return losses
