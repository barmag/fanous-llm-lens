"""Train the Stage 2dash model: a faithful-scale, one-layer attention-only Arabic
transformer, plus a small custom BPE tokenizer.

This is the heavy, run-once step (~1 h on the iGPU at ~338M tokens). It is deliberately a
*script*, not a notebook cell — the reference notebook loads the checkpoint it saves and
does the (fast) interpretability.

Why this config (see plan / A Mathematical Framework for Transformer Circuits):
  - 1 layer, attention-only, d_model=512, n_heads=8, d_head=64 — paper-class scale
    where skip-trigram circuits become legible rather than noise.
  - normalization_type=None + positional_embedding_type="shortformer" so the model
    decomposes *exactly* into a bigram direct path + per-head skip-trigram circuits.
  - ~12k unicode BPE (readable Arabic subwords) — vocab is the main data lever;
    Chinchilla D≈20·N puts the optimal data budget near ~270M tokens for this size.

Corpus is MSA-heavy by necessity (Masri data is scarce): Arabic Wikipedia + all the
Egyptian-dialect tweets. Outputs (under --out, gitignored):
  tokenizer.json, model.pt (state_dict + config), metrics.json, and cached
  corpus.txt / tokens.npy so re-runs skip re-streaming/re-tokenising.

The shipped model used --tokens 500_000_000, but Arabic Wikipedia is exhausted first, so
the actual corpus is ~338M tokens (1 epoch). The notebook's "~340M" claim refers to that.

Run (GPU needs the gfx1151 masquerade):
  HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm python \
      notebooks/education/train_stage2dash.py --tokens 500_000_000
  # push the checkpoint to the Hub so the notebook's Colab path works (needs HF login):
  ... --push-hub --hf-repo <user>/fanous-stage2dash-attn-only-1l
  # quick end-to-end smoke + throughput projection:
  ... --calibrate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

import numpy as np
import tiny
import torch

# --------------------------------------------------------------------------- #
# Corpus: stream + clean (same sources/cleaning as Stage 2a, just much more).
# --------------------------------------------------------------------------- #
_AR = re.compile(r"[^\sء-ي]")
_NOISE = re.compile(r"[a-zA-Z0-9_@]+")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _NOISE.sub("", text)
    return _AR.sub("", text)


def build_corpus(char_budget: int, cache_path: str) -> str:
    """Stream Arabic Wikipedia (MSA) + all EG tweets (Masri), clean, concat to
    ~char_budget characters. Cached to disk so re-runs are instant."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            text = f.read()
        print(f"[corpus] cache hit: {len(text):,} chars from {cache_path}")
        return text

    from datasets import load_dataset

    print(f"[corpus] streaming up to {char_budget:,} chars (MSA Wikipedia + EG tweets)...")
    parts: list[str] = []
    total = 0

    # Masri first (small, ~few M chars) — keep all of it so the dialect signal is in.
    tweets = load_dataset("amgadhasan/arabic_tweets_dialects", split="train")
    eg = tweets.filter(lambda x: x["dialect"] == "EG")
    for r in eg:
        c = clean(r["text"])
        if c:
            parts.append(c)
            total += len(c) + 1
    masri_chars = total
    print(f"[corpus] Masri (EG tweets): {masri_chars:,} chars")

    # MSA fills the rest of the budget.
    wiki = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
    for r in wiki:
        if total >= char_budget:
            break
        c = clean(r["text"])
        if c:
            parts.append(c)
            total += len(c) + 1
    text = " ".join(parts)
    print(
        f"[corpus] total: {len(text):,} chars (MSA {len(text) - masri_chars:,} + Masri {masri_chars:,})"
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# --------------------------------------------------------------------------- #
# Tokenizer: small unicode-level BPE -> readable Arabic subword tokens.
# --------------------------------------------------------------------------- #
def train_tokenizer(text: str, vocab_size: int, out_path: str):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if os.path.exists(out_path):
        print(f"[tok] cache hit: {out_path}")
        return Tokenizer.from_file(out_path)

    print(f"[tok] training {vocab_size}-vocab BPE...")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["[UNK]"])
    # chunk the text so the trainer iterates instead of holding one giant string
    chunk = 1_000_000
    tok.train_from_iterator(
        (text[i : i + chunk] for i in range(0, len(text), chunk)), trainer=trainer
    )
    tok.save(out_path)
    print(f"[tok] saved {tok.get_vocab_size()} tokens -> {out_path}")
    return tok


def tokenize(text: str, tok, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        ids = np.load(cache_path)
        print(f"[tok] token cache hit: {len(ids):,} ids from {cache_path}")
        return ids
    print("[tok] encoding corpus (chunked)...")
    # chunk so a multi-GB corpus doesn't build one giant Encoding in memory
    chunks, step = [], 5_000_000
    for i in range(0, len(text), step):
        chunks.append(np.asarray(tok.encode(text[i : i + step]).ids, dtype=np.uint16))
    ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    np.save(cache_path, ids)
    print(f"[tok] {len(ids):,} tokens -> {cache_path}")
    return ids


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    device = tiny.device()
    print(f"[train] device={device}")

    char_budget = args.corpus_chars or int(args.tokens * 4.2)  # ~chars/token headroom
    text = build_corpus(char_budget, os.path.join(out, "corpus.txt"))
    tok = train_tokenizer(text, args.vocab, os.path.join(out, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    ids = tokenize(text, tok, os.path.join(out, "tokens.npy"))
    if len(ids) > args.tokens:
        ids = ids[: args.tokens]
    print(f"[train] training on {len(ids):,} tokens, vocab={vocab}")

    # [N, n_ctx] batches
    n = len(ids) // args.n_ctx
    data = torch.from_numpy(ids[: n * args.n_ctx].astype(np.int64)).reshape(n, args.n_ctx)
    print(f"[train] sequences: {tuple(data.shape)}")

    model = tiny.make_tiny_model(
        n_layers=1,
        n_heads=args.n_heads,
        d_vocab=vocab,
        n_ctx=args.n_ctx,
        d_model=args.d_model,
        attn_only=True,
        normalization_type=None,
        positional_embedding_type="shortformer",
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.99))
    steps = args.steps or (len(ids) // (args.batch * args.n_ctx))
    warmup = max(1, int(0.02 * steps))

    def lr_at(step):  # linear warmup -> cosine decay
        if step < warmup:
            return step / warmup
        import math

        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    g = torch.Generator().manual_seed(tiny.DEFAULT_SEED)
    model.train()
    t0 = time.time()
    tokens_seen = 0
    losses = []
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (args.batch,), generator=g)
        batch = data[idx].to(device)
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_at(step)
        opt.zero_grad()
        loss = model(batch, return_type="loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += args.batch * args.n_ctx
        if step % 50 == 0 or step == steps - 1:
            losses.append((step, float(loss)))
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-9)
            eta = (steps - step - 1) * (dt / max(step + 1, 1)) / 60
            print(
                f"  step {step:>6}/{steps}  loss={float(loss):.3f}  "
                f"{tps:,.0f} tok/s  eta {eta:.1f} min",
                flush=True,
            )
        if args.calibrate and step >= args.calibrate_steps:
            dt = time.time() - t0
            tps = tokens_seen / dt
            print(
                f"\n[calibrate] {tps:,.0f} tok/s -> "
                f"{args.tokens / tps / 60:.1f} min for {args.tokens:,} tokens "
                f"({args.tokens / tps / 3600:.2f} h)"
            )
            return

    # save
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n_layers": 1,
                "n_heads": args.n_heads,
                "d_model": args.d_model,
                "d_head": args.d_model // args.n_heads,
                "d_vocab": vocab,
                "n_ctx": args.n_ctx,
                "attn_only": True,
                "normalization_type": None,
                "positional_embedding_type": "shortformer",
            },
        },
        os.path.join(out, "model.pt"),
    )
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    metrics = {
        "tokens": len(ids),
        "vocab": vocab,
        "n_params": n_params,
        "steps": steps,
        "loss_start": losses[0][1],
        "loss_end": losses[-1][1],
        "minutes": (time.time() - t0) / 60,
        "seed": tiny.DEFAULT_SEED,
        "commit": sha,
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] done: {json.dumps(metrics, indent=2)}")

    if args.push_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hf_repo, repo_type="model", exist_ok=True)
        for fn in ("tokenizer.json", "model.pt", "metrics.json"):
            api.upload_file(path_or_fileobj=os.path.join(out, fn),
                            path_in_repo=fn, repo_id=args.hf_repo, repo_type="model")
        print(f"[train] pushed checkpoint to https://huggingface.co/{args.hf_repo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=500_000_000,
                   help="token budget; Arabic Wikipedia exhausts first (~338M, 1 epoch)")
    p.add_argument("--vocab", type=int, default=12_000)
    p.add_argument("--n-ctx", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--steps", type=int, default=0, help="override step count (0 = derive from tokens)"
    )
    p.add_argument("--corpus-chars", type=int, default=0, help="override char budget (0 = auto)")
    p.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", "stage2dash"
        ),
    )
    p.add_argument(
        "--calibrate", action="store_true", help="short run: project throughput then stop"
    )
    p.add_argument("--calibrate-steps", type=int, default=200)
    p.add_argument("--push-hub", action="store_true", help="upload checkpoint to the HF Hub (needs login)")
    p.add_argument("--hf-repo", default="yassermakram/fanous-stage2dash-attn-only-1l",
                   help="HF repo id for --push-hub (must match HF_REPO in the notebook)")
    args = p.parse_args()
    # make `import tiny` work from anywhere
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    train(args)


if __name__ == "__main__":
    main()
