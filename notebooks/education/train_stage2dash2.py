"""Train the Stage 2dash² model: a faithful-scale, TWO-layer attention-only Arabic
transformer, reusing the Stage 2dash tokenizer + token cache.

Run-once heavy step (headless — the iGPU drives the display). The reference notebook
stage2_dash2_composition_induction_reference.ipynb loads the checkpoint and does the
fast interpretability (composition algebra + induction head).

Why this config (A Mathematical Framework for Transformer Circuits, two-layer section):
  - 2 layers, attention-only, d_model=512, n_heads=8 — paper-class scale where
    head composition forms a legible induction head.
  - normalization_type=None + positional_embedding_type="shortformer" so the two-layer
    path expansion is exact and induction is purely content-based (a principled
    deviation from the paper's LN + learned-positional attn-only-2l).
  - Reuses the Stage 2dash 12k unicode BPE tokenizer.json + tokens.npy (identical vocab
    and corpus) so the notebook's 1-layer-vs-2-layer comparison is on identical tokens.

A verification gate asserts an induction head emerged (induction score >= threshold)
before the checkpoint is saved; per-head scores are written to metrics.json.

Run (headless, gfx1151 masquerade):
  HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm python \
      notebooks/education/train_stage2dash2.py --bf16
  ... --push-hub --hf-repo <user>/fanous-stage2dash2-attn-only-2l
  ... --calibrate            # throughput projection then stop
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from contextlib import nullcontext

import numpy as np
import tiny
import torch

from corpus import build_corpus, train_tokenizer, tokenize

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    device = tiny.device()
    print(f"[train] device={device}")

    # Reuse the Stage 2dash tokenizer + corpus + token cache (identical vocab/corpus).
    src = args.reuse_from
    char_budget = args.corpus_chars or int(args.tokens * 4.2)
    text = build_corpus(char_budget, os.path.join(src, "corpus.txt"))
    tok = train_tokenizer(text, args.vocab, os.path.join(src, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    ids = tokenize(text, tok, os.path.join(src, "tokens.npy"))
    if len(ids) > args.tokens:
        ids = ids[: args.tokens]
    print(f"[train] training on {len(ids):,} tokens, vocab={vocab} (reused from {src})")
    # copy the tokenizer into the 2dash² out dir so the checkpoint is self-contained
    import shutil
    shutil.copy(os.path.join(src, "tokenizer.json"), os.path.join(out, "tokenizer.json"))

    # [N, n_ctx] batches
    n = len(ids) // args.n_ctx
    data = torch.from_numpy(ids[: n * args.n_ctx].astype(np.int64)).reshape(n, args.n_ctx)
    print(f"[train] sequences: {tuple(data.shape)}")

    model = tiny.make_tiny_model(
        n_layers=2,
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

    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if (args.bf16 and device == "cuda") else nullcontext())

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
        with amp:
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

    # verification gate: assert an induction head formed before saving
    model.eval()
    ind = tiny.induction_scores(model)          # (n_layers, n_heads)
    passed, layer, head = tiny.induction_gate(ind, args.induction_threshold)
    best = float(ind.max())
    print(f"[gate] best induction score {best:.3f} at layer {layer} head {head}")
    if not passed:
        raise SystemExit(
            f"[gate] FAILED: best induction score {best:.3f} < {args.induction_threshold}. "
            f"No induction head formed — checkpoint NOT saved.")
    model.train()

    # save
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n_layers": 2,
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
        "induction_scores": ind.tolist(),
        "best_induction_score": best,
        "induction_head": [layer, head],
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
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints", "stage2dash2"))
    p.add_argument(
        "--calibrate", action="store_true", help="short run: project throughput then stop"
    )
    p.add_argument("--calibrate-steps", type=int, default=200)
    p.add_argument("--push-hub", action="store_true", help="upload checkpoint to the HF Hub (needs login)")
    p.add_argument("--hf-repo", default="yassermakram/fanous-stage2dash2-attn-only-2l",
                   help="HF repo id for --push-hub (must match HF_REPO in the notebook)")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast for the forward (params stay fp32); recommended on GPU")
    p.add_argument("--induction-threshold", type=float, default=0.4,
                   help="min best induction score required to save the checkpoint")
    p.add_argument("--reuse-from", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints", "stage2dash"),
        help="dir holding the 2dash corpus.txt / tokenizer.json / tokens.npy to reuse")
    args = p.parse_args()
    # make `import tiny` work from anywhere
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    train(args)


if __name__ == "__main__":
    main()
