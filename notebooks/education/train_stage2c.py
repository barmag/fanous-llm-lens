"""Train the Stage 2c model: a faithful-scale, TWO-layer attention-only TinyStories
transformer for the induction-head notebook.

Run-once heavy step (headless — the iGPU drives the display). The notebook
stage2c_induction_tinystories.ipynb loads the checkpoint and does the fast
interpretability (five induction panels + sandbox).

Why this config (diagnosed 2026-07-03; probes in the session, verdict in the notebook):
  - The old in-notebook recipe (2 heads, d_model=256, n_ctx=64, 1.7M tokens recycled
    ~10 epochs) never forms an induction head: canonical induction score flat at
    0.002 over 6k extra steps. Two heads can't spare one for a clean prev-token
    head (K-composition's prerequisite), and a tiny recycled corpus lets the model
    memorize its trigrams, starving copying-from-context of marginal loss.
  - Scale mirrors train_stage2dash2.py, the recipe that passed the 0.4 induction
    gate: 2 layers, attention-only, d_model=512, n_heads=8, n_ctx=512,
    lr 1e-3 with warmup + cosine decay — but on TinyStories with a fresh
    (few-epoch) corpus and a 2048-token BPE.
  - Architecture defaults to LN + standard learned positions (--norm/--pos flip
    it): dash2's no-LN + shortformer combo, principled for its exact path
    algebra, formed induction far slower on TinyStories (0.022 at 130M tokens
    vs 0.096 for a smaller standard-arch probe on half the data).

A verification gate asserts an induction head emerged (induction score >= threshold)
before the checkpoint is saved; per-head scores are written to metrics.json.
The induction score is also logged every 500 steps so a run that will fail the
gate is visible long before it finishes.

Run (headless, gfx1151 masquerade):
  HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --no-sync python \
      notebooks/education/train_stage2c.py --bf16
  ... --calibrate            # throughput projection then stop
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext

import numpy as np
import tiny
import torch


# --------------------------------------------------------------------------- #
# Corpus + tokenizer (TinyStories, same cleaning as the notebook's data cell)
# --------------------------------------------------------------------------- #
def build_corpus(char_budget: int, cache_path: str) -> str:
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= char_budget:
            print(f"[corpus] cache hit: {len(text):,} chars from {cache_path}")
            return text
        print(f"[corpus] cache too small ({len(text):,} < {char_budget:,}), rebuilding")

    from datasets import load_dataset

    print(f"[corpus] streaming up to {char_budget:,} chars of TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    parts: list[str] = []
    total = 0
    for item in ds:
        if total >= char_budget:
            break
        t = item["text"].strip()
        if t and len(t) > 10:
            parts.append(t)
            total += len(t) + 1
    text = "\n".join(parts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[corpus] saved {len(text):,} chars to {cache_path}")
    return text


def train_tokenizer(text: str, vocab_size: int, out_path: str):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if os.path.exists(out_path):
        tok = Tokenizer.from_file(out_path)
        if tok.get_vocab_size() == vocab_size:
            print(f"[tok] cache hit: {out_path}")
            return tok
        print(f"[tok] cached vocab {tok.get_vocab_size()} != {vocab_size}, retraining")

    print(f"[tok] training {vocab_size}-vocab BPE...")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=["[UNK]", "[BOS]", "[EOS]"]
    )
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder()
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
        print(f"[tok] cache hit: {len(ids):,} ids")
        return ids
    print("[tok] encoding corpus...")
    chunks = []
    step = 5_000_000
    for i in range(0, len(text), step):
        chunks.append(np.asarray(tok.encode(text[i : i + step]).ids, dtype=np.uint16))
    ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    np.save(cache_path, ids)
    print(f"[tok] saved {len(ids):,} tokens to {cache_path}")
    return ids


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    device = tiny.device()
    print(f"[train] device={device}")

    char_budget = args.corpus_chars or int(args.tokens / args.epochs * 3.6)
    text = build_corpus(char_budget, os.path.join(out, "corpus.txt"))
    tok = train_tokenizer(text, args.vocab, os.path.join(out, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    # cache keyed on vocab so a --vocab change can't silently reuse stale ids
    ids = tokenize(text, tok, os.path.join(out, f"tokens_v{vocab}.npy"))
    epochs = args.tokens / max(len(ids), 1)
    print(
        f"[train] {len(ids):,} corpus tokens, vocab={vocab}, "
        f"{args.tokens:,} training tokens = {epochs:.1f} epochs"
    )

    n = len(ids) // args.n_ctx
    data = torch.from_numpy(ids[: n * args.n_ctx].astype(np.int64)).reshape(n, args.n_ctx)
    print(f"[train] sequences: {tuple(data.shape)}")

    # Fixed held-out batch for a low-variance loss reading: the per-step training
    # loss below is a single args.batch-sequence minibatch and is dominated by
    # sampling noise (diagnosed 2026-07-11 — bumps of similar size to any
    # candidate "phase change" appear uniformly throughout a run). Excluded from
    # the training sampling pool so eval_loss is a genuine held-out signal.
    eval_n = min(256, max(1, n // 4))
    train_pool = n - eval_n
    eval_batch = data[train_pool:].to(device)

    norm = None if args.norm == "none" else args.norm
    model = tiny.make_tiny_model(
        n_layers=2,
        n_heads=args.n_heads,
        d_vocab=vocab,
        n_ctx=args.n_ctx,
        d_model=args.d_model,
        attn_only=True,
        normalization_type=norm,
        positional_embedding_type=args.pos,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")
    config = {
        "n_layers": 2,
        "n_heads": args.n_heads,
        "d_model": args.d_model,
        "d_head": args.d_model // args.n_heads,
        "d_vocab": vocab,
        "n_ctx": args.n_ctx,
        "attn_only": True,
        "normalization_type": norm,
        "positional_embedding_type": args.pos,
    }

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.99))
    steps = args.steps or (args.tokens // (args.batch * args.n_ctx))
    warmup = max(1, int(0.02 * steps))

    def lr_at(step):  # linear warmup -> cosine decay
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (args.bf16 and device == "cuda")
        else nullcontext()
    )

    g = torch.Generator().manual_seed(tiny.DEFAULT_SEED)
    model.train()
    t0 = time.time()
    tokens_seen = 0
    losses = []
    eval_losses = []
    induction_curve = []
    inprogress = os.path.join(out, "model_inprogress.pt")
    for step in range(steps):
        idx = torch.randint(0, train_pool, (args.batch,), generator=g)
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
            model.eval()
            with torch.no_grad(), amp:
                eval_loss = float(model(eval_batch, return_type="loss"))
            model.train()
            eval_losses.append((step, eval_loss))
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-9)
            eta = (steps - step - 1) * (dt / max(step + 1, 1)) / 60
            print(
                f"  step {step:>6}/{steps}  loss={float(loss):.3f}  "
                f"eval_loss={eval_loss:.3f}  "
                f"{tps:,.0f} tok/s  eta {eta:.1f} min",
                flush=True,
            )
        if step % args.induction_every == 0 or step == steps - 1:
            model.eval()
            ind = tiny.induction_scores(model)
            model.train()
            best = float(ind.max())
            induction_curve.append((step, best))
            print(
                f"  [induction] step {step:>6}  best={best:.3f}  "
                f"L1={[round(float(v), 3) for v in ind[1]]}",
                flush=True,
            )
        if step > 0 and step % 1000 == 0:
            # same payload as the final save so a gate-failed or crashed run
            # leaves weights the notebook's load cell can read directly
            torch.save({"state_dict": model.state_dict(), "config": config}, inprogress)
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
    ind = tiny.induction_scores(model)  # (n_layers, n_heads)
    passed, layer, head = tiny.induction_gate(ind, args.induction_threshold)
    best = float(ind.max())
    print(f"[gate] best induction score {best:.3f} at layer {layer} head {head}")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    metrics = {
        "corpus_tokens": len(ids),
        "training_tokens": tokens_seen,
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
        "gate_passed": passed,
        "loss_curve": losses,
        "eval_loss_curve": eval_losses,
        "induction_curve": induction_curve,
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if not passed:
        raise SystemExit(
            f"[gate] FAILED: best induction score {best:.3f} < {args.induction_threshold}. "
            f"No induction head formed — model.pt NOT saved (metrics.json written; "
            f"last periodic weights remain in {inprogress})."
        )

    torch.save({"state_dict": model.state_dict(), "config": config}, os.path.join(out, "model.pt"))
    if os.path.exists(inprogress):
        os.remove(inprogress)
    print(f"[train] done: {json.dumps(metrics, indent=2)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokens",
        type=int,
        default=130_000_000,
        help="training-token budget (with --epochs sets the corpus size)",
    )
    p.add_argument(
        "--epochs", type=float, default=2.0, help="max data reuse; corpus is sized to tokens/epochs"
    )
    p.add_argument("--vocab", type=int, default=2048)
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
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", "stage2c_tinystories"
        ),
    )
    p.add_argument(
        "--norm",
        choices=["LN", "none"],
        default="LN",
        help="normalization (LN formed induction faster than none in the "
        "2026-07-03 probes on TinyStories)",
    )
    p.add_argument(
        "--pos",
        choices=["standard", "shortformer"],
        default="standard",
        help="positional embedding type (standard beat shortformer here; "
        "shortformer matches train_stage2dash2.py's exact-algebra setup)",
    )
    p.add_argument(
        "--calibrate", action="store_true", help="short run: project throughput then stop"
    )
    p.add_argument("--calibrate-steps", type=int, default=200)
    p.add_argument(
        "--bf16",
        action="store_true",
        help="bf16 autocast for the forward (params stay fp32); recommended on GPU",
    )
    p.add_argument(
        "--induction-every",
        type=int,
        default=500,
        help="steps between induction-score evals during training (denser = finer "
        "localization of the phase transition, at the cost of extra eval passes)",
    )
    p.add_argument(
        "--induction-threshold",
        type=float,
        default=0.2,
        help="min best induction score required to save the checkpoint. 0.2 is "
        "calibrated for TinyStories: the 200M-token run phase-changes at ~1000 "
        "steps and plateaus at ~0.22, about 15x the ~1/73 attention a uniform "
        "causal head would put on the induction stripe in the eval's 100-token "
        "sequences; dash2's 0.4 was calibrated on its higher-entropy Arabic "
        "12k-vocab corpus",
    )
    args = p.parse_args()
    # make `import tiny` work from anywhere
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    train(args)


if __name__ == "__main__":
    main()
