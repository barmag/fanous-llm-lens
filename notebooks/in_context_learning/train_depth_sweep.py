"""Overnight depth/architecture sweep for icl_from_scratch.ipynb.

Standalone script, not a notebook cell: it needs to run for hours unattended, and
`nbconvert --execute --inplace` only writes the notebook file back once the *whole*
run finishes (see feedback_nbconvert_inplace_clobbers_concurrent_edits memory) — a
long background nbconvert run would clobber any interactive notebook editing done in
the meantime. This script trains, logs, and checkpoints entirely on disk instead; a
later notebook cell reads its results back in.

Reuses the exact tokenizer/tokens already cached by icl_from_scratch.ipynb under
checkpoints/icl_pile/ — no new corpus work, only new models.

Each config is a variant of the same architecture (d_model=256, n_heads=8, n_ctx=2048,
LN, shortformer positions) along the two axes the notebook's markdown flagged as
untested: depth (2, 3, 6 layers, attn-only, per Olsson et al.'s own depth comparison)
and MLPs (2-layer standard transformer, not attn-only, holding depth fixed at the
baseline to isolate that one variable).

Fixes the resolution gap diagnosed in the notebook: EVAL_EVERY=100 undersampled the
first ~1000 steps, where the whole ICL-score transition happens. `eval_schedule()`
below logs far more densely early and coarsens later, instead of a uniform stride.

Per-config wall time is capped (TARGET_MINUTES) so one slow config (e.g. 6 layers)
can't consume the whole overnight budget; STEPS is chosen from calibrated tok/s to fit
under the cap or TOKEN_BUDGET, whichever binds first. Each config is independently
checkpoint-resumable (same pattern as the notebook's 3a-3g cells) and wrapped in a
try/except so one config crashing doesn't take down the rest of the sweep.
"""

import json
import math
import os
import sys
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__) + "/../education")
import tiny  # noqa: E402

BASE_CACHE = os.path.join(os.path.dirname(__file__), "checkpoints/icl_pile")
SWEEP_CACHE = os.path.join(os.path.dirname(__file__), "checkpoints/icl_depth_sweep")
RESULTS_PATH = f"{SWEEP_CACHE}/results.json"
TOP_LOG = f"{SWEEP_CACHE}/sweep_log.txt"

VOCAB_SIZE = 2048
N_CTX = 2048
TOKEN_BUDGET = 200_000_000
TARGET_MINUTES = 150  # per-config wall-time safety cap
WARMUP_FRAC = 0.02
LR = 1e-3
SNAPSHOT_EVERY = 500

CONFIGS = [
    dict(label="L2_attn", n_layers=2, attn_only=True),
    dict(label="L3_attn", n_layers=3, attn_only=True),
    dict(label="L6_attn", n_layers=6, attn_only=True),
    dict(label="L2_mlp", n_layers=2, attn_only=False),
]

BATCH_CANDIDATES = [32, 16, 8]  # conservative vs. the main notebook's 64 -- unattended, no one to
# rescue an OOM crash overnight; fall back automatically instead


def top_log(msg):
    print(msg, flush=True)
    os.makedirs(SWEEP_CACHE, exist_ok=True)
    with open(TOP_LOG, "a") as f:
        f.write(msg + "\n")


def eval_schedule(total_steps):
    """Dense early, sparse late -- fixes the under-sampled-transition bug found in the main run."""
    pts = set()
    pts.update(range(0, min(100, total_steps), 5))
    pts.update(range(100, min(300, total_steps), 20))
    pts.update(range(300, min(1000, total_steps), 50))
    pts.update(range(1000, total_steps, 100))
    pts.add(total_steps - 1)
    return sorted(s for s in pts if 0 <= s < total_steps)


def load_data():
    tokens = np.load(f"{BASE_CACHE}/tokens_v{VOCAB_SIZE}.npy")
    eval_tokens_n = 100 * N_CTX
    train_tokens, eval_tokens = tokens[:-eval_tokens_n], tokens[-eval_tokens_n:]

    def chunk(ids, seq_len):
        usable = (len(ids) // seq_len) * seq_len
        return ids[:usable].reshape(-1, seq_len)

    return chunk(train_tokens, N_CTX), chunk(eval_tokens, N_CTX)


def calibrate(model_kwargs, device, train_seqs):
    """Try BATCH_CANDIDATES in order, falling back on OOM. Returns (batch, tok/s)."""
    for batch in BATCH_CANDIDATES:
        try:
            model = tiny.make_tiny_model(**model_kwargs)
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            g = torch.Generator().manual_seed(tiny.DEFAULT_SEED)
            train_data = torch.from_numpy(train_seqs.astype(np.int64))
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model.train()
            t0 = time.time()
            tokens_seen = 0
            for _ in range(20):
                idx = torch.randint(0, len(train_seqs), (batch,), generator=g)
                seq = train_data[idx].to(device)
                opt.zero_grad()
                loss = model(seq, return_type="loss")
                loss.backward()
                opt.step()
                tokens_seen += batch * N_CTX
            dt = time.time() - t0
            tps = tokens_seen / dt
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
            del model, opt
            if device == "cuda":
                torch.cuda.empty_cache()
            top_log(f"  [calibrate] batch={batch} -> {tps:,.0f} tok/s, peak {peak:.1f} GB")
            return batch, tps
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            top_log(f"  [calibrate] batch={batch} OOM, falling back")
            if device == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("all batch candidates OOM'd")


def run_config(cfg, train_seqs, eval_seqs, device):
    label = cfg["label"]
    cache_dir = f"{SWEEP_CACHE}/{label}"
    os.makedirs(f"{cache_dir}/snapshots", exist_ok=True)
    model_path, metrics_path = f"{cache_dir}/model.pt", f"{cache_dir}/metrics.json"
    inprogress_path = f"{cache_dir}/model_inprogress.pt"

    model_kwargs = dict(
        n_layers=cfg["n_layers"],
        n_heads=8,
        d_model=256,
        n_ctx=N_CTX,
        d_vocab=VOCAB_SIZE,
        attn_only=cfg["attn_only"],
        normalization_type="LN",
        positional_embedding_type="shortformer",
    )

    top_log(f"[{label}] config: {model_kwargs}")
    model = tiny.make_tiny_model(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    top_log(f"[{label}] params: {n_params:,}")

    batch, tps = calibrate(model_kwargs, device, train_seqs)
    steps_for_budget = TOKEN_BUDGET // (batch * N_CTX)
    steps_for_cap = int(TARGET_MINUTES * 60 * tps) // (batch * N_CTX)
    steps = min(steps_for_budget, steps_for_cap)
    if steps_for_cap < steps_for_budget:
        top_log(
            f"[{label}] {tps:,.0f} tok/s too slow for {TOKEN_BUDGET:,} tokens within "
            f"{TARGET_MINUTES} min -- capping at {steps:,} steps instead of {steps_for_budget:,}"
        )
    eval_points = set(eval_schedule(steps))
    eta_min = steps * batch * N_CTX / tps / 60
    top_log(f"[{label}] STEPS={steps:,} batch={batch} ~{eta_min:.1f} min, {len(eval_points)} eval points")

    # resume support, same pattern as the notebook's 3a/3b cells
    ckpt, ckpt_source = None, None
    for path in (model_path, inprogress_path):
        if os.path.exists(path):
            ckpt, ckpt_source = torch.load(path, map_location=device, weights_only=False), path
            break

    if ckpt is not None:
        assert ckpt["config"] == model_kwargs, f"checkpoint config mismatch at {ckpt_source}"
        model.load_state_dict(ckpt["state_dict"])
        with open(metrics_path) as f:
            run_metrics = json.load(f)
        start_step = ckpt.get("step", run_metrics["steps"][-1]) + 1
    else:
        run_metrics = None
        start_step = 0

    if start_step >= steps:
        top_log(f"[{label}] checkpoint already covers {steps} steps, skipping")
        return run_metrics

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05, betas=(0.9, 0.99))
    g = torch.Generator().manual_seed(tiny.DEFAULT_SEED)
    if ckpt is not None and "opt_state_dict" in ckpt:
        opt.load_state_dict(ckpt["opt_state_dict"])
        g.set_state(ckpt["gen_state"])
        top_log(f"[{label}] resuming from {ckpt_source} at step {start_step}/{steps}")
    else:
        top_log(f"[{label}] starting fresh, 0/{steps} steps")

    warmup = max(1, int(WARMUP_FRAC * steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    train_data = torch.from_numpy(train_seqs.astype(np.int64))
    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    if run_metrics is not None:
        steps_log = run_metrics["steps"]
        loss_log = run_metrics["loss"]
        eval_loss_log = run_metrics["eval_loss"]
        per_pos_log = run_metrics["per_position_loss"]
        induction_log = run_metrics["induction_score"]
        tokens_seen = run_metrics["training_tokens"]
        prior_minutes = run_metrics["minutes"]
    else:
        steps_log, loss_log, eval_loss_log, per_pos_log, induction_log = [], [], [], [], []
        tokens_seen = 0
        prior_minutes = 0.0

    model.train()
    t0 = time.time()
    for step in range(start_step, steps):
        idx = torch.randint(0, len(train_seqs), (batch,), generator=g)
        seq = train_data[idx].to(device)
        for pg in opt.param_groups:
            pg["lr"] = LR * lr_at(step)
        opt.zero_grad()
        with amp:
            loss = model(seq, return_type="loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += batch * N_CTX

        if step in eval_points:
            model.eval()
            with torch.no_grad(), amp:
                eval_loss_per_token = model(eval_data, return_type="loss", loss_per_token=True)
            eval_loss = float(eval_loss_per_token.mean())
            per_position = eval_loss_per_token.float().mean(dim=0).cpu().tolist()
            induction = tiny.induction_scores(model)
            model.train()

            steps_log.append(step)
            loss_log.append(float(loss))
            eval_loss_log.append(eval_loss)
            per_pos_log.append(per_position)
            induction_log.append(float(induction.max()))

            dt = time.time() - t0
            eta = (steps - step - 1) * (dt / max(step - start_step + 1, 1)) / 60
            top_log(
                f"[{label}] step {step:>5}/{steps}  loss={float(loss):.3f}  eval_loss={eval_loss:.3f}  "
                f"induction={induction_log[-1]:.3f}  eta {eta:.1f} min"
            )

        if step > 0 and step % SNAPSHOT_EVERY == 0:
            state = {
                "state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "gen_state": g.get_state(),
                "config": model_kwargs,
                "step": step,
            }
            torch.save(state, f"{cache_dir}/snapshots/step_{step:06d}.pt")
            torch.save(state, inprogress_path)
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "steps": steps_log,
                        "loss": loss_log,
                        "eval_loss": eval_loss_log,
                        "per_position_loss": per_pos_log,
                        "induction_score": induction_log,
                        "training_tokens": tokens_seen,
                        "vocab": VOCAB_SIZE,
                        "n_params": n_params,
                        "batch_size": batch,
                        "minutes": prior_minutes + (time.time() - t0) / 60,
                        "seed": tiny.DEFAULT_SEED,
                        "config_label": label,
                    },
                    f,
                )

    model.eval()
    induction_final = tiny.induction_scores(model)
    run_metrics = {
        "steps": steps_log,
        "loss": loss_log,
        "eval_loss": eval_loss_log,
        "per_position_loss": per_pos_log,
        "induction_score": induction_log,
        "training_tokens": tokens_seen,
        "vocab": VOCAB_SIZE,
        "n_params": n_params,
        "batch_size": batch,
        "minutes": prior_minutes + (time.time() - t0) / 60,
        "seed": tiny.DEFAULT_SEED,
        "config_label": label,
        "best_induction_score": float(induction_final.max()),
    }
    with open(metrics_path, "w") as f:
        json.dump(run_metrics, f)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "gen_state": g.get_state(),
            "config": model_kwargs,
            "step": steps - 1,
        },
        model_path,
    )
    if os.path.exists(inprogress_path):
        os.remove(inprogress_path)
    top_log(
        f"[{label}] done: {steps} steps, {tokens_seen:,} tokens, {run_metrics['minutes']:.1f} min, "
        f"eval_loss {eval_loss_log[-1]:.3f}, best induction {run_metrics['best_induction_score']:.3f}"
    )
    return run_metrics


def main():
    device = tiny.device()
    top_log(f"=== depth sweep starting, device={device} ===")
    train_seqs, eval_seqs = load_data()
    top_log(f"[data] train {train_seqs.shape}, eval {eval_seqs.shape}")

    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    for cfg in CONFIGS:
        label = cfg["label"]
        try:
            run_metrics = run_config(cfg, train_seqs, eval_seqs, device)
            results[label] = {
                "config": cfg,
                "eval_loss_final": run_metrics["eval_loss"][-1] if run_metrics["eval_loss"] else None,
                "best_induction_score": run_metrics.get("best_induction_score"),
                "minutes": run_metrics["minutes"],
                "steps": len(run_metrics["steps"]),
            }
        except Exception:
            top_log(f"[{label}] FAILED:\n{traceback.format_exc()}")
            results[label] = {"config": cfg, "error": traceback.format_exc()}
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    top_log("=== depth sweep complete ===")
    top_log(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
