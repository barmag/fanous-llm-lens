"""Phase A (depth) — does adding layers reconstruct the morphology a tokenizer fused?

The verdict-after-the-verdict. Phase A ([run_probes.py](run_probes.py)) showed the zero-layer
pooled-mean probe is **insensitive to training** — it scores tokenization, not the model. This
driver adds 1–2 transformer blocks and measures the quantity that was ≈0 at zero layers: the
**trained − untrained increment**, per depth, per feature.

Hypothesis (see `docs/superpowers/specs/` / the hypothesis note): the increment stays ≈0 for
**definiteness** (a one-token artifact — nothing to reconstruct) and grows largest for **number on
`morphological`** (the tokenizer that fuses the `ون`/`ات` suffix into the stem), because depth has
to *rebuild* a feature the inventory threw away. This is an **isolated-word / intra-word-MLP**
test (words encoded alone; see train_depth_model.py docstring) — for number the only lever is MLP
nonlinearity on the stem token, so a null reads narrowly.

Method, mirroring Phase A where it can:
- **Same held-out word types, labels, controls, type-split** as run_probes.py (reused directly).
- **Per seed (3), paired:** construct the model, extract the **untrained** residual stream at
  layers 0/1/2, train the *same* model, extract again. Same-seed pairing → a clean per-seed Δ; the
  tokenizer build is cached once per approach, so seeds only multiply the fast training loop.
- **StandardScaler before the logistic probe** — load-bearing once depths differ in activation
  scale (Phase A could skip it: layer 0 was the only depth). Fixed split/seed so Δ is clean.
- Report Δ as **mean ± range across seeds** so a gain inside Phase A's ±0.014 noise floor isn't
  over-claimed.

Layer 0 here is the 2-layer model's *own* embeddings — **not** the Phase A zero-layer model (joint
training lets the blocks absorb work the embeddings carried alone), so it is reported as its own
datapoint, not as "reproduces Phase A."

Run: ``HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run --extra rocm --extra dev --extra tokenizers \\
    python experiments/embedding-probes/run_probes_depth.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Reuse Phase A's data/label machinery verbatim — same types, same controls, same split.
from run_probes import MIN_PER_CLASS, SEED, held_out_word_types, label_words
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from train_depth_model import (
    DepthConfig,
    DepthTransformer,
    build_token_stream,
    make_data,
    train_inplace,
)

from fanous_lens.tokenizers.corpora import load_corpora

APPROACHES = ["bpe", "unigram", "wordpiece", "morfessor", "morphological"]
SEEDS = [0, 1, 2]
N_LAYERS = 2
FEATURES = ["definite", "number", "length", "random"]  # headline: definite, number


def pooled_all_depths(
    model: DepthTransformer, encode, words: list[str], n_layers: int, device: str, chunk: int = 256
) -> dict[int, np.ndarray]:
    """Mean residual-stream vector per word at each depth 0..n_layers (isolated-word encoding).

    Words are short, so we pad each chunk to its max token length and pool only the real positions.
    Causal attention means a real position never attends to the trailing pads, so padding is inert.
    """
    d = model.embed.weight.shape[1]
    max_pos = model.pos.num_embeddings
    out = {layer: np.zeros((len(words), d), dtype=np.float32) for layer in range(n_layers + 1)}
    enc = [encode(w)[0] for w in words]  # ids only
    for start in range(0, len(words), chunk):
        block = enc[start : start + chunk]
        lens = [min(len(ids), max_pos) for ids in block]
        maxlen = max((ln for ln in lens), default=1) or 1
        arr = torch.zeros((len(block), maxlen), dtype=torch.long)
        for i, ids in enumerate(block):
            if lens[i]:
                arr[i, : lens[i]] = torch.tensor(ids[: lens[i]], dtype=torch.long)
        states = model.hidden_states(arr.to(device))
        for layer, h in enumerate(states):
            hc = h.cpu().numpy()
            for i, ln in enumerate(lens):
                if ln:
                    out[layer][start + i] = hc[i, :ln].mean(axis=0)
    return out


def probe_auc_scaled(x: np.ndarray, y: np.ndarray) -> float | None:
    """Type-split, **standardized** logistic-regression AUC; None if a class is too small."""
    mask = y >= 0
    x, y = x[mask], y[mask]
    if y.sum() < MIN_PER_CLASS or (len(y) - y.sum()) < MIN_PER_CLASS:
        return None
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(x_tr), y_tr)
    proba = clf.predict_proba(scaler.transform(x_te))[:, 1]
    return round(float(roc_auc_score(y_te, proba)), 3)


def harvest(model, encode, words, labels, node, tag: str, device: str) -> None:
    """Probe every (register, feature, depth) for one model state and append to ``node``."""
    for reg in ("MSA", "Masri"):
        pooled = pooled_all_depths(model, encode, words[reg], N_LAYERS, device)
        for ly in range(N_LAYERS + 1):
            for f in FEATURES:
                auc = probe_auc_scaled(pooled[ly], labels[reg][f])
                node[reg][f][ly][tag].append(auc)


def _stats(vals: list[float]) -> dict[str, float]:
    a = np.array(vals, dtype=float)
    return {
        "mean": round(float(a.mean()), 3),
        "lo": round(float(a.min()), 3),
        "hi": round(float(a.max()), 3),
    }


def main() -> None:
    words = held_out_word_types(n_msa=4_000, n_masri=4_000, skip=20_000)
    labels = {reg: label_words(words[reg]) for reg in ("MSA", "Masri")}
    for reg in ("MSA", "Masri"):
        d = labels[reg]
        print(
            f"{reg}: {len(words[reg])} types | definite +{int(d['definite'].sum())} | "
            f"plural +{int((d['number'] == 1).sum())} sing {int((d['number'] == 0).sum())}",
            flush=True,
        )

    msa, masri = load_corpora(max_msa=20_000, max_masri=10_000)
    corpus = msa + masri

    # raw[approach][reg][feature][layer]["trained"|"untrained"] = [auc per seed]
    raw: dict = {}
    for approach in APPROACHES:
        print(f"\n=== {approach}: building tokenizer (once) ===", flush=True)
        cfg0 = DepthConfig(approach=approach, n_layers=N_LAYERS)
        encode, _tc, ids = build_token_stream(approach, corpus, cfg0.vocab_size, cfg0.max_tokens)
        data = make_data(ids, cfg0.seq_len)

        node = {
            reg: {
                f: {ly: {"trained": [], "untrained": []} for ly in range(N_LAYERS + 1)}
                for f in FEATURES
            }
            for reg in ("MSA", "Masri")
        }
        for seed in SEEDS:
            cfg = DepthConfig(approach=approach, n_layers=N_LAYERS, seed=seed)
            torch.manual_seed(seed)
            model = DepthTransformer(
                cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.seq_len
            )
            model.to(cfg.device).eval()

            harvest(model, encode, words, labels, node, "untrained", cfg.device)
            losses = train_inplace(model, data, cfg)
            harvest(model, encode, words, labels, node, "trained", cfg.device)
            print(f"  seed {seed}: final loss {losses[-1]}", flush=True)
        raw[approach] = node

    # Aggregate: mean trained/untrained + Δ band across seeds.
    summary: dict = {}
    for approach, node in raw.items():
        summary[approach] = {}
        for reg in ("MSA", "Masri"):
            summary[approach][reg] = {}
            for f in FEATURES:
                summary[approach][reg][f] = {}
                for ly in range(N_LAYERS + 1):
                    # Pair per seed first, then keep seeds where both states gave an AUC.
                    pairs = [
                        (t, u)
                        for t, u in zip(
                            node[reg][f][ly]["trained"], node[reg][f][ly]["untrained"], strict=True
                        )
                        if t is not None and u is not None
                    ]
                    if not pairs:
                        summary[approach][reg][f][ly] = None
                        continue
                    tr = [t for t, _ in pairs]
                    un = [u for _, u in pairs]
                    deltas = [t - u for t, u in pairs]
                    summary[approach][reg][f][ly] = {
                        "trained": _stats(tr),
                        "untrained": _stats(un),
                        "delta": _stats(deltas),
                    }

    print("\n=== DEPTH PROBE: trained AUC (Δ vs untrained, mean across seeds) ===")
    for reg in ("MSA", "Masri"):
        print(f"\n[{reg}]  layers 0→{N_LAYERS}")
        hdr = f"{'approach':14}{'feature':>10}" + "".join(
            f"{'L' + str(ly):>16}" for ly in range(N_LAYERS + 1)
        )
        print(hdr)
        for approach in APPROACHES:
            for f in ("definite", "number"):
                cells = ""
                for ly in range(N_LAYERS + 1):
                    cell = summary[approach][reg][f][ly]
                    if cell is None:
                        cells += f"{'—':>16}"
                    else:
                        cells += (
                            f"{cell['trained']['mean']:.2f} (Δ{cell['delta']['mean']:+.2f})".rjust(
                                16
                            )
                        )
                print(f"{approach:14}{f:>10}{cells}")

    out_path = Path(__file__).parent / "phase_a_depth_results.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
