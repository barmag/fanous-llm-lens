"""Phase A control — probe UNTRAINED embeddings to separate inventory from learning.

A pooled-word probe can score high for a trivial reason: if a tokenizer gives every
definite word a shared ``ال`` token, the definite pools carry a constant offset that a probe
recovers **even with randomly-initialised embeddings** — no training required. That is the
Phase-A form of the Phase-B tautology (``morphological`` matching the gold by construction).

This script probes a *fresh, untrained* ``W_E`` (same shape, same tokenizer) with the same
labels and split as :mod:`run_probes`. The informative quantity is the **trained − untrained
increment**: how much the model actually *learned* to encode the feature, above what the
tokenizer's inventory hands it for free. CPU-only; reads the trained AUCs from
``phase_a_results.json`` and the tokenizer configs from the saved checkpoints.

Run: ``uv run --extra rocm --extra dev --extra tokenizers \\
    python experiments/embedding-probes/run_baseline.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import run_probes as R
import torch
from train_embedding_model import ZeroLayerTransformer

from fanous_lens.tokenizers.train import get_tokenizer

CKPT = Path(__file__).parent / "checkpoints"
FEATURES = ["definite", "number"]


def main() -> None:
    trained = json.loads((Path(__file__).parent / "phase_a_results.json").read_text())
    words = R.held_out_word_types(n_msa=4_000, n_masri=4_000, skip=20_000)
    labels = {reg: R.label_words(words[reg]) for reg in ("MSA", "Masri")}

    rows: dict[str, dict] = {}
    for approach in R.APPROACHES:
        ckpt = torch.load(CKPT / f"{approach}_zerolayer.pt", weights_only=False)
        vocab_size, d_model = ckpt["vocab_size"], ckpt["d_model"]
        encode = get_tokenizer(approach, ckpt["tok_config"])
        torch.manual_seed(R.SEED)
        untrained_we = ZeroLayerTransformer(vocab_size, d_model).embed.weight.detach()
        rows[approach] = {}
        for reg in ("MSA", "Masri"):
            vecs = R.pooled_vectors(untrained_we, encode, words[reg])
            rows[approach][reg] = {f: R.probe_auc(vecs, labels[reg][f]) for f in FEATURES}

    print("\n=== UNTRAINED baseline AUC  (trained → from phase_a_results.json) ===")
    hdr = f"{'approach':14}{'reg':>7}" + "".join(f"{f + ' (Δ)':>18}" for f in FEATURES)
    print(hdr)
    for approach in R.APPROACHES:
        for reg in ("MSA", "Masri"):
            cells = ""
            for f in FEATURES:
                u = rows[approach][reg][f]
                t = trained[approach][reg][f]
                delta = round(t - u, 3) if (u is not None and t is not None) else None
                cells += f"{f'{u}→{t} ({delta:+})':>18}" if delta is not None else f"{'—':>18}"
            print(f"{approach:14}{reg:>7}{cells}")
    (Path(__file__).parent / "phase_a_baseline.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    print("\nJSON:\n" + json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
