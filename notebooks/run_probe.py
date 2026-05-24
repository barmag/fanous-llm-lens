import os
# Commented out: Strix Halo gfx1100 has native ROCm support; overriding to 11.5.1 causes segmentation faults.
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"

import json
import sys
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformer

def main():
    print("=== Environment Setup ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA/ROCm is not available. Running on CPU.")

    # 1. Load Dataset
    print("\n=== Loading Dataset ===")
    DATA_PATH = "../eval/prompts/msa-masri-pairs-v1.json"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "/home/yassermakram/code/fanous-llm-lens/eval/prompts/msa-masri-pairs-v1.json"
        
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Handle both schema variants
    prompts = dataset.get("prompts", dataset.get("pairs"))
    print(f"Loaded {len(prompts)} dialectal triples.")

    # 2. Load Model
    print("\n=== Loading Model (EleutherAI/pythia-160m-deduped) ===")
    MODEL_NAME = "EleutherAI/pythia-160m-deduped"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    print(f"Model layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, Dim: {model.cfg.d_model}")

    # Helper function for attention extraction
    def get_attention_matrix(text):
        tokens = model.to_tokens(text)
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        patterns = []
        for layer in range(model.cfg.n_layers):
            patterns.append(cache["pattern", layer].detach().cpu())
        return tokens.cpu(), patterns

    def compute_entropy(attn_matrix):
        eps = 1e-9
        entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + eps), dim=-1)
        return entropy

    def analyze_prompt_pair(msa_text, masri_text):
        _, msa_pat = get_attention_matrix(msa_text)
        _, masri_pat = get_attention_matrix(masri_text)
        
        results = []
        for layer in range(model.cfg.n_layers):
            m_attn = msa_pat[layer][0]
            ma_attn = masri_pat[layer][0]
            
            m_ent = compute_entropy(m_attn).mean(dim=-1)
            ma_ent = compute_entropy(ma_attn).mean(dim=-1)
            
            m_bos = m_attn[:, :, 0].mean(dim=-1)
            ma_bos = ma_attn[:, :, 0].mean(dim=-1)
            
            for head in range(model.cfg.n_heads):
                results.append({
                    "layer": layer,
                    "head": head,
                    "msa_entropy": m_ent[head].item(),
                    "masri_entropy": ma_ent[head].item(),
                    "msa_bos_attn": m_bos[head].item(),
                    "masri_bos_attn": ma_bos[head].item(),
                    "entropy_diff": ma_ent[head].item() - m_ent[head].item(),
                    "bos_attn_diff": ma_bos[head].item() - m_bos[head].item()
                })
                
        return pd.DataFrame(results)

    # 3. Running Analysis
    print("\n=== Running Attention Probing ===")
    all_dfs = []
    for i, p in enumerate(prompts):
        sys.stdout.write(f"\rProcessing pair {i+1}/{len(prompts)}... ")
        sys.stdout.flush()
        df_pair = analyze_prompt_pair(p["msa"], p["masri"])
        df_pair["prompt_id"] = p["id"]
        df_pair["category"] = p["category"]
        all_dfs.append(df_pair)
    print("Done!")

    summary_df = pd.concat(all_dfs, ignore_index=True)
    summary_df.to_csv("attention_divergence_summary.csv", index=False)
    print("Saved raw summary to attention_divergence_summary.csv")

    # Average metrics across all pairs
    grouped = summary_df.groupby(["layer", "head"]).mean(numeric_only=True).reset_index()

    # 4. Find Top Divergent Heads
    print("\n=== Top 10 Dialectal Divergent Heads (by Entropy Difference: Masri - MSA) ===")
    # Positive difference means more dispersed/uncertain attention for Masri
    top_pos_entropy = grouped.sort_values(by="entropy_diff", ascending=False).head(5)
    # Negative difference means more dispersed/uncertain attention for MSA
    top_neg_entropy = grouped.sort_values(by="entropy_diff", ascending=True).head(5)
    
    print("\n--- Higher Uncertainty in Masri (Possible fallback/struggle) ---")
    for _, row in top_pos_entropy.iterrows():
        print(f"Layer {int(row['layer'])} Head {int(row['head'])}: Entropy Delta = {row['entropy_diff']:.4f} (MSA: {row['msa_entropy']:.4f}, Masri: {row['masri_entropy']:.4f})")

    print("\n--- Higher Uncertainty in MSA ---")
    for _, row in top_neg_entropy.iterrows():
        print(f"Layer {int(row['layer'])} Head {int(row['head'])}: Entropy Delta = {row['entropy_diff']:.4f} (MSA: {row['msa_entropy']:.4f}, Masri: {row['masri_entropy']:.4f})")

    print("\n=== Top 5 Dialectal Divergent Heads (by BOS Attention Difference) ===")
    top_bos = grouped.sort_values(by="bos_attn_diff", key=abs, ascending=False).head(5)
    for _, row in top_bos.iterrows():
        print(f"Layer {int(row['layer'])} Head {int(row['head'])}: BOS Attn Delta = {row['bos_attn_diff']:.4f} (MSA: {row['msa_bos_attn']:.4f}, Masri: {row['masri_bos_attn']:.4f})")

    # 5. Visualizing
    print("\n=== Generating Heatmap Visualizations ===")
    entropy_pivot = grouped.pivot(index="layer", columns="head", values="entropy_diff")
    fig = px.imshow(
        entropy_pivot.values,
        labels=dict(x="Attention Head", y="Layer", color="Entropy Difference (Masri - MSA)"),
        x=entropy_pivot.columns,
        y=entropy_pivot.index,
        color_continuous_scale="RdBu",
        title="Attention Entropy Divergence (Masri vs MSA) across Pythia-160m"
    )
    
    # Save as HTML
    fig.write_html("divergence_heatmap.html")
    print("Saved heatmap visualization to divergence_heatmap.html")

if __name__ == "__main__":
    main()
