import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def main():
    path = Path("results/transformer_gedig/score_smoke.json")
    if not path.exists():
        print("No score_smoke.json found")
        return

    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    
    # Setup style
    sns.set_theme(style="whitegrid")
    
    # 1. Layer-wise F Score (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="layer", y="F", data=df, palette="viridis")
    plt.title("Distribution of geDIG F-Score by Layer (BERT Base)")
    plt.xlabel("Layer Index")
    plt.ylabel("Free Energy (F)")
    plt.axhline(df["baseline_F_random"].mean(), color="r", linestyle="--", label="Random Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/transformer_gedig/layer_wise_f.png")
    print("[done] saved layer_wise_f.png")

    # 2. Head Diversity (Layer 0)
    plt.figure(figsize=(12, 6))
    l0_df = df[df["layer"] == 0]
    sns.boxplot(x="head", y="F", data=l0_df, palette="coolwarm")
    plt.title("Head Diversity in Layer 0: geDIG F-Score")
    plt.xlabel("Head Index")
    plt.ylabel("Free Energy (F)")
    plt.axhline(l0_df["baseline_F_random"].mean(), color="r", linestyle="--", label="Random Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/transformer_gedig/head_diversity_l0.png")
    print("[done] saved head_diversity_l0.png")
    
    # 3. Layer vs Head Heatmap (Mean F)
    plt.figure(figsize=(12, 8))
    pivot = df.pivot_table(index="layer", columns="head", values="F", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r") # reverse cmap so low F (good) is bright
    plt.title("Mean geDIG F-Score Heatmap (Layer x Head)")
    plt.tight_layout()
    plt.savefig("results/transformer_gedig/layer_head_heatmap.png")
    print("[done] saved layer_head_heatmap.png")

if __name__ == "__main__":
    main()
