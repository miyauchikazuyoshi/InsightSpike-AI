import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Plot layer-wise geDIG metrics from a score JSON.")
    ap.add_argument("--in", dest="input_path", default="results/transformer_gedig/score_smoke.json", help="Input JSON path.")
    ap.add_argument("--out-dir", default="results/transformer_gedig", help="Output directory.")
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI for PNGs.")
    args = ap.parse_args()

    path = Path(args.input_path)
    if not path.exists():
        print(f"No input found: {path}")
        return

    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup style
    sns.set_theme(style="whitegrid")
    
    # 1. Layer-wise F Score (Boxplot)
    plt.figure(figsize=(12.4, 7.6))
    sns.boxplot(x="layer", y="F", data=df, palette="viridis")
    plt.title("Distribution of geDIG F-Score by Layer (BERT Base)")
    plt.xlabel("Layer Index")
    plt.ylabel("Free Energy (F)")
    plt.axhline(df["baseline_F_random"].mean(), color="r", linestyle="--", label="Random Baseline")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "layer_wise_f.png"
    plt.savefig(out_path, dpi=args.dpi)
    print(f"[done] saved {out_path}")

    if "head" in df.columns and df["head"].nunique() > 1:
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
        out_path = out_dir / "head_diversity_l0.png"
        plt.savefig(out_path, dpi=args.dpi)
        print(f"[done] saved {out_path}")
    
        # 3. Layer vs Head Heatmap (Mean F)
        plt.figure(figsize=(12, 8))
        pivot = df.pivot_table(index="layer", columns="head", values="F", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r") # reverse cmap so low F (good) is bright
        plt.title("Mean geDIG F-Score Heatmap (Layer x Head)")
        plt.tight_layout()
        out_path = out_dir / "layer_head_heatmap.png"
        plt.savefig(out_path, dpi=args.dpi)
        print(f"[done] saved {out_path}")

if __name__ == "__main__":
    main()
