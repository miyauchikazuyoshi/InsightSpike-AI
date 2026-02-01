import json
import pandas as pd
from pathlib import Path

def analyze():
    path = Path("results/transformer_gedig/score_smoke.json")
    if not path.exists():
        print("No score_smoke.json found")
        return

    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    
    print("--- Overall Stats ---")
    print(df[["F", "E_eff", "delta_epc", "delta_h"]].describe())
    
    print("\n--- Average F by Layer ---")
    layer_stats = df.groupby("layer")["F"].agg(["mean", "std", "count"])
    print(layer_stats)
    
    print("\n--- Average F by Head (Layer 0) ---")
    head_stats = df[df["layer"] == 0].groupby("head")["F"].agg(["mean", "std"])
    print(head_stats)

    print("\n--- Baseline Comparison ---")
    # Compare mean F of real attention vs baselines
    real_f = df["F"].mean()
    rand_f = df["baseline_F_random"].mean()
    uni_f = df["baseline_F_uniform"].mean()
    loc_f = df["baseline_F_local_w5"].mean()
    diag_f = df["baseline_F_diagonal"].mean()
    
    print(f"Real: {real_f:.4f}")
    print(f"Random: {rand_f:.4f}")
    print(f"Uniform: {uni_f:.4f}")
    print(f"Local: {loc_f:.4f}")
    print(f"Diagonal: {diag_f:.4f}")

if __name__ == "__main__":
    analyze()
