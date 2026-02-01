#!/usr/bin/env python3
"""
Aggregate score_smoke.json to compare real F vs baselines.
Outputs simple stats: mean F, mean baseline_F_random, delta, effect size (Cohen's d vs random).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2) if (na + nb - 2) > 0 else 0.0
    return (np.mean(a) - np.mean(b)) / np.sqrt(pooled + 1e-12)


def aggregate(path: Path) -> Dict:
    records = json.loads(path.read_text())
    df = pd.DataFrame(records)
    # real vs random baseline
    f_real = df["F"].to_numpy()
    f_rand = df["baseline_F_random"].to_numpy()
    summary = {
        "rows": len(df),
        "F_mean": float(np.mean(f_real)),
        "F_std": float(np.std(f_real)),
        "F_random_mean": float(np.mean(f_rand)),
        "F_random_std": float(np.std(f_rand)),
        "delta_F_real_minus_random": float(np.mean(f_real) - np.mean(f_rand)),
        "cohen_d_real_vs_random": float(cohen_d(f_real, f_rand)),
    }
    # add per-model/threshold slice to quickly spot where gains show up
    grouped: Dict[str, Dict[str, float]] = {}
    for (model, use_pct), g in df.groupby(["model", "use_percentile"]):
        fr = g["F"].to_numpy()
        br = g["baseline_F_random"].to_numpy()
        grouped[f"{model}|percentile={use_pct}"] = {
            "rows": len(g),
            "F_mean": float(np.mean(fr)),
            "F_random_mean": float(np.mean(br)),
            "delta_F_real_minus_random": float(np.mean(fr) - np.mean(br)),
            "cohen_d_real_vs_random": float(cohen_d(fr, br)),
        }
    summary["by_model_threshold"] = grouped
    return summary


def main() -> None:
    src = Path("results/transformer_gedig/score_smoke.json")
    out = Path("results/transformer_gedig/score_baseline_agg.json")
    summary = aggregate(src)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
