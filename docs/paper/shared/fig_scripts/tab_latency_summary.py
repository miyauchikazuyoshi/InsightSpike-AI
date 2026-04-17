#!/usr/bin/env python3
"""
Tab. Latency‑Summary (A8): P50/P95/P99 latency by (domain, H, k)

Input CSVs (optional):
  - data/latency/maze_latency.csv
  - data/latency/rag_latency.csv
  columns: run_id, H, k, latency_ms

Outputs:
  - docs/paper/templates/tab_latency_summary.tex (LaTeX tabular)
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd


def load_csv(path: Path, domain: str) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        req = {"H", "k", "latency_ms"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns in {path}: {miss}")
    else:
        # Synthetic placeholder grid
        Hs = [1, 2, 3]
        ks = [4, 8, 12]
        rows = []
        rng = np.random.default_rng(7 if domain == "maze" else 9)
        for H in Hs:
            for k in ks:
                samples = rng.normal(80 + 40*H + 5*k, 20 + 6*H + 2*k, size=64)
                for s in samples:
                    rows.append({"H": H, "k": k, "latency_ms": max(5, float(s))})
        df = pd.DataFrame(rows)
    df["domain"] = domain
    return df


def percentile_stats(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series({
            "P50": np.percentile(g["latency_ms"], 50),
            "P95": np.percentile(g["latency_ms"], 95),
            "P99": np.percentile(g["latency_ms"], 99),
            "n": len(g),
        })
    return df.groupby(["domain", "H", "k"], as_index=False).apply(agg).reset_index(drop=True)


def to_latex_tables(df: pd.DataFrame) -> str:
    lines = []
    for domain in ["maze", "rag"]:
        sub = df[df["domain"] == domain]
        lines.append(f"% Latency summary for {domain}")
        lines.append("\\begin{table}[ht]")
        lines.append("  \\centering")
        lines.append(f"  \\caption{{追加レイテンシ分布（{domain.upper()}，P50/P95/P99）}}")
        lines.append("  \\begin{tabular}{lrrrr}")
        lines.append("    \\toprule")
        # LaTeX newline requires \\; escape backslashes in Python string
        lines.append("    H×k & P50 (ms) & P95 (ms) & P99 (ms) & n \\\\ ")
        lines.append("    \\midrule")
        for (H, k), g in sub.groupby(["H", "k"]):
            P50, P95, P99, n = g[["P50", "P95", "P99", "n"]].iloc[0]
            lines.append(f"    {H}$\\times${k} & {P50:0.0f} & {P95:0.0f} & {P99:0.0f} & {int(n)} \\\\ ")
        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[3]
    in_maze = Path(os.environ.get("LAT_MAZE", repo_root / "data/latency/maze_latency.csv"))
    in_rag = Path(os.environ.get("LAT_RAG", repo_root / "data/latency/rag_latency.csv"))
    df = pd.concat([
        load_csv(in_maze, "maze"),
        load_csv(in_rag, "rag"),
    ], ignore_index=True)
    stats = percentile_stats(df)

    outdir = repo_root / "docs/paper/templates"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "tab_latency_summary.tex"
    out.write_text(to_latex_tables(stats), encoding="utf-8")
    print(f"✅ Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
