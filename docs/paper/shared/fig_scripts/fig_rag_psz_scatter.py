#!/usr/bin/env python3
"""
Fig. R‑PSZ‑Scatter: Acceptance vs Additional Latency with FMR color (A7)

Expected input CSV (default: data/rag_eval/psz_points.csv):
  columns: run_id, query_id, H, k, PER, acceptance, FMR, latency_ms

Note: Paper defines PSZ latency as P50 of additional latency (\le 200ms).
This scatter plots per‑query additional latency (x) as an orientation aid;
the PSZ gating itself should be read from the latency summary table.

If the CSV is missing, the script will generate a small synthetic dataset
to produce a placeholder figure so that the paper can compile.
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        # Basic sanity columns
        required = {"PER", "acceptance", "FMR"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        # Optional H/k
        if "H" not in df.columns:
            df["H"] = 3
        if "k" not in df.columns:
            df["k"] = 8
        return df
    # Fallback synthetic
    rng = np.random.default_rng(42)
    n = 120
    H = rng.choice([1, 2, 3], size=n, p=[0.3, 0.3, 0.4])
    PER = rng.normal(140, 20, size=n).clip(80, 200)
    acceptance = (rng.normal(0.9, 0.08, size=n)).clip(0.4, 1.0)
    # Color by FMR (prefer low)
    FMR = (rng.beta(1.2, 25, size=n)).clip(0.0, 0.15)
    df = pd.DataFrame({
        "run_id": np.arange(n),
        "query_id": np.arange(n),
        "H": H,
        "k": rng.choice([4, 8, 12], size=n),
        "PER": PER,
        "acceptance": acceptance,
        "FMR": FMR,
        "latency_ms": rng.normal(180, 60, size=n).clip(40, 520),
    })
    return df


def main():
    root = Path(__file__).resolve().parents[0]
    # /repo/docs/paper/fig_scripts -> repo_root = parents[2]
    repo_root = root.parents[2]
    outdir = repo_root / "docs/paper/figures"
    outdir.mkdir(parents=True, exist_ok=True)

    in_path = Path(os.environ.get("PSZ_INPUT", repo_root / "data/rag_eval/psz_points.csv"))
    df = load_data(in_path)

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    # Ensure white background for PDF embedding
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # Scatter: x=additional latency (ms), y=acceptance, color=FMR, marker by H
    markers = {1: "o", 2: "s", 3: "D"}
    cmap = plt.get_cmap("viridis_r")  # lower FMR looks better (dark-blue)
    for h in sorted(df["H"].unique()):
        sub = df[df["H"] == h]
        x = sub.get("latency_ms", pd.Series([np.nan]*len(sub)))
        sc = ax.scatter(
            x,
            sub["acceptance"] * 100.0,
            c=sub["FMR"],
            cmap=cmap,
            vmin=0.0,
            vmax=max(0.02, float(df["FMR"].quantile(0.98))),
            s=36,
            marker=markers.get(int(h), "o"),
            alpha=0.9,
            label=f"H={h}",
        )

    # PSZ band and guideline overlay
    # Horizontal band: acceptance ≥95%
    ax.axhspan(95, 105, color="tab:green", alpha=0.10, label="PSZ: acc≥95%")
    # Vertical guide for latency (P50 threshold 200ms)
    ax.axvline(200, color="gray", linestyle="--", lw=1.2, alpha=0.7)

    # PSZ rectangle (acceptance≥95% & latency≤200ms)
    y_min, y_max = 95, 105
    ax.fill_betweenx([y_min, y_max], 0, 200, color="tab:green", alpha=0.08, zorder=0)
    # Annotate PSZ conditions inside the rectangle
    ax.text(12, 96.2,
            "PSZ band\nacc ≥ 95%\nFMR ≤ 2% (color)\nP50 ≤ 200ms",
            ha="left", va="bottom", fontsize=9, color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.25"))
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("FMR (False Merge Rate)")
    ax.set_xlabel("Additional Latency (ms)")
    ax.set_ylabel("Acceptance (%)")
    ax.set_title("Acceptance vs Additional Latency with FMR coloring (R‑PSZ‑Scatter)")
    ax.legend(title="Hop H", frameon=False, loc="lower right")
    # X/Y limits: allow wide latency; acceptance in [0,100]
    try:
        x_max = float(pd.to_numeric(df["latency_ms"], errors="coerce").max())
    except Exception:
        x_max = 500.0
    ax.set_xlim(0, max(500.0, x_max * 1.05))
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    out_pdf = outdir / "fig7_psz_scatter.pdf"
    out_png = outdir / "fig7_psz_scatter.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"✅ Wrote {out_pdf}")


if __name__ == "__main__":
    sys.exit(main())
