#!/usr/bin/env python3
"""
Plot correlation between checkpoint accuracy and hop-wise subgraph F_mean (CLS anchor).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/transformer_gedig/score_phase4_poc.json"))
    ap.add_argument("--summary", type=Path, default=Path("results/transformer_gedig/checkpoints/distilbert_sst2_poc/run_summary.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures/fig_phase4_hopcorr.png"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.DataFrame(json.loads(args.scores.read_text()))
    summary = json.loads(args.summary.read_text())
    accs = summary.get("checkpoints", {})
    sub = df[df.get("subgraph") == True]

    rows = []
    for (m, hop), g in sub.groupby(["model", "hop"]):
        acc = accs.get(m)
        if acc is None:
            continue
        rows.append({"checkpoint": m, "hop": float(hop), "accuracy": float(acc), "F_mean": g["F"].mean()})
    if not rows:
        raise SystemExit("no subgraph rows")
    hops = sorted(set(r["hop"] for r in rows))
    corr_vals = []
    for h in hops:
        sel = [r for r in rows if r["hop"] == h]
        if len(sel) < 2:
            corr_vals.append(0.0)
            continue
        acc = np.array([r["accuracy"] for r in sel])
        f = np.array([r["F_mean"] for r in sel])
        corr = float(np.corrcoef(acc, f)[0, 1])
        corr_vals.append(corr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(int(h)) for h in hops], corr_vals)
    ax.set_xlabel("hop (CLS anchor)")
    ax.set_ylabel("corr(acc, F_mean)")
    ax.set_ylim(-1, 1)
    ax.set_title("Checkpoint accuracy vs subgraph F_mean")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
