#!/usr/bin/env python3
"""
Plot F_mean vs checkpoint accuracy for Phase4 PoC (DistilBERT SST-2 tiny run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/transformer_gedig/score_phase4_poc.json"), help="geDIG scores JSON")
    ap.add_argument("--summary", type=Path, default=Path("results/transformer_gedig/checkpoints/distilbert_sst2_poc/run_summary.json"), help="checkpoint summary with accuracies")
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures/fig_phase4_checkpoints.png"), help="Output PNG")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.DataFrame(json.loads(args.scores.read_text()))
    summary = json.loads(args.summary.read_text())
    accs = summary.get("checkpoints", {})

    rows = []
    for name, acc in accs.items():
        g = df[df["model"] == name]
        if g.empty:
            continue
        rows.append({"checkpoint": name, "accuracy": acc, "F_mean": g["F"].mean()})
    if not rows:
        raise SystemExit("no matching checkpoints in scores")
    rows = sorted(rows, key=lambda r: int(r["checkpoint"].split("-")[-1]))

    labels = [r["checkpoint"] for r in rows]
    acc_vals = [r["accuracy"] for r in rows]
    f_vals = [r["F_mean"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(labels, acc_vals, marker="o", color="C0", label="accuracy")
    ax2.plot(labels, f_vals, marker="x", color="C1", label="F_mean")
    ax1.set_ylabel("accuracy", color="C0")
    ax2.set_ylabel("F_mean", color="C1")
    ax1.set_xlabel("checkpoint")
    fig.suptitle("Phase4 PoC: checkpoint accuracy vs F_mean")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
