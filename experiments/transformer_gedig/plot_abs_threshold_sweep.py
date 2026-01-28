#!/usr/bin/env python3
"""
Plot absolute-threshold sweep (tau) for geDIG metrics.

Reads multiple score_abs_tau*_l12.json files, filters fixed-threshold + head_agg=mean,
and plots mean EPC/SP/H/F vs tau on a single compact figure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean
from typing import List, Tuple

import matplotlib.pyplot as plt


def _parse_tau(path: str) -> float:
    base = os.path.basename(path)
    # expected: score_abs_tau0.05_l12.json or score_abs_tau0.01_l12.json
    token = base.replace("score_abs_tau", "").replace("_l12.json", "")
    try:
        return float(token)
    except ValueError:
        return float("nan")


def _summarize(path: str) -> Tuple[float, float, float, float, float]:
    rows = json.loads(open(path, encoding="utf-8").read())
    fixed = [r for r in rows if not r.get("use_percentile") and r.get("head_agg") == "mean"]
    if not fixed:
        return (float("nan"),) * 5
    avg = lambda k: mean(r[k] for r in fixed)
    return (
        _parse_tau(path),
        avg("delta_epc"),
        avg("delta_sp"),
        avg("delta_h"),
        avg("F"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        default="results/transformer_gedig/score_abs_tau*_l12.json",
        help="Glob pattern for score files.",
    )
    ap.add_argument(
        "--out",
        default="results/transformer_gedig/fig_abs_threshold_sweep_l12.png",
        help="Output PNG path.",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[warn] no files matched: {args.pattern}")
        return

    rows: List[Tuple[float, float, float, float, float]] = []
    for path in files:
        rows.append(_summarize(path))

    rows = [r for r in rows if r[0] == r[0]]  # filter NaN taus
    rows.sort(key=lambda x: x[0])

    taus = [r[0] for r in rows]
    epc = [r[1] for r in rows]
    sp = [r[2] for r in rows]
    h = [r[3] for r in rows]
    f = [r[4] for r in rows]

    plt.figure(figsize=(6.0, 3.4))
    plt.plot(taus, epc, marker="o", label="EPC (edge density)")
    plt.plot(taus, sp, marker="s", label="SP (path efficiency)")
    plt.plot(taus, f, marker="^", label="F")
    plt.plot(taus, h, marker="x", linestyle="--", alpha=0.7, label="H (entropy)")

    plt.xlabel("Absolute threshold τ")
    plt.ylabel("Mean metric (12 layers, n=8 texts)")
    plt.title("Absolute-threshold sweep (BERT Base, L=12)")
    plt.ylim(-0.6, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"[done] saved {args.out}")


if __name__ == "__main__":
    main()
