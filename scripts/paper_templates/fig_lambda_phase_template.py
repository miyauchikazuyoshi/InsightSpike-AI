#!/usr/bin/env python3
"""
Minimal lambda-scan plotting template (Maze/RAG).

Usage examples:
  python scripts/paper_templates/fig_lambda_phase_template.py \
    --agg results/maze-lambda/maze15_s80_agg.json \
    --kind maze \
    --out docs/paper/figures/fig_lambda_maze.png

  python scripts/paper_templates/fig_lambda_phase_template.py \
    --agg results/rag-lambda/agg.json \
    --kind rag \
    --out docs/paper/figures/fig_lambda_rag.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=Path, required=True, help="Aggregated JSON from lambda sweep")
    ap.add_argument("--kind", type=str, choices=["maze", "rag"], required=True, help="Plot presets: maze or rag")
    ap.add_argument("--out", type=Path, required=True, help="Output PNG path")
    return ap.parse_args()


def load_groups(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    groups = data.get("lambda_groups") or []
    return sorted(groups, key=lambda d: d.get("lambda_weight", 0.0))


def _plot_maze(groups: List[Dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    lam = [float(g["lambda_weight"]) for g in groups]
    success = [g.get("success_rate") for g in groups]
    fmr = [g.get("psz_fmr") for g in groups]
    time_p50 = [g.get("psz_latency_p50_ms") for g in groups]
    gedmin = [g.get("avg_ged_min_proxy") for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(lam, success, marker="o")
    axes[0].set_ylabel("success_rate")
    axes[0].set_xlabel("lambda")
    axes[0].set_title("Maze success")

    axes[1].plot(lam, fmr, marker="x", label="psz_fmr")
    axes[1].set_xlabel("lambda")
    axes[1].legend()
    axes[1].set_title("FMR (PSZ)")

    axes[2].plot(lam, time_p50, marker="o", label="latency_p50_ms")
    if any(gedmin):
        axes[2].twinx().plot(lam, gedmin, marker="x", color="C2", label="ged_min_proxy")
    axes[2].set_xlabel("lambda")
    axes[2].set_title("Latency / ged_min_proxy")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")


def _plot_rag(groups: List[Dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    lam = [float(g["lambda_weight"]) for g in groups]
    acc = [g.get("acceptance_rate") for g in groups]
    fmr = [g.get("fmr") for g in groups]
    lat = [g.get("latency_p50") for g in groups]
    per = [g.get("per_mean") for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(lam, acc, marker="o", label="acceptance_rate")
    axes[0].plot(lam, fmr, marker="x", label="fmr")
    axes[0].legend()
    axes[0].set_xlabel("lambda")
    axes[0].set_title("Acc / FMR")

    axes[1].plot(lam, lat, marker="o")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("latency_p50")
    axes[1].set_title("Latency P50")

    axes[2].plot(lam, per, marker="o")
    axes[2].set_xlabel("lambda")
    axes[2].set_ylabel("per_mean")
    axes[2].set_title("PER vs lambda")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")


def main() -> None:
    args = parse_args()
    groups = load_groups(args.agg)
    if not groups:
        raise SystemExit(f"No lambda_groups in {args.agg}")
    if args.kind == "maze":
        _plot_maze(groups, args.out)
    else:
        _plot_rag(groups, args.out)


if __name__ == "__main__":
    main()
