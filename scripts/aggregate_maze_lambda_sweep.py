#!/usr/bin/env python3
"""
Aggregate maze query-hub summaries by lambda and optionally plot.

Reads per-run summary JSONs produced by
`experiments/maze-query-hub-prototype/run_experiment_query.py`
and groups them by `lambda_weight` (taken from summary → config fallback).

Example:
  python scripts/aggregate_maze_lambda_sweep.py \
    --dir results/maze-lambda-sweep \
    --glob "*_summary.json" \
    --out results/maze-lambda-sweep/maze_lambda_agg.json \
    --plot results/maze-lambda-sweep/maze_lambda_plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

DEFAULT_KEYS = [
    "success_rate",
    "avg_steps",
    "avg_edges",
    "g0_mean",
    "gmin_mean",
    "avg_k_star",
    "avg_delta_sp",
    "avg_delta_sp_min",
    "best_hop_mean",
    "avg_time_ms_eval",
    "p95_time_ms_eval",
    # PSZ metrics (if present)
    "psz_acceptance_rate",
    "psz_fmr",
    "psz_latency_p50_ms",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="Directory containing *_summary.json files")
    ap.add_argument("--glob", type=str, default="*_summary.json", help="Glob pattern relative to --dir")
    ap.add_argument("--out", type=Path, required=True, help="Output aggregated JSON path")
    ap.add_argument("--plot", type=Path, default=None, help="Optional PNG path to plot lambda vs metrics")
    return ap.parse_args()


def load_summary(path: Path) -> Dict:
    data = json.loads(path.read_text())
    summary = data.get("summary", data)
    lam = summary.get("lambda_weight")
    if lam is None:
        try:
            lam = data.get("config", {}).get("gedig", {}).get("lambda_weight")
        except Exception:
            lam = None
    if lam is None:
        raise ValueError(f"lambda_weight not found in {path}")
    return float(lam), summary


def aggregate_by_lambda(files: List[Path]) -> Dict[float, Dict[str, float]]:
    groups: Dict[float, List[Dict[str, float]]] = {}
    for p in files:
        lam, summary = load_summary(p)
        numeric: Dict[str, float] = {}
        for k in DEFAULT_KEYS:
            v = summary.get(k)
            if isinstance(v, (int, float)):
                numeric[k] = float(v)
        groups.setdefault(lam, []).append(numeric)

    aggregated: Dict[float, Dict[str, float]] = {}
    for lam, items in groups.items():
        out: Dict[str, float] = {"count": float(len(items))}
        for k in DEFAULT_KEYS:
            vals = [itm[k] for itm in items if k in itm]
            if vals:
                out[k] = float(mean(vals))
        aggregated[lam] = out
    return aggregated


def maybe_plot(data: Dict[float, Dict[str, float]], plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped (matplotlib not available: {e})")
        return

    lambdas = sorted(data.keys())
    if not lambdas:
        print("[plot] no data to plot")
        return

    def series(key: str):
        vals = [data[l].get(key) for l in lambdas]
        return [v for v in vals], [l for v, l in zip(vals, lambdas) if v is not None]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Success / steps
    y1, x1 = series("success_rate")
    if x1:
        axes[0].plot(sorted(x1), [data[l]["success_rate"] for l in sorted(x1)], marker="o")
        axes[0].set_ylabel("success_rate")
    y_steps, x_steps = series("avg_steps")
    if x_steps:
        axes[0].twinx().plot(sorted(x_steps), [data[l]["avg_steps"] for l in sorted(x_steps)], marker="x", color="C1")
        axes[0].set_ylabel("success_rate / avg_steps")
    axes[0].set_xlabel("lambda")
    axes[0].set_title("Success vs lambda")

    # PSZ / FMR if available
    y_fmr, x_fmr = series("psz_fmr")
    if x_fmr:
        axes[1].plot(sorted(x_fmr), [data[l]["psz_fmr"] for l in sorted(x_fmr)], marker="o")
    y_acc, x_acc = series("psz_acceptance_rate")
    if x_acc:
        axes[1].twinx().plot(sorted(x_acc), [data[l]["psz_acceptance_rate"] for l in sorted(x_acc)], marker="x", color="C1")
    axes[1].set_xlabel("lambda")
    axes[1].set_title("PSZ (if present)")

    # Eval time
    y_eval, x_eval = series("avg_time_ms_eval")
    if x_eval:
        axes[2].plot(sorted(x_eval), [data[l]["avg_time_ms_eval"] for l in sorted(x_eval)], marker="o")
    axes[2].set_xlabel("lambda")
    axes[2].set_ylabel("avg_time_ms_eval")
    axes[2].set_title("Latency vs lambda")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"[plot] wrote {plot_path}")


def main() -> None:
    args = parse_args()
    files = sorted(args.dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files match {args.glob} under {args.dir}")

    aggregated = aggregate_by_lambda(files)
    output = {
        "source_dir": str(args.dir),
        "files": [str(f) for f in files],
        "lambda_groups": [
            {"lambda_weight": lam, **metrics} for lam, metrics in sorted(aggregated.items(), key=lambda x: x[0])
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"[aggregate] wrote {args.out} (lambdas={len(aggregated)})")

    if args.plot:
        maybe_plot({lam: metrics for lam, metrics in aggregated.items()}, args.plot)


if __name__ == "__main__":
    main()
