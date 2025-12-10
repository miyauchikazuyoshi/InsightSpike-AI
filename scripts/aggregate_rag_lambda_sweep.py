#!/usr/bin/env python3
"""
Aggregate exp2to4_lite (RAG lite) results by lambda and optionally plot.

Reads JSON outputs from experiments/exp2to4_lite/src/run_suite.py and groups
the chosen baseline metrics by `lambda_weight`.

Example:
  python scripts/aggregate_rag_lambda_sweep.py \
    --dir experiments/exp2to4_lite/results \
    --glob "exp23_paper_lambda*_*.json" \
    --baseline gedig_ag_dg \
    --out results/rag-lambda/agg.json \
    --plot results/rag-lambda/plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

DEFAULT_KEYS = [
    "per_mean",
    "acceptance_rate",
    "fmr",
    "latency_p50",
    "latency_p95",
    "ag_rate",
    "dg_rate",
    "zsr",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="Directory containing run_suite JSON outputs")
    ap.add_argument("--glob", type=str, default="*.json", help="Glob pattern relative to --dir")
    ap.add_argument("--baseline", type=str, default="gedig_ag_dg", help="Baseline name to aggregate")
    ap.add_argument("--out", type=Path, required=True, help="Output aggregated JSON path")
    ap.add_argument("--plot", type=Path, default=None, help="Optional PNG path to plot lambda vs metrics")
    return ap.parse_args()


def load_metrics(path: Path, baseline: str) -> Tuple[float, Dict[str, float]]:
    data = json.loads(path.read_text())
    cfg = data.get("config", {}) or {}
    results = data.get("results", {}) or {}
    if baseline not in results:
        raise ValueError(f"Baseline '{baseline}' not found in {path}")
    res = results[baseline] or {}
    lam = cfg.get("lambda_weight", res.get("lambda_weight"))
    if lam is None:
        raise ValueError(f"lambda_weight not found in {path}")
    metrics: Dict[str, float] = {}
    for k in DEFAULT_KEYS:
        v = res.get(k)
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    return float(lam), metrics


def aggregate_by_lambda(files: List[Path], baseline: str) -> Dict[float, Dict[str, float]]:
    groups: Dict[float, List[Dict[str, float]]] = {}
    for p in files:
        lam, metrics = load_metrics(p, baseline)
        groups.setdefault(lam, []).append(metrics)

    aggregated: Dict[float, Dict[str, float]] = {}
    for lam, items in groups.items():
        out: Dict[str, float] = {"count": float(len(items))}
        for k in DEFAULT_KEYS:
            vals = [m[k] for m in items if k in m]
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
        xs = []
        ys = []
        for lam in lambdas:
            val = data[lam].get(key)
            if val is not None:
                xs.append(lam)
                ys.append(val)
        return xs, ys

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Acceptance / FMR
    x_acc, y_acc = series("acceptance_rate")
    if x_acc:
        axes[0].plot(x_acc, y_acc, marker="o", label="acceptance")
    x_fmr, y_fmr = series("fmr")
    if x_fmr:
        axes[0].plot(x_fmr, y_fmr, marker="x", label="fmr")
    axes[0].set_xlabel("lambda")
    axes[0].set_title("Acceptance / FMR")
    axes[0].legend()

    # Latency
    x_lat, y_lat = series("latency_p50")
    if x_lat:
        axes[1].plot(x_lat, y_lat, marker="o")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("latency_p50")
    axes[1].set_title("Latency P50")

    # PER
    x_per, y_per = series("per_mean")
    if x_per:
        axes[2].plot(x_per, y_per, marker="o")
    axes[2].set_xlabel("lambda")
    axes[2].set_ylabel("per_mean")
    axes[2].set_title("PER vs lambda")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"[plot] wrote {plot_path}")


def main() -> None:
    args = parse_args()
    files = sorted(args.dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files match {args.glob} under {args.dir}")

    aggregated = aggregate_by_lambda(files, args.baseline)
    output = {
        "source_dir": str(args.dir),
        "baseline": args.baseline,
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
