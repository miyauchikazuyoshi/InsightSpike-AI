#!/usr/bin/env python3
"""
Compare two maze runs (baseline vs variant) and print deltas.

Usage:
  python experiments/maze-query-hub-prototype/baselines/compare_runs.py \
    --base-summary path/to/base_summary.json --base-steps path/to/base_steps.json \
    --var-summary  path/to/var_summary.json  --var-steps  path/to/var_steps.json

Outputs a JSON diff (key metrics) and prints a small table to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def key_metrics(summary: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, float]:
    ssum = summary.get("summary") or {}
    out = {
        "success_rate": float(ssum.get("success_rate", 0.0)),
        "avg_steps": float(ssum.get("avg_steps", 0.0)),
        "avg_edges": float(ssum.get("avg_edges", 0.0)),
        "g0_mean": float(ssum.get("g0_mean", 0.0)),
        "gmin_mean": float(ssum.get("gmin_mean", 0.0)),
        "avg_time_ms_eval": float(ssum.get("avg_time_ms_eval", 0.0)),
    }
    ag = sum(1 for r in steps if r.get("ag_fire")) / (len(steps) or 1)
    dg = sum(1 for r in steps if r.get("dg_fire")) / (len(steps) or 1)
    out["ag_rate"] = float(ag)
    out["dg_rate"] = float(dg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two maze runs")
    ap.add_argument("--base-summary", type=Path, required=True)
    ap.add_argument("--base-steps", type=Path, required=True)
    ap.add_argument("--var-summary", type=Path, required=True)
    ap.add_argument("--var-steps", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    base_s = load_json(args.base_summary)
    base_t = load_json(args.base_steps)
    var_s = load_json(args.var_summary)
    var_t = load_json(args.var_steps)

    km_base = key_metrics(base_s, base_t)
    km_var = key_metrics(var_s, var_t)
    delta = {k: float(km_var.get(k, 0.0) - km_base.get(k, 0.0)) for k in sorted(set(km_base) | set(km_var))}
    out = {"baseline": km_base, "variant": km_var, "delta": delta}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

