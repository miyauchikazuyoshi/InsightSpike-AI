#!/usr/bin/env python3
"""
Aggregate maze batch results into a single JSON for paper tables.

- Scans a directory for per-seed summary/step logs
  e.g., paper25_25x25_s500_seed*_summary.json and *_steps.json
- Computes mean of key metrics across seeds
- Optionally computes AG/DG rates from step logs (fraction of steps with ag_fire/dg_fire)
- Emits a compact JSON suitable for docs/paper/data/*.json

Usage:
  python scripts/aggregate_maze_batch.py \
    --dir experiments/maze-query-hub-prototype/results/batch_25x25 \
    --prefix paper25_25x25_s500_eval_seed \
    --out docs/paper/data/maze_25x25_eval_s500.json

  python scripts/aggregate_maze_batch.py \
    --dir experiments/maze-query-hub-prototype/results/batch_25x25 \
    --prefix paper25_25x25_s500_seed \
    --out docs/paper/data/maze_25x25_l3_s500.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


KEYS_MEAN = [
    "success_rate",
    "avg_steps",
    "avg_edges",
    "g0_mean",
    "gmin_mean",
    "avg_time_ms_eval",
    "p95_time_ms_eval",
    "avg_k_star",
    "avg_delta_sp",
    "avg_delta_sp_min",
    "best_hop_mean",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="Batch directory path")
    ap.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Filename prefix up to the seed, e.g., 'paper25_25x25_s500_seed' or '..._eval_seed'",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    return ap.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def aggregate(dir_path: Path, prefix: str) -> Dict[str, float]:
    # Summary files
    summary_files = sorted(dir_path.glob(f"{prefix}*_summary.json"))
    step_files = sorted(dir_path.glob(f"{prefix}*_steps.json"))
    if not summary_files:
        raise SystemExit(f"No summary files found for prefix '{prefix}' in {dir_path}")

    summaries: List[dict] = []
    for p in summary_files:
        d = load_json(p)
        s = d.get("summary", d)
        summaries.append(s)

    out: Dict[str, float] = {
        "note": f"Aggregated across {len(summaries)} seeds from {dir_path}/{prefix}*_summary.json",
        "seeds": len(summaries),
    }

    for k in KEYS_MEAN:
        vals = [s[k] for s in summaries if k in s]
        if vals:
            out[k] = float(mean(vals))

    # Optional AG/DG rates from step logs (fraction per seed, then averaged)
    if step_files and len(step_files) == len(summaries):
        ag_rates: List[float] = []
        dg_rates: List[float] = []
        for p in step_files:
            steps = load_json(p)
            if not isinstance(steps, list) or not steps:
                continue
            n = len(steps)
            ag = sum(1 for r in steps if isinstance(r, dict) and r.get("ag_fire"))
            dg = sum(1 for r in steps if isinstance(r, dict) and r.get("dg_fire"))
            ag_rates.append(ag / n)
            dg_rates.append(dg / n)
        if ag_rates:
            out["ag_rate"] = float(mean(ag_rates))
        if dg_rates:
            out["dg_rate"] = float(mean(dg_rates))

    return out


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = aggregate(args.dir, args.prefix)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"[aggregate] wrote {args.out} (seeds={result.get('seeds')})")


if __name__ == "__main__":
    main()

