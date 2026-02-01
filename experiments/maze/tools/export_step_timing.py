#!/usr/bin/env python3
"""
Export per-step timing (time_ms_eval) from steps.json to CSV, optionally plot.

Usage:
  PYTHONPATH=src MPLCONFIGDIR=results/mpl \
  python experiments/maze-query-hub-prototype/tools/export_step_timing.py \
    --steps experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s100_eval_core_steps.json \
    --out-csv experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s100_eval_core_timing.csv \
    --plot experiments/maze-query-hub-prototype/results/l3_fast/_51x51_s100_eval_core_timing.png \
    --logy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_steps(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        out: List[Dict[str, Any]] = []
        for v in data.values():
            if isinstance(v, list):
                out.extend([x for x in v if isinstance(x, dict)])
        return out
    return []


def numeric(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-step timing to CSV")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()

    steps = sorted(load_steps(args.steps), key=lambda r: int(r.get("step", 0)))
    rows: List[List[Any]] = [["step", "time_ms_eval", "g0", "gmin", "best_hop"]]
    times: List[float] = []
    for r in steps:
        step = int(r.get("step", 0))
        t = numeric(r.get("time_ms_eval", 0.0))
        times.append(t)
        rows.append([step, t, numeric(r.get("g0", 0.0)), numeric(r.get("gmin", 0.0)), int(r.get("best_hop", 0))])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    if args.plot is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [int(r[0]) for r in rows[1:]]
            ys = [float(r[1]) for r in rows[1:]]
            plt.figure(figsize=(9, 3.2))
            plt.plot(xs, ys, label="time_ms_eval")
            if args.logy:
                plt.yscale("log")
            plt.xlabel("step")
            plt.ylabel("time [ms]")
            plt.grid(True, alpha=0.3)
            plt.legend()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
        except Exception as e:
            print(f"[warn] plot skipped: {e}")

    # Print small summary
    if times:
        step0 = times[0]
        step99 = times[99] if len(times) > 99 else times[-1]
        avg = sum(times) / len(times)
        print(json.dumps({"step0_ms": step0, "step99_ms": step99, "avg_ms": avg}, indent=2))


if __name__ == "__main__":
    main()

