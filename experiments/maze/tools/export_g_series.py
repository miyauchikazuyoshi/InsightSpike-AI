#!/usr/bin/env python3
"""
Export per-step g0/gmin series for a given seed from steps.json.

Usage:
  PYTHONPATH=src MPLCONFIGDIR=results/mpl \
  python experiments/maze-query-hub-prototype/tools/export_g_series.py \
    --steps experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_steps.json \
    --seed 31 \
    --out-csv experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_seed31_g.csv \
    --plot experiments/maze-query-hub-prototype/results/paper_grid/_25x25_s250_l3_seed31_g.png
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Export g0/gmin per-step series")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    steps = [r for r in load_steps(args.steps) if int(r.get("seed", -1)) == int(args.seed)]
    steps.sort(key=lambda r: int(r.get("step", 0)))
    rows: List[List[Any]] = [["step", "g0", "gmin", "is_dead_end", "ag_fire", "dg_fire"]]
    for r in steps:
        rows.append([
            int(r.get("step", 0)),
            float(r.get("g0", 0.0)),
            float(r.get("gmin", r.get("g0", 0.0))),
            bool(r.get("is_dead_end", False)),
            bool(r.get("ag_fire", False)),
            bool(r.get("dg_fire", False)),
        ])
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
            g0 = [float(r[1]) for r in rows[1:]]
            gmin = [float(r[2]) for r in rows[1:]]
            plt.figure(figsize=(9, 3.2))
            plt.plot(xs, g0, label="g0", color="#1c7ed6")
            plt.plot(xs, gmin, label="gmin", color="#f03e3e")
            plt.xlabel("step")
            plt.ylabel("g")
            plt.grid(True, alpha=0.3)
            plt.legend()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
        except Exception as e:
            print(f"[warn] plot skipped: {e}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()

