#!/usr/bin/env python3
from __future__ import annotations

"""
Export richer per-step diagnostics to CSV (and optional plot) to study
time growth and gate/hop behavior on short runs.

Columns:
  step, time_ms_eval, time_ms_candidates, g0, gmin, best_hop,
  ag_fire, dg_fire, theta_ag, possible_moves, is_dead_end,
  sp_sssp_du, sp_sssp_dv, sp_dv_leaf_skips, sp_cycle_verifies

Usage:
  .venv/bin/python experiments/maze-query-hub-prototype/tools/export_step_diagnostics.py \
    --steps experiments/maze-query-hub-prototype/results/25x25_quick/steps.json \
    --out-csv experiments/maze-query-hub-prototype/results/25x25_quick/diag.csv \
    --plot experiments/maze-query-hub-prototype/results/25x25_quick/diag.png
"""

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


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def i(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-step diagnostics")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()

    steps = sorted(load_steps(args.steps), key=lambda r: int(r.get("step", 0)))
    header = [
        "step", "time_ms_eval", "time_ms_candidates", "g0", "gmin", "best_hop",
        "ag_fire", "dg_fire", "theta_ag", "possible_moves", "is_dead_end",
        "sp_sssp_du", "sp_sssp_dv", "sp_dv_leaf_skips", "sp_cycle_verifies",
    ]
    rows: List[List[Any]] = [header]
    times: List[float] = []
    for r in steps:
        step = i(r.get("step", 0))
        t_eval = fnum(r.get("time_ms_eval", 0.0))
        t_cand = fnum(r.get("time_ms_candidates", 0.0))
        g0 = fnum(r.get("g0", 0.0))
        gmin = fnum(r.get("gmin", 0.0))
        best_hop = i(r.get("best_hop", 0))
        ag = bool(r.get("ag_fire", False))
        dg = bool(r.get("dg_fire", False))
        th_ag = fnum(r.get("theta_ag", 0.0))
        pm = r.get("possible_moves")
        pm_len = len(pm) if isinstance(pm, list) else -1
        is_dead = bool(r.get("is_dead_end", False))
        du = i(r.get("sp_sssp_du", 0))
        dv = i(r.get("sp_sssp_dv", 0))
        leaf = i(r.get("sp_dv_leaf_skips", 0))
        cyc = i(r.get("sp_cycle_verifies", 0))
        rows.append([step, t_eval, t_cand, g0, gmin, best_hop, int(ag), int(dg), th_ag, pm_len, int(is_dead), du, dv, leaf, cyc])
        times.append(t_eval)

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

    if times:
        print(json.dumps({
            "step0_ms": times[0],
            "stepN_ms": times[-1],
            "avg_ms": (sum(times) / len(times)),
            "steps": len(times),
        }, indent=2))


if __name__ == "__main__":
    main()
