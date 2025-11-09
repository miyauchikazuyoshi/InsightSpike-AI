#!/usr/bin/env python3
"""
Compare per-step metrics across three runs (A/B/C) and extract timing snapshots.

Inputs: three steps.json files produced by run_experiment_query.py

Outputs JSON to stdout with:
 - per-step: g0/gmin/best_hop for A/B/C (aligned by step index)
 - deltas: abs diffs vs A for B/C (avg_abs over common range)
 - timing: step 0 and step 99 (if present) time_ms_eval for each run
 - simple trend summary for time_ms_eval (avg, p95 if feasible)

Usage:
  PYTHONPATH=src python experiments/maze-query-hub-prototype/tools/compare_runs_abc.py \
    --a-steps path/to/A_steps.json --b-steps path/to/B_steps.json --c-steps path/to/C_steps.json \
    --a-name core --b-name cached_allpairs --c-name l3_core_fb
"""
from __future__ import annotations

import argparse
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


def numeric(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def extract_series(steps: List[Dict[str, Any]]):
    # Align by step index ascending
    steps_sorted = sorted(steps, key=lambda r: int(r.get("step", 0)))
    g0 = [numeric(r.get("g0", 0.0)) for r in steps_sorted]
    gmin = [numeric(r.get("gmin", 0.0)) for r in steps_sorted]
    best = [int(r.get("best_hop", 0)) for r in steps_sorted]
    t_eval = [numeric(r.get("time_ms_eval", 0.0)) for r in steps_sorted]
    return g0, gmin, best, t_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare A/B/C runs per-step")
    ap.add_argument("--a-steps", type=Path, required=True)
    ap.add_argument("--b-steps", type=Path, required=True)
    ap.add_argument("--c-steps", type=Path, required=True)
    ap.add_argument("--a-name", type=str, default="A")
    ap.add_argument("--b-name", type=str, default="B")
    ap.add_argument("--c-name", type=str, default="C")
    args = ap.parse_args()

    A = load_steps(args.a_steps)
    B = load_steps(args.b_steps)
    C = load_steps(args.c_steps)
    g0A, gminA, hopA, tA = extract_series(A)
    g0B, gminB, hopB, tB = extract_series(B)
    g0C, gminC, hopC, tC = extract_series(C)

    n = min(len(g0A), len(g0B), len(g0C))
    def avg_abs_diff(x, y):
        m = min(len(x), len(y), n)
        if m == 0:
            return 0.0
        return sum(abs(float(x[i]) - float(y[i])) for i in range(m)) / m

    out: Dict[str, Any] = {
        "runs": {
            args.a_name: {"steps": len(g0A)},
            args.b_name: {"steps": len(g0B)},
            args.c_name: {"steps": len(g0C)},
        },
        "per_step": {
            "count_aligned": n,
        },
        "deltas": {
            f"{args.b_name}_vs_{args.a_name}": {
                "avg_abs_g0": avg_abs_diff(g0B, g0A),
                "avg_abs_gmin": avg_abs_diff(gminB, gminA),
                "avg_abs_besthop": avg_abs_diff(hopB, hopA),
            },
            f"{args.c_name}_vs_{args.a_name}": {
                "avg_abs_g0": avg_abs_diff(g0C, g0A),
                "avg_abs_gmin": avg_abs_diff(gminC, gminA),
                "avg_abs_besthop": avg_abs_diff(hopC, hopA),
            },
        },
        "timing": {
            args.a_name: {}, args.b_name: {}, args.c_name: {}
        }
    }

    def pick_timing(ts: List[float]):
        d: Dict[str, float] = {}
        if ts:
            d["step_0_ms"] = float(ts[0])
            if len(ts) > 99:
                d["step_99_ms"] = float(ts[99])
            else:
                d["step_99_ms"] = float(ts[-1])
            d["avg_ms"] = float(sum(ts) / len(ts))
        return d

    out["timing"][args.a_name] = pick_timing(tA)
    out["timing"][args.b_name] = pick_timing(tB)
    out["timing"][args.c_name] = pick_timing(tC)

    # Optionally include a tiny preview of the first 5 per-step tuples for debugging
    preview = []
    for i in range(min(5, n)):
        preview.append({
            "step": i,
            args.a_name: {"g0": g0A[i], "gmin": gminA[i], "hop": hopA[i], "t": tA[i] if i < len(tA) else 0.0},
            args.b_name: {"g0": g0B[i], "gmin": gminB[i], "hop": hopB[i], "t": tB[i] if i < len(tB) else 0.0},
            args.c_name: {"g0": g0C[i], "gmin": gminC[i], "hop": hopC[i], "t": tC[i] if i < len(tC) else 0.0},
        })
    out["per_step"]["preview"] = preview

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

