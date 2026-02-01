#!/usr/bin/env python3
"""
Quick comparison of evaluator vs L3 hop0 metrics from two steps.json files.

Usage:
  python experiments/maze-query-hub-prototype/tools/compare_eval_l3.py \
    --eval-steps path/to/eval_steps.json \
    --l3-steps path/to/l3_steps.json

Outputs summary stats of |g0 diff| and DG agreement rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_steps(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Some logs use { seed: [steps...] } shape; flatten
        all_steps: List[Dict[str, Any]] = []
        for _, steps in data.items():
            if isinstance(steps, list):
                all_steps.extend(steps)
        return all_steps
    elif isinstance(data, list):
        return data
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare evaluator vs L3 hop0 metrics")
    ap.add_argument("--eval-steps", type=Path, required=True)
    ap.add_argument("--l3-steps", type=Path, required=True)
    args = ap.parse_args()

    eval_steps = _load_steps(args.eval_steps)
    l3_steps = _load_steps(args.l3_steps)

    n = min(len(eval_steps), len(l3_steps))
    if n == 0:
        print(json.dumps({"count": 0, "avg_abs_g0_diff": 0.0, "dg_agreement": 0.0}, indent=2))
        return

    g0_diffs: List[float] = []
    dg_agree = 0
    for i in range(n):
        e = eval_steps[i]
        l = l3_steps[i]
        try:
            g0e = float(e.get("g0", 0.0))
        except Exception:
            g0e = 0.0
        try:
            g0l = float(l.get("g0", 0.0))
        except Exception:
            g0l = 0.0
        g0_diffs.append(abs(g0e - g0l))
        if bool(e.get("dg_fire", False)) == bool(l.get("dg_fire", False)):
            dg_agree += 1

    avg_abs_g0 = (sum(g0_diffs) / len(g0_diffs)) if g0_diffs else 0.0
    result = {
        "count": n,
        "avg_abs_g0_diff": avg_abs_g0,
        "dg_agreement": (dg_agree / n) if n > 0 else 0.0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

