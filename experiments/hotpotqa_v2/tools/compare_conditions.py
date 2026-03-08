#!/usr/bin/env python3
"""Compare results across experimental conditions.

Reads summary.json from each condition directory and produces a
comparison table.

Usage::

    python experiments/hotpotqa_v2/tools/compare_conditions.py \
        experiments/hotpotqa_v2/results/condition_*/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare condition results")
    parser.add_argument("summaries", nargs="+", help="Paths to summary.json files")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.summaries):
        with open(path) as f:
            s = json.load(f)
        name = s.get("condition", Path(path).parent.name)
        mode = s.get("structural_mode", "?")
        g0 = s.get("gamma_0", "?")
        g1 = s.get("gamma_1", "?")
        results = s.get("results", {})
        all_r = results.get("all", {})
        bridge_r = results.get("bridge", {})
        comp_r = results.get("comparison", {})
        rows.append({
            "condition": name,
            "mode": mode,
            "g0": g0,
            "g1": g1,
            "n": all_r.get("count", 0),
            "em": all_r.get("em", 0),
            "f1": all_r.get("f1", 0),
            "sf_f1": all_r.get("sf_f1", 0),
            "bridge_em": bridge_r.get("em", 0),
            "bridge_f1": bridge_r.get("f1", 0),
            "comp_em": comp_r.get("em", 0),
            "comp_f1": comp_r.get("f1", 0),
        })

    # Print table
    header = (
        f"{'Condition':<25s} {'Mode':<12s} {'g0':>4s} {'g1':>4s} "
        f"{'N':>5s} {'EM':>7s} {'F1':>7s} {'SF-F1':>7s} "
        f"{'Br-EM':>7s} {'Br-F1':>7s} {'Co-EM':>7s} {'Co-F1':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['condition']:<25s} {r['mode']:<12s} {r['g0']:>4} {r['g1']:>4} "
            f"{r['n']:>5d} {r['em']:>7.4f} {r['f1']:>7.4f} {r['sf_f1']:>7.4f} "
            f"{r['bridge_em']:>7.4f} {r['bridge_f1']:>7.4f} "
            f"{r['comp_em']:>7.4f} {r['comp_f1']:>7.4f}"
        )


if __name__ == "__main__":
    main()
