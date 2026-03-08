#!/usr/bin/env python3
"""Analyze experiment results across conditions.

Aggregates summary.json files and produces per-condition comparison
including bridge/comparison breakdown and topology statistics.

Usage::

    python experiments/hotpotqa_v2/scripts/analyze_results.py \
        --results-dir experiments/hotpotqa_v2/results/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_condition(results_dir: Path) -> dict:
    """Analyze a single condition directory."""
    summary_path = results_dir / "summary.json"
    results_path = results_dir / "results.jsonl"

    info = {"name": results_dir.name}

    if summary_path.exists():
        with open(summary_path) as f:
            info["summary"] = json.load(f)

    if results_path.exists():
        b0_dist: Counter[int] = Counter()
        b1_dist: Counter[int] = Counter()
        ag_count = 0
        dg_count = 0
        total = 0

        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                total += 1
                b0_dist[r.get("delta_betti_0", 0)] += 1
                b1_dist[r.get("delta_betti_1", 0)] += 1
                if r.get("ag_fired"):
                    ag_count += 1
                if r.get("dg_fired"):
                    dg_count += 1

        info["topology"] = {
            "total": total,
            "delta_b0_distribution": dict(b0_dist),
            "delta_b1_distribution": dict(b1_dist),
            "ag_fire_rate": ag_count / max(total, 1),
            "dg_fire_rate": dg_count / max(total, 1),
        }

    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    conditions = sorted(results_dir.iterdir())

    for cond_dir in conditions:
        if not cond_dir.is_dir() or cond_dir.name.startswith("."):
            continue
        info = analyze_condition(cond_dir)
        print(f"\n{'='*60}")
        print(f"Condition: {info['name']}")

        if "summary" in info:
            s = info["summary"]
            for label, res in s.get("results", {}).items():
                print(f"  {label:>12s}: n={res['count']:>5d}  EM={res['em']:.4f}  F1={res['f1']:.4f}")

        if "topology" in info:
            t = info["topology"]
            print(f"  Topology stats (n={t['total']}):")
            print(f"    delta_b0 distribution: {t['delta_b0_distribution']}")
            print(f"    delta_b1 distribution: {t['delta_b1_distribution']}")
            print(f"    AG fire rate: {t['ag_fire_rate']:.3f}")
            print(f"    DG fire rate: {t['dg_fire_rate']:.3f}")


if __name__ == "__main__":
    main()
