#!/usr/bin/env python3
"""Win/Loss analysis between two conditions (or v1 vs v2).

Compares per-question EM across two results.jsonl files and reports
win/loss/tie counts, stratified by question_type.

Usage::

    python experiments/hotpotqa_v2/tools/win_loss_analysis.py \
        experiments/hotpotqa_v2/results/condition_a/results.jsonl \
        experiments/hotpotqa_v2/results/condition_d/results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(path: str) -> dict[str, dict]:
    """Load results indexed by example_id."""
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r["example_id"]] = r
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="Baseline results.jsonl")
    parser.add_argument("target", help="Target results.jsonl")
    args = parser.parse_args()

    base = load_results(args.baseline)
    target = load_results(args.target)

    common_ids = sorted(set(base.keys()) & set(target.keys()))

    base_name = Path(args.baseline).parent.name
    target_name = Path(args.target).parent.name

    stats: dict[str, dict[str, int]] = {}  # qt -> {win, loss, tie}

    for eid in common_ids:
        b = base[eid]
        t = target[eid]
        qt = t.get("question_type", b.get("question_type", "unknown"))

        if qt not in stats:
            stats[qt] = {"win": 0, "loss": 0, "tie": 0}
        if "all" not in stats:
            stats["all"] = {"win": 0, "loss": 0, "tie": 0}

        b_em = b.get("em", 0)
        t_em = t.get("em", 0)

        if t_em > b_em:
            stats[qt]["win"] += 1
            stats["all"]["win"] += 1
        elif t_em < b_em:
            stats[qt]["loss"] += 1
            stats["all"]["loss"] += 1
        else:
            stats[qt]["tie"] += 1
            stats["all"]["tie"] += 1

    print(f"Comparison: {target_name} vs {base_name}")
    print(f"Common examples: {len(common_ids)}")
    print()

    header = f"{'Type':<15s} {'Win':>6s} {'Loss':>6s} {'Tie':>6s} {'Net':>6s}"
    print(header)
    print("-" * len(header))
    for qt in sorted(stats.keys()):
        s = stats[qt]
        net = s["win"] - s["loss"]
        sign = "+" if net > 0 else ""
        print(f"{qt:<15s} {s['win']:>6d} {s['loss']:>6d} {s['tie']:>6d} {sign}{net:>5d}")


if __name__ == "__main__":
    main()
