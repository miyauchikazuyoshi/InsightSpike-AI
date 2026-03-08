#!/usr/bin/env python3
"""Plot delta_beta_0 distribution from experiment results.

Reads results.jsonl and produces a histogram of delta_beta_0 values,
split by question_type (bridge vs comparison).

Usage::

    python experiments/hotpotqa_v2/tools/plot_betti_distribution.py \
        experiments/hotpotqa_v2/results/condition_d/results.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Path to results.jsonl")
    parser.add_argument("--output", default=None, help="Output PNG (optional, requires matplotlib)")
    args = parser.parse_args()

    bridge_db0: list[int] = []
    comp_db0: list[int] = []

    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            db0 = r.get("delta_betti_0", 0)
            qt = r.get("question_type", "")
            if qt == "bridge":
                bridge_db0.append(db0)
            elif qt == "comparison":
                comp_db0.append(db0)

    print(f"Bridge questions: {len(bridge_db0)}")
    print(f"  delta_beta_0 distribution: {dict(Counter(bridge_db0))}")
    print(f"Comparison questions: {len(comp_db0)}")
    print(f"  delta_beta_0 distribution: {dict(Counter(comp_db0))}")

    if args.output:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(bridge_db0, bins=range(min(bridge_db0 or [0]) - 1, max(bridge_db0 or [0]) + 2), edgecolor="black")
            axes[0].set_title("Bridge: delta_beta_0")
            axes[0].set_xlabel("delta_beta_0")
            axes[0].set_ylabel("Count")

            axes[1].hist(comp_db0, bins=range(min(comp_db0 or [0]) - 1, max(comp_db0 or [0]) + 2), edgecolor="black", color="orange")
            axes[1].set_title("Comparison: delta_beta_0")
            axes[1].set_xlabel("delta_beta_0")

            plt.tight_layout()
            plt.savefig(args.output, dpi=150)
            print(f"Saved plot to {args.output}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
