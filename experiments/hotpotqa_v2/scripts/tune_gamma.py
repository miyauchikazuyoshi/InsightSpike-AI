#!/usr/bin/env python3
"""Grid search for optimal gamma_0, gamma_1 values.

Runs a lightweight grid search over gamma_0 and gamma_1 using
a small sample of the dataset and picks the combination that
maximises bridge-type F1.

Usage::

    LLM_PROVIDER=mock PYTHONPATH=src python experiments/hotpotqa_v2/scripts/tune_gamma.py \
        --data experiments/hotpotqa_v2/data/hotpotqa_sample_100.jsonl \
        --limit 50
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

_EXP_ROOT = Path(__file__).parent.parent
_REPO_ROOT = _EXP_ROOT.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.hotpotqa_v2.src.data_loader import HotpotQALoader
from experiments.hotpotqa_v2.src.evaluator import HotpotQAEvaluator
from experiments.hotpotqa_v2.src.adapter import GeDIGv2Adapter


def run_trial(
    examples,
    gamma_0: float,
    gamma_1: float,
) -> dict:
    """Run a single trial with given gamma values."""
    adapter = GeDIGv2Adapter(
        structural_mode="betti_full",
        gamma_0=gamma_0,
        gamma_1=gamma_1,
    )
    evaluator = HotpotQAEvaluator()

    for example in examples:
        try:
            result = adapter.process(example)
            evaluator.evaluate_single(
                example_id=example.id,
                prediction=result.answer,
                ground_truth=example.answer,
                question_type=example.question_type,
            )
        except Exception:
            pass
        adapter.reset()

    agg = evaluator.aggregate_by_type()
    return {
        "gamma_0": gamma_0,
        "gamma_1": gamma_1,
        "all": agg.get("all", evaluator.aggregate()).to_dict(),
        "bridge": agg.get("bridge", evaluator.aggregate()).to_dict() if "bridge" in agg else None,
        "comparison": agg.get("comparison", evaluator.aggregate()).to_dict() if "comparison" in agg else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--gamma-values", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    loader = HotpotQALoader(args.data)
    examples = loader.load()[: args.limit]
    print(f"[tune] Loaded {len(examples)} examples")
    print(f"[tune] Grid: gamma values = {args.gamma_values}")

    grid = list(itertools.product(args.gamma_values, args.gamma_values))
    results = []

    for i, (g0, g1) in enumerate(grid):
        t0 = time.time()
        trial = run_trial(examples, g0, g1)
        elapsed = time.time() - t0
        results.append(trial)

        all_em = trial["all"]["em"]
        bridge_f1 = trial["bridge"]["f1"] if trial["bridge"] else 0.0
        print(
            f"  [{i+1}/{len(grid)}] g0={g0:.1f} g1={g1:.1f} "
            f"EM={all_em:.4f} bridge_F1={bridge_f1:.4f} ({elapsed:.1f}s)"
        )

    # Find best by bridge F1
    best = max(
        results,
        key=lambda r: (r["bridge"]["f1"] if r["bridge"] else 0.0),
    )
    print(f"\n{'='*60}")
    print(f"Best: gamma_0={best['gamma_0']}, gamma_1={best['gamma_1']}")
    print(f"  All EM: {best['all']['em']:.4f}")
    if best["bridge"]:
        print(f"  Bridge F1: {best['bridge']['f1']:.4f}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"grid_results": results, "best": best}, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
