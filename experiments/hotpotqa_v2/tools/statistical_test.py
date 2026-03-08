#!/usr/bin/env python3
"""Statistical significance tests for HotpotQA experiment comparison.

Computes McNemar's test (EM) and paired bootstrap (F1) between two methods.

Usage::

    python experiments/hotpotqa_v2/tools/statistical_test.py \
        results_a/results.jsonl results_b/results.jsonl \
        --labels "Hybrid-E1" "IRCoT"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict[str, dict]:
    """Load results.jsonl → {example_id: record}."""
    results = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            eid = rec.get("example_id", rec.get("id"))
            results[eid] = rec
    return results


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """McNemar's test for paired binary outcomes (EM).

    Null hypothesis: the two methods have equal accuracy.
    """
    assert len(correct_a) == len(correct_b)
    n = len(correct_a)

    # Contingency table
    # b01 = A wrong, B right
    # b10 = A right, B wrong
    b01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    b10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    b11 = sum(1 for a, b in zip(correct_a, correct_b) if a and b)
    b00 = sum(1 for a, b in zip(correct_a, correct_b) if not a and not b)

    # McNemar's chi-squared (with continuity correction)
    denom = b01 + b10
    if denom == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / denom
        # Approximate p-value from chi2 distribution (1 df)
        # Using scipy if available, else manual approximation
        try:
            from scipy.stats import chi2 as chi2_dist
            p_value = 1 - chi2_dist.cdf(chi2, df=1)
        except ImportError:
            # Rough approximation for common thresholds
            import math
            p_value = math.erfc(math.sqrt(chi2 / 2))

    return {
        "test": "McNemar",
        "n": n,
        "a_right_b_wrong": b10,
        "a_wrong_b_right": b01,
        "both_right": b11,
        "both_wrong": b00,
        "chi2": chi2,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test for continuous scores (F1).

    Returns confidence interval and p-value for the difference A - B.
    """
    import random

    assert len(scores_a) == len(scores_b)
    n = len(scores_a)

    rng = random.Random(seed)

    observed_diff = sum(a - b for a, b in zip(scores_a, scores_b)) / n

    # Bootstrap
    diffs = []
    count_extreme = 0
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        diff = sum(scores_a[i] - scores_b[i] for i in indices) / n
        diffs.append(diff)
        if observed_diff >= 0 and diff <= 0:
            count_extreme += 1
        elif observed_diff < 0 and diff >= 0:
            count_extreme += 1

    diffs.sort()
    ci_lower = diffs[int(n_bootstrap * 0.025)]
    ci_upper = diffs[int(n_bootstrap * 0.975)]
    p_value = count_extreme / n_bootstrap

    return {
        "test": "paired_bootstrap",
        "n": n,
        "n_bootstrap": n_bootstrap,
        "observed_diff": observed_diff,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical significance test")
    parser.add_argument("results_a", type=str, help="Path to results_a.jsonl")
    parser.add_argument("results_b", type=str, help="Path to results_b.jsonl")
    parser.add_argument("--labels", nargs=2, default=["A", "B"],
                        help="Labels for the two methods")
    parser.add_argument("--bootstrap-n", type=int, default=10000,
                        help="Number of bootstrap samples")
    args = parser.parse_args()

    results_a = load_results(Path(args.results_a))
    results_b = load_results(Path(args.results_b))

    # Find common examples
    common_ids = sorted(set(results_a.keys()) & set(results_b.keys()))
    if not common_ids:
        print("ERROR: No common example IDs found!")
        sys.exit(1)

    print(f"Common examples: {len(common_ids)}")
    print(f"  {args.labels[0]}: {len(results_a)} total")
    print(f"  {args.labels[1]}: {len(results_b)} total")
    print()

    # Extract paired scores
    em_a = [bool(results_a[eid].get("em", 0)) for eid in common_ids]
    em_b = [bool(results_b[eid].get("em", 0)) for eid in common_ids]
    f1_a = [float(results_a[eid].get("f1", 0)) for eid in common_ids]
    f1_b = [float(results_b[eid].get("f1", 0)) for eid in common_ids]

    # Summary statistics
    em_mean_a = sum(em_a) / len(em_a)
    em_mean_b = sum(em_b) / len(em_b)
    f1_mean_a = sum(f1_a) / len(f1_a)
    f1_mean_b = sum(f1_b) / len(f1_b)

    print(f"{'Metric':<10} {args.labels[0]:>15} {args.labels[1]:>15} {'Diff':>10}")
    print("-" * 55)
    print(f"{'EM':<10} {em_mean_a:>15.1%} {em_mean_b:>15.1%} {em_mean_a - em_mean_b:>+10.1%}")
    print(f"{'F1':<10} {f1_mean_a:>15.4f} {f1_mean_b:>15.4f} {f1_mean_a - f1_mean_b:>+10.4f}")
    print()

    # McNemar's test on EM
    print("=" * 55)
    print("McNemar's Test (EM — Exact Match)")
    print("=" * 55)
    mcn = mcnemar_test(em_a, em_b)
    print(f"  {args.labels[0]} right, {args.labels[1]} wrong: {mcn['a_right_b_wrong']}")
    print(f"  {args.labels[0]} wrong, {args.labels[1]} right: {mcn['a_wrong_b_right']}")
    print(f"  Both right:                         {mcn['both_right']}")
    print(f"  Both wrong:                         {mcn['both_wrong']}")
    print(f"  Chi-squared:  {mcn['chi2']:.4f}")
    print(f"  p-value:      {mcn['p_value']:.6f}")
    print(f"  Significant (p<0.05): {'YES ✓' if mcn['significant_005'] else 'NO'}")
    print(f"  Significant (p<0.01): {'YES ✓' if mcn['significant_001'] else 'NO'}")
    print()

    # Paired bootstrap on F1
    print("=" * 55)
    print("Paired Bootstrap Test (F1)")
    print("=" * 55)
    boot = paired_bootstrap(f1_a, f1_b, n_bootstrap=args.bootstrap_n)
    print(f"  Observed diff ({args.labels[0]} - {args.labels[1]}): {boot['observed_diff']:+.4f}")
    print(f"  95% CI: [{boot['ci_95_lower']:+.4f}, {boot['ci_95_upper']:+.4f}]")
    print(f"  p-value: {boot['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'YES ✓' if boot['significant_005'] else 'NO'}")
    print()

    # Save results
    output = {
        "labels": args.labels,
        "n_common": len(common_ids),
        "summary": {
            args.labels[0]: {"em": em_mean_a, "f1": f1_mean_a},
            args.labels[1]: {"em": em_mean_b, "f1": f1_mean_b},
        },
        "mcnemar": mcn,
        "bootstrap_f1": boot,
    }
    # Convert numpy types to Python native for JSON serialization
    def _to_native(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        return obj

    out_path = Path(args.results_a).parent / "significance_test.json"
    with open(out_path, "w") as f:
        json.dump(_to_native(output), f, indent=2)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
