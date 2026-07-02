#!/usr/bin/env python3
"""Pre-registered analysis for redesign variant #1: replay propagation + guide-off.

Committed BEFORE the experiment runs (docs/prereg/maze_sleep_ablation_v2.md).
v1's analyze_sleep_ablation.py is part of the frozen v1 prereg and is left
untouched; this is the v2 analysis with:
  - explicit seed list (warmup-success seeds from the v1 run; deterministic
    stratification, since warmup is guidance-independent),
  - arms named replay/off,
  - P2 = eval dead-end encounters (the most direct gradient-usage signal).

Usage:
    .venv/bin/python3 experiments/maze/analyze_sleep_ablation_v2.py \
        [--results-dir .../sleep_ablation_v2] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260703  # fixed, distinct from v1

# warmup-success seeds observed in the v1 ablation (deterministic property of
# the seed+config, independent of guidance and of the propagation arm)
SEEDS = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15,
         17, 18, 19, 20, 21, 24, 25, 26, 29]

ARM_A, ARM_B = "replay", "off"  # prediction: A better (fewer steps) than B


def load_pair(results_dir: Path, seed: int) -> dict | None:
    rec: dict = {"seed": seed}
    for cond in (ARM_A, ARM_B):
        path = results_dir / f"{cond}_seed{seed}.json"
        if not path.exists():
            print(f"WARNING: missing {path} — seed {seed} excluded", file=sys.stderr)
            return None
        with open(path) as f:
            d = json.load(f)
        per_seed = d.get("curriculum", {}).get("per_seed", {}).get(str(seed), {})
        ev_run = (d.get("runs") or [{}])[0]
        wu_run = (d.get("warmup_runs") or [{}])[0]
        if not per_seed:
            print(f"WARNING: no curriculum meta in {path} — seed excluded", file=sys.stderr)
            return None
        rec[cond] = {
            "eval_success": bool(per_seed.get("eval", {}).get("success", False)),
            "eval_steps": int(per_seed.get("eval", {}).get("steps", 0)),
            "eval_deadends": int(ev_run.get("dead_end_steps") or 0),
            "warmup_success": bool(per_seed.get("warmup", {}).get("success", False)),
            "warmup_steps": int(per_seed.get("warmup", {}).get("steps", 0)),
            "warmup_deadends": int(wu_run.get("dead_end_steps") or 0),
            "sleep_propagate": str(per_seed.get("sleep_propagate", "?")),
            "propagated_nodes": int(per_seed.get("inherited_propagated_nodes", -1)),
        }
    return rec


def censored(arm: dict, max_steps: int) -> int:
    return arm["eval_steps"] if arm["eval_success"] else max_steps


def boot_ci(diffs: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diffs)
    means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAP_ITERS)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_or_note(diffs: np.ndarray) -> dict:
    if np.all(diffs == 0):
        return {"stat": None, "p": None, "note": "all paired differences are zero"}
    res = stats.wilcoxon(diffs, alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "results/graph_persistent_dg/sleep_ablation_v2")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = [r for s in SEEDS if (r := load_pair(args.results_dir, s)) is not None]
    if not pairs:
        print("ERROR: no complete pairs", file=sys.stderr)
        return 1

    report: dict = {
        "prereg": "docs/prereg/maze_sleep_ablation_v2.md",
        "arms": [ARM_A, ARM_B],
        "n_pairs": len(pairs),
        "n_planned": len(SEEDS),
    }

    # ---- P3 manipulation checks ----
    a_prop_ok = all(r[ARM_A]["propagated_nodes"] > 0 and r[ARM_A]["sleep_propagate"] == ARM_A for r in pairs)
    b_prop_ok = all(r[ARM_B]["propagated_nodes"] == 0 and r[ARM_B]["sleep_propagate"] == ARM_B for r in pairs)
    wu_ok = all(r[ARM_A]["warmup_success"] and r[ARM_B]["warmup_success"] for r in pairs)
    wu_diffs = np.array([r[ARM_A]["warmup_steps"] - r[ARM_B]["warmup_steps"] for r in pairs], dtype=float)
    report["P3_manipulation_check"] = {
        "replay_all_propagated": a_prop_ok,
        "off_all_zero_propagated": b_prop_ok,
        "all_warmups_succeeded_as_stratified": wu_ok,
        "warmup_steps_paired_diff_mean": float(wu_diffs.mean()),
        "warmup_wilcoxon": wilcoxon_or_note(wu_diffs),
        "PASS": bool(a_prop_ok and b_prop_ok and wu_ok and np.all(wu_diffs == 0)),
    }

    # ---- P1 primary: eval steps (censored), all stratified pairs ----
    diffs = np.array([censored(r[ARM_A], args.max_steps) - censored(r[ARM_B], args.max_steps) for r in pairs], dtype=float)
    lo, hi = boot_ci(diffs)
    report["P1_primary_eval_steps"] = {
        "censoring": f"failure -> steps={args.max_steps}",
        "mean_replay": float(np.mean([censored(r[ARM_A], args.max_steps) for r in pairs])),
        "mean_off": float(np.mean([censored(r[ARM_B], args.max_steps) for r in pairs])),
        "paired_diff_mean_replay_minus_off": float(diffs.mean()),
        "bootstrap_ci95": [lo, hi],
        "wilcoxon": wilcoxon_or_note(diffs),
        "prediction_P1": "replay < off (diff < 0, CI95 excludes 0, p < 0.05)",
    }
    both = [r for r in pairs if r[ARM_A]["eval_success"] and r[ARM_B]["eval_success"]]
    if both:
        sdiffs = np.array([r[ARM_A]["eval_steps"] - r[ARM_B]["eval_steps"] for r in both], dtype=float)
        slo, shi = boot_ci(sdiffs)
        report["P1_sensitivity_both_success"] = {
            "n": len(both), "paired_diff_mean": float(sdiffs.mean()),
            "bootstrap_ci95": [slo, shi], "wilcoxon": wilcoxon_or_note(sdiffs),
        }
    else:
        report["P1_sensitivity_both_success"] = {"n": 0, "note": "no both-success pairs"}

    # ---- P2 secondary: eval dead-end encounters (gradient-usage signal) ----
    de_diffs = np.array([r[ARM_A]["eval_deadends"] - r[ARM_B]["eval_deadends"] for r in pairs], dtype=float)
    dlo, dhi = boot_ci(de_diffs)
    report["P2_eval_deadend_encounters"] = {
        "mean_replay": float(np.mean([r[ARM_A]["eval_deadends"] for r in pairs])),
        "mean_off": float(np.mean([r[ARM_B]["eval_deadends"] for r in pairs])),
        "paired_diff_mean": float(de_diffs.mean()),
        "bootstrap_ci95": [dlo, dhi],
        "wilcoxon": wilcoxon_or_note(de_diffs),
        "prediction_P2": "replay < off (avoids dead ends experienced in warmup)",
    }

    # ---- success rates (descriptive; floor/ceiling context for self-nav) ----
    report["success_rates"] = {
        "replay": float(np.mean([r[ARM_A]["eval_success"] for r in pairs])),
        "off": float(np.mean([r[ARM_B]["eval_success"] for r in pairs])),
        "discordant_replay_only": sum(1 for r in pairs if r[ARM_A]["eval_success"] and not r[ARM_B]["eval_success"]),
        "discordant_off_only": sum(1 for r in pairs if not r[ARM_A]["eval_success"] and r[ARM_B]["eval_success"]),
    }

    # ---- verdicts (mechanical, prereg section 3 wording) ----
    p1 = report["P1_primary_eval_steps"]
    p1_pass = (p1["paired_diff_mean_replay_minus_off"] < 0 and p1["bootstrap_ci95"][1] < 0
               and p1["wilcoxon"]["p"] is not None and p1["wilcoxon"]["p"] < 0.05)
    p2 = report["P2_eval_deadend_encounters"]
    p2_pass = (p2["paired_diff_mean"] < 0 and p2["bootstrap_ci95"][1] < 0
               and p2["wilcoxon"]["p"] is not None and p2["wilcoxon"]["p"] < 0.05)
    report["verdicts"] = {
        "P3_apparatus": "PASS" if report["P3_manipulation_check"]["PASS"] else "FAIL (results VOID)",
        "P1": "PASS" if p1_pass else "FAIL -> retry budget 1 of 2 consumed; record defeat per prereg §3",
        "P2": "PASS" if p2_pass else "FAIL (exploratory; verdict wording per prereg §3)",
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nreport written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
