#!/usr/bin/env python3
"""Pre-registered analysis for v4: deadend carving (dim8/dim9 value unification).

Committed BEFORE the experiment runs (docs/prereg/maze_sleep_v4_deadend_carving.md).
Stratify by deterministic warmup outcome; failed stratum -> P1 (eval steps,
censored) and P2 (dead-end encounters); succeeded stratum -> P4 regression
check (carving must not degrade the established replay effect). P3 verifies
the carving actually reached the Q table via the recorded q_min.

Usage:
    .venv/bin/python3 experiments/maze/analyze_sleep_v4_carving.py \
        [--results-dir .../sleep_v4_carving] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260705  # fixed, distinct from v1/v2/v3

ARM_A, ARM_B = "carving", "current"  # prediction: A better (fewer steps) in failed stratum
P4_REGRESSION_CI_LIMIT = 20.0  # prereg section 2: succeeded-stratum CI95 upper bound must be < +20 steps


def load_pair(results_dir: Path, seed: int) -> dict | None:
    rec: dict = {"seed": seed}
    for cond in (ARM_A, ARM_B):
        path = results_dir / f"{cond}_seed{seed}.json"
        if not path.exists():
            return None
        with open(path) as f:
            d = json.load(f)
        per_seed = d.get("curriculum", {}).get("per_seed", {}).get(str(seed), {})
        ev_run = (d.get("runs") or [{}])[0]
        if not per_seed:
            print(f"WARNING: no curriculum meta in {path} — seed excluded", file=sys.stderr)
            return None
        sq = per_seed.get("sleep_q") or {}
        rec[cond] = {
            "eval_success": bool(per_seed.get("eval", {}).get("success", False)),
            "eval_steps": int(per_seed.get("eval", {}).get("steps", 0)),
            "eval_deadends": int(ev_run.get("dead_end_steps") or 0),
            "warmup_success": bool(per_seed.get("warmup", {}).get("success", False)),
            "warmup_steps": int(per_seed.get("warmup", {}).get("steps", 0)),
            "sleep_propagate": str(per_seed.get("sleep_propagate", "?")),
            "propagated_nodes": int(per_seed.get("inherited_propagated_nodes", -1)),
            "q_min": float(sq.get("q_min")) if isinstance(sq, dict) and sq.get("q_min") is not None else None,
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
    if len(diffs) == 0:
        return {"stat": None, "p": None, "note": "empty stratum"}
    if np.all(diffs == 0):
        return {"stat": None, "p": None, "note": "all paired differences are zero"}
    res = stats.wilcoxon(diffs, alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def paired_block(pairs: list[dict], value_fn, label: str, prediction: str) -> dict:
    diffs = np.array([value_fn(r[ARM_A]) - value_fn(r[ARM_B]) for r in pairs], dtype=float)
    lo, hi = boot_ci(diffs)
    return {
        "n": len(pairs),
        f"mean_{ARM_A}": float(np.mean([value_fn(r[ARM_A]) for r in pairs])),
        f"mean_{ARM_B}": float(np.mean([value_fn(r[ARM_B]) for r in pairs])),
        "paired_diff_mean": float(diffs.mean()),
        "bootstrap_ci95": [lo, hi],
        "wilcoxon": wilcoxon_or_note(diffs),
        "prediction": prediction,
        "label": label,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "results/graph_persistent_dg/sleep_v4_carving")
    ap.add_argument("--seed-start", type=int, default=60)
    ap.add_argument("--seed-end", type=int, default=119)  # covers one-shot extension range
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = [r for s in range(args.seed_start, args.seed_end + 1)
             if (r := load_pair(args.results_dir, s)) is not None]
    if not pairs:
        print("ERROR: no complete pairs", file=sys.stderr)
        return 1

    strata_ok = all(r[ARM_A]["warmup_success"] == r[ARM_B]["warmup_success"] for r in pairs)
    failed = [r for r in pairs if not r[ARM_A]["warmup_success"]]
    succ = [r for r in pairs if r[ARM_A]["warmup_success"]]

    report: dict = {
        "prereg": "docs/prereg/maze_sleep_v4_deadend_carving.md",
        "arms": [ARM_A, ARM_B],
        "n_pairs_total": len(pairs),
        "n_failed_warmup": len(failed),
        "n_succeeded_warmup": len(succ),
        "failed_seeds": [r["seed"] for r in failed],
        "extension_rule": "if n_failed_warmup < 5 after seeds 60-89: run 90-119 once",
    }

    # ---- P3 manipulation checks ----
    wu_diffs = np.array([r[ARM_A]["warmup_steps"] - r[ARM_B]["warmup_steps"] for r in pairs], dtype=float)
    carve_qmin = [r[ARM_A]["q_min"] for r in pairs if r[ARM_A]["q_min"] is not None]
    curr_qmin = [r[ARM_B]["q_min"] for r in pairs if r[ARM_B]["q_min"] is not None]
    p3 = {
        "warmup_paired_diff_all_zero": bool(np.all(wu_diffs == 0)),
        "strata_agree_across_arms": strata_ok,
        "propagated_nodes_positive_both_arms": all(
            r[ARM_A]["propagated_nodes"] > 0 and r[ARM_B]["propagated_nodes"] > 0 for r in pairs),
        "carving_qmin_max": float(max(carve_qmin)) if carve_qmin else None,   # must be <= -0.9
        "current_qmin_min": float(min(curr_qmin)) if curr_qmin else None,     # must be  > -0.9
        "qmin_recorded_pairs": len(carve_qmin),
    }
    p3["PASS"] = bool(
        p3["warmup_paired_diff_all_zero"] and strata_ok and p3["propagated_nodes_positive_both_arms"]
        and carve_qmin and max(carve_qmin) <= -0.9
        and curr_qmin and min(curr_qmin) > -0.9
    )
    report["P3_manipulation_check"] = p3

    # ---- P1 primary: failed-stratum eval steps (censored) ----
    if failed:
        report["P1_failed_eval_steps"] = paired_block(
            failed, lambda a: censored(a, args.max_steps),
            "failed-stratum eval steps (censored)", "carving < current (CI95 excludes 0, p<0.05)")
        report["P2_failed_deadends"] = paired_block(
            failed, lambda a: a["eval_deadends"],
            "failed-stratum dead-end encounters", "carving < current")
        report["failed_success_rates"] = {
            ARM_A: float(np.mean([r[ARM_A]["eval_success"] for r in failed])),
            ARM_B: float(np.mean([r[ARM_B]["eval_success"] for r in failed])),
        }
    else:
        report["P1_failed_eval_steps"] = {"n": 0, "note": "no failed stratum — apply extension rule"}

    # ---- P4 regression check: succeeded stratum ----
    if succ:
        report["P4_succeeded_regression"] = paired_block(
            succ, lambda a: censored(a, args.max_steps),
            "succeeded-stratum eval steps (censored)",
            f"no degradation: CI95 upper bound < +{P4_REGRESSION_CI_LIMIT}")

    # ---- verdicts (mechanical, prereg wording) ----
    verdicts = {"P3_apparatus": "PASS" if p3["PASS"] else "FAIL (results VOID)"}
    if not failed:
        verdicts["P1"] = "NO STRATUM — apply extension rule before any verdict"
    else:
        p1 = report["P1_failed_eval_steps"]
        p1_pass = (p1["paired_diff_mean"] < 0 and p1["bootstrap_ci95"][1] < 0
                   and p1["wilcoxon"]["p"] is not None and p1["wilcoxon"]["p"] < 0.05)
        if p1_pass:
            verdicts["P1"] = "PASS"
        elif p1["paired_diff_mean"] < 0:
            verdicts["P1"] = "DIRECTION OK, UNDERPOWERED (insufficient evidence per prereg §6)"
        else:
            verdicts["P1"] = "FAIL -> record defeat: carving does not improve the failed stratum; seed-52 was seed-specific"
    if succ:
        p4 = report["P4_succeeded_regression"]
        verdicts["P4"] = ("PASS (no degradation)" if p4["bootstrap_ci95"][1] < P4_REGRESSION_CI_LIMIT
                          else "REGRESSION — full adoption withheld per prereg §3")
    report["verdicts"] = verdicts

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nreport written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
