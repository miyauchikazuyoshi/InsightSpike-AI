#!/usr/bin/env python3
"""Pre-registered analysis for v5: budget-split warmup (Wake-Sleep-Wake repetition value).

Committed BEFORE the experiment runs (docs/prereg/maze_sleep_v5_budget_split.md).
Stratify by the DETERMINISTIC warmup outcome of the cyc1 arm (single 500-step
warmup). Failed stratum -> P1 (eval steps, censored) + P2 (success rate,
discovery rate, dead-ends); succeeded stratum -> P4 regression guard (the
registered top risk: splitting may sacrifice the established succeeded-stratum
effect). P3 verifies budget compliance and cycle structure.

Usage:
    .venv/bin/python3 experiments/maze/analyze_sleep_v5_split.py \
        [--results-dir .../sleep_v5_split] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260706  # fixed, distinct from v1-v4

ARM_A, ARM_B = "cyc2", "cyc1"  # prediction: A (split) better in failed stratum
P4_REGRESSION_CI_LIMIT = 30.0  # prereg section 2: succeeded-stratum CI95 upper bound must be < +30


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
        cycles = per_seed.get("warmup_cycles") or []
        rec[cond] = {
            "eval_success": bool(per_seed.get("eval", {}).get("success", False)),
            "eval_steps": int(per_seed.get("eval", {}).get("steps", 0)),
            "eval_deadends": int(ev_run.get("dead_end_steps") or 0),
            "warmup_success": bool(per_seed.get("warmup", {}).get("success", False)),
            "wsw_cycles": int(per_seed.get("wsw_cycles", 1)),
            "warmup_cycles": cycles,
            "warmup_any_goal": bool(per_seed.get("warmup_any_goal", False)),
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
    if len(diffs) == 0:
        return {"stat": None, "p": None, "note": "empty stratum"}
    if np.all(diffs == 0):
        return {"stat": None, "p": None, "note": "all paired differences are zero"}
    res = stats.wilcoxon(diffs, alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def paired_block(pairs: list[dict], value_fn, prediction: str) -> dict:
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
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "results/graph_persistent_dg/sleep_v5_split")
    ap.add_argument("--seed-start", type=int, default=90)
    ap.add_argument("--seed-end", type=int, default=149)  # covers one-shot extension range
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = [r for s in range(args.seed_start, args.seed_end + 1)
             if (r := load_pair(args.results_dir, s)) is not None]
    if not pairs:
        print("ERROR: no complete pairs", file=sys.stderr)
        return 1

    # Stratify by the cyc1 arm's (deterministic) single-warmup outcome
    failed = [r for r in pairs if not r[ARM_B]["warmup_success"]]
    succ = [r for r in pairs if r[ARM_B]["warmup_success"]]

    report: dict = {
        "prereg": "docs/prereg/maze_sleep_v5_budget_split.md",
        "arms": [ARM_A, ARM_B],
        "n_pairs_total": len(pairs),
        "n_failed_cyc1_warmup": len(failed),
        "n_succeeded_cyc1_warmup": len(succ),
        "failed_seeds": [r["seed"] for r in failed],
        "extension_rule": "if failed stratum < 5 after seeds 90-119: run 120-149 once",
    }

    # ---- P3 manipulation checks ----
    half = args.max_steps // 2  # 250 per split cycle
    a_ok = all(
        r[ARM_A]["wsw_cycles"] == 2 and len(r[ARM_A]["warmup_cycles"]) == 2
        and all(int(c.get("steps", 0)) <= half for c in r[ARM_A]["warmup_cycles"])
        for r in pairs)
    b_ok = all(
        r[ARM_B]["wsw_cycles"] == 1 and len(r[ARM_B]["warmup_cycles"]) == 1
        and int(r[ARM_B]["warmup_cycles"][0].get("steps", 0)) <= args.max_steps
        for r in pairs)
    prop_ok = all(r[ARM_A]["propagated_nodes"] > 0 and r[ARM_B]["propagated_nodes"] > 0 for r in pairs)
    report["P3_manipulation_check"] = {
        "cyc2_structure_and_budget_ok": a_ok,
        "cyc1_structure_and_budget_ok": b_ok,
        "propagated_nodes_positive_both_arms": prop_ok,
        "PASS": bool(a_ok and b_ok and prop_ok),
    }

    # ---- P1 primary: failed-stratum eval steps (censored) ----
    if failed:
        report["P1_failed_eval_steps"] = paired_block(
            failed, lambda a: censored(a, args.max_steps),
            "cyc2 < cyc1 (CI95 excludes 0, p<0.05)")
        b = sum(1 for r in failed if r[ARM_A]["eval_success"] and not r[ARM_B]["eval_success"])
        c = sum(1 for r in failed if not r[ARM_A]["eval_success"] and r[ARM_B]["eval_success"])
        report["P2_failed_secondary"] = {
            "success_rate_cyc2": float(np.mean([r[ARM_A]["eval_success"] for r in failed])),
            "success_rate_cyc1": float(np.mean([r[ARM_B]["eval_success"] for r in failed])),
            "discordant_cyc2_only": b,
            "discordant_cyc1_only": c,
            "mcnemar_p": float(stats.binomtest(b, b + c, 0.5).pvalue) if (b + c) else None,
            "warmup_any_goal_rate_cyc2": float(np.mean([r[ARM_A]["warmup_any_goal"] for r in failed])),
            "warmup_any_goal_rate_cyc1": float(np.mean([r[ARM_B]["warmup_any_goal"] for r in failed])),
            "deadends": paired_block(failed, lambda a: a["eval_deadends"], "cyc2 < cyc1 (descriptive)"),
        }
    else:
        report["P1_failed_eval_steps"] = {"n": 0, "note": "no failed stratum — apply extension rule"}

    # ---- P4 regression guard: succeeded stratum ----
    if succ:
        p4 = paired_block(succ, lambda a: censored(a, args.max_steps),
                          f"no regression: CI95 upper bound < +{P4_REGRESSION_CI_LIMIT}")
        # descriptive: which succeeded seeds had long cyc1 warmups (the pre-registered risk profile)
        p4["cyc1_warmup_steps_over_250"] = sum(
            1 for r in succ if int(r[ARM_B]["warmup_cycles"][0].get("steps", 0)) > half)
        report["P4_succeeded_regression"] = p4

    # ---- verdicts (mechanical, prereg wording) ----
    verdicts = {"P3_apparatus": "PASS" if report["P3_manipulation_check"]["PASS"] else "FAIL (results VOID)"}
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
            verdicts["P1"] = "FAIL -> record defeat: split hypothesis rejected; exploration 5/5 was seed-group-specific"
    if succ:
        p4r = report["P4_succeeded_regression"]
        ci_lo, ci_hi = p4r["bootstrap_ci95"]
        if ci_hi < P4_REGRESSION_CI_LIMIT:
            verdicts["P4"] = "PASS (no regression)"
        elif ci_lo > 0:
            verdicts["P4"] = "REGRESSION CONFIRMED — full adoption withheld; adaptive splitting becomes the v6 candidate per prereg §3"
        else:
            verdicts["P4"] = "INCONCLUSIVE (CI spans 0 but upper bound >= limit) — report as-is, adoption decision to author"
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
