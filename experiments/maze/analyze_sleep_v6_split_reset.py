#!/usr/bin/env python3
"""Pre-registered analysis for v6: budget-split warmup + episode-boundary reset.

Committed BEFORE the experiment runs (docs/prereg/maze_sleep_v6_split_reset.md).

v5 established that unconditional splitting regresses the succeeded stratum
(+122.5, p=1.8e-05); exploration traced the sub-250 share of that regression
to cross-episode revisit contamination in the concatenated Q rebuild and
eliminated it 5/5 with --sleep-q-episode-reset. v6 re-runs the duel on fresh
seeds with the fix in place.

P1 (PRIMARY, non-inferiority): succeeded stratum (cyc1-warmup success) —
    paired diff (cyc2reset - cyc1) CI95 upper bound < +30 steps.
P2 (secondary, descriptive): failed stratum — direction, rescues, discovery,
    dead-ends (n expected ~6; v5 showed this stratum is underpowered at n=6,
    so it is registered as descriptive, not a hypothesis test).
P3 (manipulation): structure/budget per arm, replay active both arms,
    episode_boundaries_applied non-empty in the cyc2reset arm.
P4 (secondary, superiority probe): all-pairs two-sided test — does repetition
    with the fix ever BEAT the single warmup overall (seed-104-type gains vs
    over-250 discovery-loss cost)?

Usage:
    .venv/bin/python3 experiments/maze/analyze_sleep_v6_split_reset.py \
        [--results-dir .../sleep_v6_split_reset] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260707  # fixed, distinct from v1-v5

ARM_A, ARM_B = "cyc2reset", "cyc1"
P1_NONINFERIORITY_CI_LIMIT = 30.0  # prereg section 2: succeeded-stratum CI95 upper < +30


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
            "episode_boundaries": list((per_seed.get("sleep_q") or {}).get(
                "episode_boundaries_applied") or []),
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
        "identical_pairs": int(np.sum(diffs == 0)),
        "arm_a_better": int(np.sum(diffs < 0)),
        "arm_b_better": int(np.sum(diffs > 0)),
        "prediction": prediction,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "results/graph_persistent_dg/sleep_v6_split_reset")
    ap.add_argument("--seed-start", type=int, default=120)
    ap.add_argument("--seed-end", type=int, default=149)
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
        "prereg": "docs/prereg/maze_sleep_v6_split_reset.md",
        "arms": [ARM_A, ARM_B],
        "n_pairs_total": len(pairs),
        "n_failed_cyc1_warmup": len(failed),
        "n_succeeded_cyc1_warmup": len(succ),
        "failed_seeds": [r["seed"] for r in failed],
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
    # The reset must actually have engaged in the split arm: exactly one
    # boundary, equal to warmup-cycle-1's recorded step count.
    reset_ok = all(
        len(r[ARM_A]["episode_boundaries"]) == 1
        and int(r[ARM_A]["episode_boundaries"][0]) == int(r[ARM_A]["warmup_cycles"][0].get("steps", -1))
        for r in pairs)
    report["P3_manipulation_check"] = {
        "cyc2reset_structure_and_budget_ok": a_ok,
        "cyc1_structure_and_budget_ok": b_ok,
        "propagated_nodes_positive_both_arms": prop_ok,
        "episode_reset_engaged_split_arm": reset_ok,
        "PASS": bool(a_ok and b_ok and prop_ok and reset_ok),
    }

    # ---- P1 PRIMARY: succeeded-stratum non-inferiority ----
    if succ:
        p1 = paired_block(succ, lambda a: censored(a, args.max_steps),
                          f"non-inferior: CI95 upper bound < +{P1_NONINFERIORITY_CI_LIMIT}")
        p1["cyc1_warmup_steps_over_250"] = sum(
            1 for r in succ if int(r[ARM_B]["warmup_cycles"][0].get("steps", 0)) > half)
        p1["split_arm_lost_discovery"] = sum(
            1 for r in succ if not r[ARM_A]["warmup_any_goal"])
        report["P1_succeeded_noninferiority"] = p1
    else:
        report["P1_succeeded_noninferiority"] = {"n": 0, "note": "no succeeded stratum (unexpected)"}

    # ---- P2 secondary (descriptive): failed stratum ----
    if failed:
        report["P2_failed_descriptive"] = paired_block(
            failed, lambda a: censored(a, args.max_steps),
            "cyc2reset < cyc1 (descriptive; registered as underpowered at expected n)")
        b = sum(1 for r in failed if r[ARM_A]["eval_success"] and not r[ARM_B]["eval_success"])
        c = sum(1 for r in failed if not r[ARM_A]["eval_success"] and r[ARM_B]["eval_success"])
        report["P2_failed_descriptive"].update({
            "success_rate_cyc2reset": float(np.mean([r[ARM_A]["eval_success"] for r in failed])),
            "success_rate_cyc1": float(np.mean([r[ARM_B]["eval_success"] for r in failed])),
            "discordant_cyc2reset_only": b,
            "discordant_cyc1_only": c,
            "mcnemar_p": float(stats.binomtest(b, b + c, 0.5).pvalue) if (b + c) else None,
            "warmup_any_goal_rate_cyc2reset": float(np.mean([r[ARM_A]["warmup_any_goal"] for r in failed])),
            "deadends": paired_block(failed, lambda a: a["eval_deadends"],
                                     "cyc2reset < cyc1 (descriptive)"),
        })
    else:
        report["P2_failed_descriptive"] = {"n": 0, "note": "no failed stratum in 120-149"}

    # ---- P4 secondary: all-pairs superiority probe ----
    report["P4_allpairs_probe"] = paired_block(
        pairs, lambda a: censored(a, args.max_steps),
        "two-sided probe: does split+reset beat single warmup overall?")

    # ---- verdicts (mechanical, prereg wording) ----
    verdicts = {"P3_apparatus": "PASS" if report["P3_manipulation_check"]["PASS"] else "FAIL (results VOID)"}
    if succ:
        ci_lo, ci_hi = report["P1_succeeded_noninferiority"]["bootstrap_ci95"]
        if ci_hi < P1_NONINFERIORITY_CI_LIMIT:
            verdicts["P1"] = "PASS (non-inferior) — split is no longer harmful with the fix"
        elif ci_lo > 0:
            verdicts["P1"] = ("REGRESSION PERSISTS — contamination was not the main cause on fresh seeds; "
                              "record defeat of the contamination attribution, unconditional split stays rejected")
        else:
            verdicts["P1"] = "INCONCLUSIVE (CI spans 0 but upper bound >= limit) — record as-is"
    p4 = report["P4_allpairs_probe"]
    p4_p = p4["wilcoxon"]["p"]
    if p4["bootstrap_ci95"][1] < 0 and p4_p is not None and p4_p < 0.05:
        verdicts["P4"] = "SUPERIOR — split+reset beats single warmup overall (candidate new default, author decision)"
    elif p4["bootstrap_ci95"][0] > 0 and p4_p is not None and p4_p < 0.05:
        verdicts["P4"] = "INFERIOR overall — split remains costlier than single even with the fix"
    else:
        verdicts["P4"] = "NEUTRAL (no overall difference established)"
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
