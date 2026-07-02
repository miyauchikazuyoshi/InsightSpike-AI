#!/usr/bin/env python3
"""Pre-registered analysis for v3: lift of failed warmups.

Committed BEFORE the experiment runs (docs/prereg/maze_sleep_ablation_v3.md).
All executed seeds are loaded and stratified by warmup outcome (deterministic,
guidance/propagation-independent). Failed-warmup stratum -> P1 (McNemar exact
on eval success) and P2 (dead-end encounters); succeeded stratum -> P4
(replication of v2's step effect on fresh seeds).

Usage:
    .venv/bin/python3 experiments/maze/analyze_sleep_ablation_v3.py \
        [--results-dir .../sleep_ablation_v3] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260704  # fixed, distinct from v1/v2

ARM_A, ARM_B = "replay", "off"


def load_pair(results_dir: Path, seed: int) -> dict | None:
    rec: dict = {"seed": seed}
    for cond in (ARM_A, ARM_B):
        path = results_dir / f"{cond}_seed{seed}.json"
        if not path.exists():
            return None  # not-yet-run seeds are silently skipped (discovery mode)
        with open(path) as f:
            d = json.load(f)
        per_seed = d.get("curriculum", {}).get("per_seed", {}).get(str(seed), {})
        ev_run = (d.get("runs") or [{}])[0]
        if not per_seed:
            print(f"WARNING: no curriculum meta in {path} — seed excluded", file=sys.stderr)
            return None
        rec[cond] = {
            "eval_success": bool(per_seed.get("eval", {}).get("success", False)),
            "eval_steps": int(per_seed.get("eval", {}).get("steps", 0)),
            "eval_deadends": int(ev_run.get("dead_end_steps") or 0),
            "warmup_success": bool(per_seed.get("warmup", {}).get("success", False)),
            "warmup_steps": int(per_seed.get("warmup", {}).get("steps", 0)),
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
    if len(diffs) == 0:
        return {"stat": None, "p": None, "note": "empty stratum"}
    if np.all(diffs == 0):
        return {"stat": None, "p": None, "note": "all paired differences are zero"}
    res = stats.wilcoxon(diffs, alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "results/graph_persistent_dg/sleep_ablation_v3")
    ap.add_argument("--seed-start", type=int, default=30)
    ap.add_argument("--seed-end", type=int, default=89)  # covers the one-shot extension range
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pairs = [r for s in range(args.seed_start, args.seed_end + 1)
             if (r := load_pair(args.results_dir, s)) is not None]
    if not pairs:
        print("ERROR: no complete pairs", file=sys.stderr)
        return 1

    # Deterministic stratification by warmup outcome (must agree across arms — P3)
    strata_ok = all(r[ARM_A]["warmup_success"] == r[ARM_B]["warmup_success"] for r in pairs)
    failed = [r for r in pairs if not r[ARM_A]["warmup_success"]]
    succ = [r for r in pairs if r[ARM_A]["warmup_success"]]

    report: dict = {
        "prereg": "docs/prereg/maze_sleep_ablation_v3.md",
        "arms": [ARM_A, ARM_B],
        "n_pairs_total": len(pairs),
        "n_failed_warmup": len(failed),
        "n_succeeded_warmup": len(succ),
        "extension_rule": "if n_failed_warmup < 5 after seeds 30-59: run 60-89 once (prereg section 4)",
    }

    # ---- P3 manipulation checks ----
    a_ok = all(r[ARM_A]["propagated_nodes"] > 0 and r[ARM_A]["sleep_propagate"] == ARM_A for r in pairs)
    b_ok = all(r[ARM_B]["propagated_nodes"] == 0 and r[ARM_B]["sleep_propagate"] == ARM_B for r in pairs)
    wu_diffs = np.array([r[ARM_A]["warmup_steps"] - r[ARM_B]["warmup_steps"] for r in pairs], dtype=float)
    report["P3_manipulation_check"] = {
        "replay_all_propagated": a_ok,
        "off_all_zero_propagated": b_ok,
        "strata_agree_across_arms": strata_ok,
        "warmup_steps_paired_diff_mean": float(wu_diffs.mean()),
        "PASS": bool(a_ok and b_ok and strata_ok and np.all(wu_diffs == 0)),
    }

    # ---- P1 primary: eval success in the failed-warmup stratum (McNemar exact) ----
    if failed:
        b = sum(1 for r in failed if r[ARM_A]["eval_success"] and not r[ARM_B]["eval_success"])
        c = sum(1 for r in failed if not r[ARM_A]["eval_success"] and r[ARM_B]["eval_success"])
        p1 = {
            "n": len(failed),
            "seeds": [r["seed"] for r in failed],
            "success_replay": sum(r[ARM_A]["eval_success"] for r in failed),
            "success_off": sum(r[ARM_B]["eval_success"] for r in failed),
            "discordant_replay_only": b,
            "discordant_off_only": c,
            "mcnemar_p": float(stats.binomtest(b, b + c, 0.5).pvalue) if (b + c) else None,
            "prediction_P1": "replay > off (success rate in failed-warmup stratum)",
        }
    else:
        p1 = {"n": 0, "note": "no failed-warmup seeds — apply extension rule"}
    report["P1_lift_success_rate"] = p1

    # ---- P2: dead-end encounters in the failed-warmup stratum ----
    if failed:
        de = np.array([r[ARM_A]["eval_deadends"] - r[ARM_B]["eval_deadends"] for r in failed], dtype=float)
        lo, hi = boot_ci(de)
        report["P2_deadends_failed_stratum"] = {
            "mean_replay": float(np.mean([r[ARM_A]["eval_deadends"] for r in failed])),
            "mean_off": float(np.mean([r[ARM_B]["eval_deadends"] for r in failed])),
            "paired_diff_mean": float(de.mean()),
            "bootstrap_ci95": [lo, hi],
            "wilcoxon": wilcoxon_or_note(de),
            "prediction_P2": "replay < off",
        }

    # ---- P4: v2 replication on fresh succeeded-warmup seeds ----
    if succ:
        sd = np.array([censored(r[ARM_A], args.max_steps) - censored(r[ARM_B], args.max_steps) for r in succ], dtype=float)
        lo, hi = boot_ci(sd)
        report["P4_v2_replication_succeeded_stratum"] = {
            "n": len(succ),
            "mean_steps_replay": float(np.mean([censored(r[ARM_A], args.max_steps) for r in succ])),
            "mean_steps_off": float(np.mean([censored(r[ARM_B], args.max_steps) for r in succ])),
            "paired_diff_mean": float(sd.mean()),
            "bootstrap_ci95": [lo, hi],
            "wilcoxon": wilcoxon_or_note(sd),
            "prediction_P4": "replay < off (direction of v2's P1 on fresh seeds)",
        }

    # ---- verdicts (mechanical) ----
    p1_pass = bool(failed) and p1.get("mcnemar_p") is not None and p1["mcnemar_p"] < 0.05 \
        and p1["success_replay"] > p1["success_off"]
    verdicts = {"P3_apparatus": "PASS" if report["P3_manipulation_check"]["PASS"] else "FAIL (results VOID)"}
    if not failed:
        verdicts["P1"] = "NO STRATUM — apply extension rule before any verdict"
    elif p1_pass:
        verdicts["P1"] = "PASS"
    elif p1.get("mcnemar_p") is None:
        verdicts["P1"] = "NO DISCORDANT PAIRS — insufficient evidence (distinct from FAIL per prereg §6)"
    else:
        # direction + significance both required for PASS; distinguish underpowered from refuted
        direction_ok = p1["success_replay"] > p1["success_off"]
        verdicts["P1"] = ("DIRECTION OK, UNDERPOWERED (insufficient evidence per prereg §6)"
                          if direction_ok else
                          "FAIL -> record defeat: seed-4 lift was seed-specific; claim withdrawn per prereg §3")
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
