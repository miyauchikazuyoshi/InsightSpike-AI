#!/usr/bin/env python3
"""Pre-registered analysis for the maze sleep-only ablation.

Committed BEFORE the main experiment runs, as part of the preregistration
(docs/prereg/maze_sleep_ablation.md). The statistical decisions below are
fixed by the prereg; any deviation must be reported as a deviation.

Pre-registered decisions implemented here (prereg section 4):
  P1 (Primary)   Wake2 (eval) steps, sleep-on vs sleep-off, paired by seed.
                 Censoring: failed seeds count as steps = max_steps (500).
                 Test: two-sided Wilcoxon signed-rank + paired bootstrap CI95.
                 Sensitivity: both-success pairs only (reported alongside).
  P2 (Secondary) Wake2 success rate, McNemar exact test (binomial on
                 discordant pairs).
  P3 (Manipulation check) inherited_propagated_nodes > 0 for every ON run
                 and == 0 for every OFF run; warmup metrics must show no
                 group difference (sleep happens after warmup).

Usage:
    .venv/bin/python3 experiments/maze/analyze_sleep_ablation.py \
        [--results-dir experiments/maze/results/graph_persistent_dg/sleep_ablation] \
        [--seed-start 0] [--seed-end 29] [--max-steps 500] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260702  # fixed for reproducibility


def load_pair(results_dir: Path, seed: int) -> dict | None:
    """Load one on/off pair; return None (with a warning) if either file is missing."""
    rec: dict = {"seed": seed}
    for cond in ("on", "off"):
        path = results_dir / f"{cond}_seed{seed}.json"
        if not path.exists():
            print(f"WARNING: missing {path} — seed {seed} excluded", file=sys.stderr)
            return None
        with open(path) as f:
            d = json.load(f)
        per_seed = d.get("curriculum", {}).get("per_seed", {}).get(str(seed), {})
        if not per_seed:
            print(f"WARNING: no curriculum.per_seed['{seed}'] in {path} — seed excluded", file=sys.stderr)
            return None
        rec[cond] = {
            "eval_success": bool(per_seed.get("eval", {}).get("success", False)),
            "eval_steps": int(per_seed.get("eval", {}).get("steps", 0)),
            "warmup_success": bool(per_seed.get("warmup", {}).get("success", False)),
            "warmup_steps": int(per_seed.get("warmup", {}).get("steps", 0)),
            "sleep_propagate": str(per_seed.get("sleep_propagate", "?")),
            "propagated_nodes": int(per_seed.get("inherited_propagated_nodes", -1)),
            "inherited_nodes": int(per_seed.get("inherited_graph_nodes", -1)),
            "inherited_edges": int(per_seed.get("inherited_graph_edges", -1)),
            "avg_edges": float(d.get("summary", {}).get("avg_edges", float("nan"))),
        }
    return rec


def censored_steps(arm: dict, max_steps: int) -> int:
    """Pre-registered censoring: failure counts as the step cap."""
    return arm["eval_steps"] if arm["eval_success"] else max_steps


def paired_bootstrap_ci(diffs: np.ndarray, iters: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(iters)
    for i in range(iters):
        means[i] = diffs[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_or_note(diffs: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank; degenerate all-zero case reported as such."""
    if np.all(diffs == 0):
        return {"stat": None, "p": None, "note": "all paired differences are zero"}
    res = stats.wilcoxon(diffs, alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def mcnemar_exact(pairs: list[dict]) -> dict:
    """Exact McNemar: binomial test on discordant pairs."""
    b = sum(1 for r in pairs if r["on"]["eval_success"] and not r["off"]["eval_success"])
    c = sum(1 for r in pairs if not r["on"]["eval_success"] and r["off"]["eval_success"])
    out = {"on_only_success": b, "off_only_success": c}
    if b + c == 0:
        out.update({"p": None, "note": "no discordant pairs"})
    else:
        out["p"] = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "results/graph_persistent_dg/sleep_ablation")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-end", type=int, default=29)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    args = ap.parse_args()

    pairs = []
    for seed in range(args.seed_start, args.seed_end + 1):
        rec = load_pair(args.results_dir, seed)
        if rec is not None:
            pairs.append(rec)

    n_planned = args.seed_end - args.seed_start + 1
    if not pairs:
        print("ERROR: no complete pairs found", file=sys.stderr)
        return 1

    report: dict = {
        "prereg": "docs/prereg/maze_sleep_ablation.md",
        "n_pairs": len(pairs),
        "n_planned": n_planned,
    }

    # ---- P3 manipulation checks (report first: if these fail, results are void) ----
    p3_on_ok = all(r["on"]["propagated_nodes"] > 0 and r["on"]["sleep_propagate"] == "on" for r in pairs)
    p3_off_ok = all(r["off"]["propagated_nodes"] == 0 and r["off"]["sleep_propagate"] == "off" for r in pairs)
    warmup_diffs = np.array(
        [(r["on"]["warmup_steps"] if r["on"]["warmup_success"] else args.max_steps)
         - (r["off"]["warmup_steps"] if r["off"]["warmup_success"] else args.max_steps)
         for r in pairs],
        dtype=float,
    )
    report["P3_manipulation_check"] = {
        "on_all_propagated": p3_on_ok,
        "off_all_zero_propagated": p3_off_ok,
        "warmup_steps_paired_diff_mean": float(warmup_diffs.mean()),
        "warmup_wilcoxon": wilcoxon_or_note(warmup_diffs),
        "note": "warmup precedes sleep; any warmup group difference indicates a broken apparatus",
        "PASS": bool(p3_on_ok and p3_off_ok),
    }

    # ---- P1 primary: eval steps, censored, all pairs ----
    diffs = np.array(
        [censored_steps(r["on"], args.max_steps) - censored_steps(r["off"], args.max_steps) for r in pairs],
        dtype=float,
    )
    lo, hi = paired_bootstrap_ci(diffs, BOOTSTRAP_ITERS, BOOTSTRAP_SEED)
    report["P1_primary_eval_steps"] = {
        "censoring": f"failure -> steps={args.max_steps}",
        "mean_on": float(np.mean([censored_steps(r["on"], args.max_steps) for r in pairs])),
        "mean_off": float(np.mean([censored_steps(r["off"], args.max_steps) for r in pairs])),
        "paired_diff_mean_on_minus_off": float(diffs.mean()),
        "bootstrap_ci95": [lo, hi],
        "wilcoxon": wilcoxon_or_note(diffs),
        "prediction_P1": "on < off (diff < 0, CI95 excludes 0, p < 0.05)",
    }

    # Sensitivity: both-success pairs only
    both = [r for r in pairs if r["on"]["eval_success"] and r["off"]["eval_success"]]
    if both:
        sdiffs = np.array([r["on"]["eval_steps"] - r["off"]["eval_steps"] for r in both], dtype=float)
        slo, shi = paired_bootstrap_ci(sdiffs, BOOTSTRAP_ITERS, BOOTSTRAP_SEED)
        report["P1_sensitivity_both_success"] = {
            "n": len(both),
            "paired_diff_mean": float(sdiffs.mean()),
            "bootstrap_ci95": [slo, shi],
            "wilcoxon": wilcoxon_or_note(sdiffs),
        }
    else:
        report["P1_sensitivity_both_success"] = {"n": 0, "note": "no both-success pairs"}

    # ---- P2 secondary: success rates, exact McNemar ----
    report["P2_success_rate"] = {
        "rate_on": float(np.mean([r["on"]["eval_success"] for r in pairs])),
        "rate_off": float(np.mean([r["off"]["eval_success"] for r in pairs])),
        "mcnemar_exact": mcnemar_exact(pairs),
        "prediction_P2": "on >= off",
    }

    # ---- Secondary descriptives: edges ----
    report["secondary_edges"] = {
        "avg_edges_on": float(np.nanmean([r["on"]["avg_edges"] for r in pairs])),
        "avg_edges_off": float(np.nanmean([r["off"]["avg_edges"] for r in pairs])),
        "inherited_nodes_on_mean": float(np.mean([r["on"]["inherited_nodes"] for r in pairs])),
        "inherited_nodes_off_mean": float(np.mean([r["off"]["inherited_nodes"] for r in pairs])),
    }

    # ---- Verdicts (mechanical application of prereg section 3) ----
    p1 = report["P1_primary_eval_steps"]
    p1_pass = (
        p1["paired_diff_mean_on_minus_off"] < 0
        and p1["bootstrap_ci95"][1] < 0
        and (p1["wilcoxon"]["p"] is not None and p1["wilcoxon"]["p"] < 0.05)
    )
    report["verdicts"] = {
        "P3_apparatus": "PASS" if report["P3_manipulation_check"]["PASS"] else "FAIL (results VOID per prereg §3)",
        "P1": "PASS" if p1_pass else "FAIL -> record defeat per prereg §3 (improvement was curriculum/10D, not sleep)",
        "note": "P2 direction reported descriptively; verdict wording is fixed by the prereg",
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
