#!/usr/bin/env python3
"""
Grid runner to raise 25x25 success-rate for Sleep-Q prefer mode.

This script runs the Wake→Sleep→Wake curriculum with --sleep-guide prefer and
prints a compact table so you can tune:
  - sleep_q_beta (Q prior strength)
  - sleep_plan_beta (optional BFS-plan soft bias; still NOT override)

Example (fast-ish defaults, 10 seeds):

  PYTHONPATH=src \
  .venv311/bin/python experiments/maze-query-hub-prototype/tools/run_sleepq_prefer_grid.py \
    --out-root results/maze-local/sleepq_prefer_grid \
    --seeds 10 --seed-start 0 \
    --max-steps 200 --warmup-steps 600 \
    --sleep-q-betas 4 6 8 \
    --sleep-plan-betas 0 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _run(cmd: List[str], *, env: Dict[str, str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _load_result(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(obj: Dict[str, Any]) -> Tuple[float, float, float, int]:
    """Return (success_rate, avg_steps, warmup_success_rate, eval_fail_count)."""
    summary = obj.get("summary", {}) or {}
    success_rate = float(summary.get("success_rate", 0.0) or 0.0)
    avg_steps = float(summary.get("avg_steps", 0.0) or 0.0)

    warmup_runs = obj.get("warmup_runs", []) or []
    warmup_success_rate = 0.0
    try:
        warmup_success_rate = sum(1 for r in warmup_runs if bool(r.get("success"))) / float(max(1, len(warmup_runs)))
    except Exception:
        warmup_success_rate = 0.0

    per_seed = (obj.get("curriculum", {}) or {}).get("per_seed", {}) or {}
    eval_fail = 0
    try:
        for v in per_seed.values():
            ev = (v.get("eval", {}) or {})
            if not bool(ev.get("success")):
                eval_fail += 1
    except Exception:
        eval_fail = 0

    return success_rate, avg_steps, warmup_success_rate, int(eval_fail)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sleep-Q prefer grid runner (maze 25x25)")
    ap.add_argument("--maze-size", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=200, help="Eval max steps")
    ap.add_argument("--warmup-steps", type=int, default=600, help="Warmup cap for curriculum")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=Path("results/maze-local/sleepq_prefer_grid"))
    ap.add_argument("--sleep-q-betas", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    ap.add_argument("--sleep-plan-betas", type=float, nargs="+", default=[0.0, 2.0])
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--theta-ag", type=float, default=0.2)
    ap.add_argument("--theta-dg", type=float, default=0.15)
    ap.add_argument("--commit-budget", type=int, default=2)
    ap.add_argument("--sp-pair-samples", type=int, default=40)
    ap.add_argument("--l1-cap", type=int, default=32)
    ap.add_argument("--python", type=str, default=None, help="Python interpreter to run run_experiment_query.py")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path("src").resolve()))

    py = args.python or env.get("PYTHON", None) or sys.executable

    # Fixed "fast-ish" base args: keep consistent with repo paper-ish settings.
    base_cmd = [
        py,
        "experiments/maze-query-hub-prototype/run_experiment_query.py",
        "--preset",
        "paper",
        "--maze-size",
        str(int(args.maze_size)),
        "--max-steps",
        str(int(args.max_steps)),
        "--seeds",
        str(int(args.seeds)),
        "--seed-start",
        str(int(args.seed_start)),
        "--use-main-l3",
        "--sp-cache",
        "--sp-cache-mode",
        "cached_incr",
        "--sp-pair-samples",
        str(int(args.sp_pair_samples)),
        "--max-hops",
        str(int(args.max_hops)),
        "--theta-ag",
        str(float(args.theta_ag)),
        "--theta-dg",
        str(float(args.theta_dg)),
        "--commit-budget",
        str(int(args.commit_budget)),
        "--dg-commit-all-linkset",
        "--skip-mh-on-deadend",
        "--layer1-prefilter",
        "--l1-cap",
        str(int(args.l1_cap)),
        "--cortisol-mode",
        "log",
        "--cortisol-repeat-visits",
        "2",
        "--no-pre-eval",
        "--no-post-sp-diagnostics",
        "--snapshot-level",
        "minimal",
        "--log-minimal",
        "--steps-ultra-light",
        "--sp-ds-sqlite",
        "",
        "--curriculum-warmup-steps",
        str(int(args.warmup_steps)),
        "--sleep-guide",
        "prefer",
    ]

    rows: List[Dict[str, Any]] = []
    print("\n[grid] maze={} seeds={} eval_steps={} warmup_cap={}".format(args.maze_size, args.seeds, args.max_steps, args.warmup_steps))
    for q_beta in args.sleep_q_betas:
        for p_beta in args.sleep_plan_betas:
            tag = f"mq{int(args.maze_size)}_s{int(args.max_steps)}_w{int(args.warmup_steps)}_n{int(args.seeds)}_qb{q_beta:g}_pb{p_beta:g}"
            out_json = out_root / f"{tag}.json"
            cmd = list(base_cmd) + [
                "--sleep-q-beta",
                str(float(q_beta)),
                "--sleep-plan-beta",
                str(float(p_beta)),
                "--output",
                str(out_json),
            ]
            _run(cmd, env=env)
            obj = _load_result(out_json)
            success_rate, avg_steps, warmup_sr, eval_fail = _extract_metrics(obj)
            row = {
                "tag": tag,
                "sleep_q_beta": float(q_beta),
                "sleep_plan_beta": float(p_beta),
                "success_rate": float(success_rate),
                "avg_steps": float(avg_steps),
                "warmup_success_rate": float(warmup_sr),
                "eval_fail_count": int(eval_fail),
                "output": str(out_json),
            }
            rows.append(row)
            print(
                "  qb={:>5} pb={:>5} | success={:.3f} avg_steps={:.1f} warmup={:.3f} eval_fail={}".format(
                    f"{q_beta:g}",
                    f"{p_beta:g}",
                    success_rate,
                    avg_steps,
                    warmup_sr,
                    eval_fail,
                )
            )

    grid_out = out_root / "grid_summary.json"
    grid_out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\n[done] wrote:", grid_out)


if __name__ == "__main__":
    main()
