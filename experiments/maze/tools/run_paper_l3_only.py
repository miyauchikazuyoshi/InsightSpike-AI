#!/usr/bin/env python3
"""
Paper v4 – L3-only runner (multi-hop, query-centric) for maze experiments.

Runs L3 (paper preset) with multi-hop enabled via core SP engine (no union
fallback). Suitable for producing the final experiment results without the
Evaluator route (A/B is kept for internal verification only).

Example:
  PYTHONPATH=src INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl \
  INSIGHTSPIKE_PRESET=paper \
  python experiments/maze-query-hub-prototype/tools/run_paper_l3_only.py \
    --maze-size 25 --max-steps 250 --seeds 60 \
    --out-root experiments/maze-query-hub-prototype/results/l3_only \
    --namespace l3only_25x25_s250 --ultra-light \
    --maze-snapshot-out docs/paper/data/maze_25x25.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str], env: dict | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper L3-only (multi-hop) runner")
    ap.add_argument("--maze-size", type=int, default=51)
    ap.add_argument("--maze-type", type=str, default="dfs")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--ag", type=float, default=0.4)
    ap.add_argument("--rich-logs", action="store_true", help="Record full snapshots (default: minimal + steps-ultra-light)")
    ap.add_argument("--out-root", type=Path, default=Path("experiments/maze-query-hub-prototype/results/l3_only"))
    ap.add_argument("--namespace", type=str, default="l3only")
    ap.add_argument("--sqlite", type=Path, default=None)
    ap.add_argument("--maze-snapshot-out", type=Path, default=None)
    args = ap.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = args.sqlite or (out_root / f"mq_{args.maze_size}x{args.maze_size}.sqlite")
    ns = args.namespace

    snapshot_level = "standard" if args.rich_logs else "minimal"

    common = [
        "--maze-size", str(args.maze_size),
        "--maze-type", str(args.maze_type),
        "--max-steps", str(args.max_steps),
        "--seeds", str(args.seeds),
        "--seed-start", str(args.seed_start),
        "--linkset-mode",
        "--norm-base", "link",
        "--theta-ag", str(args.ag),
        "--top-link", "1",
        "--sp-cache",
        "--sp-cache-mode", "cached_incr",
        "--sp-pair-samples", "600",
        "--sp-cand-topk", "0",
        "--l1-cap", "128",
        "--link-forced-as-base",
        "--timeline-to-graph",
        "--add-next-q",
        "--persist-timeline-edges",
        "--link-radius", "1.0",
        "--cand-radius", "1.0",
        "--theta-link", "0.0",
        "--theta-cand", "0.0",
        "--snapshot-level", snapshot_level,
    ]
    if args.maze_snapshot_out:
        common += ["--maze-snapshot-out", str(args.maze_snapshot_out)]
    if not args.rich_logs:
        common += ["--steps-ultra-light", "--no-post-sp-diagnostics"]

    summary = out_root / f"_{args.maze_size}x{args.maze_size}_s{args.max_steps}_l3only_summary.json"
    steps = out_root / f"_{args.maze_size}x{args.maze_size}_s{args.max_steps}_l3only_steps.json"
    cmd = [
        "python", "experiments/maze-query-hub-prototype/run_experiment_query.py",
        *common,
        "--use-main-l3",
        "--persist-graph-sqlite", str(sqlite_path),
        "--persist-namespace", ns,
        "--persist-forced-candidates",
        "--output", str(summary),
        "--step-log", str(steps),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path("src").resolve()))
    env.setdefault("INSIGHTSPIKE_LOG_DIR", "results/logs")
    env.setdefault("MPLCONFIGDIR", "results/mpl")
    env.setdefault("INSIGHTSPIKE_PRESET", "paper")
    run(cmd, env)

    print("\n[done] L3-only outputs:")
    print("  Summary:", summary)
    print("  Steps  :", steps)
    print("  SQLite :", sqlite_path)


if __name__ == "__main__":
    main()
