#!/usr/bin/env python3
"""
Run paper-oriented baseline comparison for Query-Hub (maze) experiments.

This script launches two runs under identical conditions:
  1) Evaluator route (reference evaluator in driver)
  2) L3 route (paper preset, query-centric hop0)

Then it exports paper-ready metrics (JSON/CSV) and produces an A/B delta JSON.

Example (recommended 51x51 / 200 steps):
  PYTHONPATH=src INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl \
  python experiments/maze-query-hub-prototype/tools/run_paper_comparison.py \
    --maze-size 51 --max-steps 200 --out-root experiments/maze-query-hub-prototype/results \
    --namespace paper_51x51_s200 --maze-snapshot-out docs/paper/data/maze_51x51.json

Notes
- Use --ultra-light for speed on longer runs (skips heavy snapshots in steps).
- Add --union to record per-hop series via evaluator fallback (L3 path only); for
  consistent per-hop recording, pass --force-per-hop (sets AG=-1.0 for L3 run).
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run(cmd: list[str], env: dict | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper comparison runner (maze)")
    ap.add_argument("--maze-size", type=int, default=51)
    ap.add_argument("--maze-type", type=str, default="dfs")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--ag", type=float, default=None, help="(deprecated) AG threshold for both; prefer --ag-eval/--ag-l3")
    ap.add_argument("--ag-eval", type=float, default=-1.0, help="AG threshold for evaluator route (g0<th -> skip). Default: -1.0 (no skip)")
    ap.add_argument("--ag-l3", type=float, default=-0.30, help="AG threshold for L3 route (g0>th -> per-hop fallback). Default: -0.30")
    ap.add_argument("--union", action="store_true", help="L3: record per-hop via union scope (evaluator fallback)")
    ap.add_argument("--force-per-hop", action="store_true", help="L3: set AG=-1.0 to ensure per-hop is recorded")
    ap.add_argument("--rich-logs", action="store_true", help="Record full snapshots; default is minimal + steps-ultra-light.")
    ap.add_argument("--out-root", type=Path, default=Path("experiments/maze-query-hub-prototype/results"))
    ap.add_argument("--namespace", type=str, default="paper_run", help="SQLite namespace when DS used")
    ap.add_argument("--sqlite", type=Path, default=None, help="Optional DS SQLite path")
    ap.add_argument("--maze-snapshot-out", type=Path, default=None, help="Optional path to save maze snapshot JSON")
    args = ap.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = args.sqlite or (out_root / f"mq_{args.maze_size}x{args.maze_size}.sqlite")
    ns = args.namespace

    snapshot_level = "standard" if args.rich_logs else "minimal"

    # Common flags
    common = [
        "--maze-size", str(args.maze_size),
        "--maze-type", str(args.maze_type),
        "--max-steps", str(args.max_steps),
        "--seeds", str(args.seeds),
        "--seed-start", str(args.seed_start),
        "--linkset-mode",
        "--norm-base", "link",
        # NOTE: --theta-ag is set per-route below
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
        "--persist-forced-candidates",
        "--link-radius", "1.0",
        "--cand-radius", "1.0",
        "--theta-link", "0.0",
        "--theta-cand", "0.0",
        "--snapshot-level", snapshot_level,
    ]
    if args.maze_snapshot_out:
        common += ["--maze-snapshot-out", str(args.maze_snapshot_out)]

    # Ultra-light path switches
    ultraflags = [] if args.rich_logs else ["--steps-ultra-light", "--no-post-sp-diagnostics"]

    # Evaluator route
    eval_sum = out_root / f"_{args.maze_size}x{args.maze_size}_s{args.max_steps}_eval_summary.json"
    eval_steps = out_root / f"_{args.maze_size}x{args.maze_size}_s{args.max_steps}_eval_steps.json"
    ag_eval = args.ag if args.ag is not None else args.ag_eval
    ag_l3 = args.ag if args.ag is not None else args.ag_l3

    py = sys.executable or "python"
    cmd_eval = [
        py, "experiments/maze-query-hub-prototype/run_experiment_query.py",
        *common, *ultraflags,
        "--theta-ag", str(ag_eval),
        "--persist-graph-sqlite", str(sqlite_path),
        "--persist-namespace", ns + "_eval",
        "--persist-forced-candidates",
        "--output", str(eval_sum),
        "--step-log", str(eval_steps),
    ]

    # L3 route (paper)
    l3_sum = out_root / f"_{args.maze_size}x{args.maze_size}_s{args.max_steps}_l3_summary.json"
    l3_steps = out_root / f"_{args.maze_size}x{args.maze_size}_s{args.max_steps}_l3_steps.json"
    l3_ag = -1.0 if args.force_per_hop else args.ag
    cmd_l3 = [
        py, "experiments/maze-query-hub-prototype/run_experiment_query.py",
        *common,
        "--theta-ag", str(ag_l3 if not args.force_per_hop else -1.0),
        *( ["--sp-scope", "union"] if args.union else [] ),
        *ultraflags,
        "--use-main-l3",
        "--eval-per-hop-on-ag",
        "--persist-graph-sqlite", str(sqlite_path),
        "--persist-namespace", ns + "_l3",
        "--persist-forced-candidates",
        "--output", str(l3_sum),
        "--step-log", str(l3_steps),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path("src").resolve()))
    env.setdefault("INSIGHTSPIKE_LOG_DIR", "results/logs")
    env.setdefault("MPLCONFIGDIR", "results/mpl")
    env["INSIGHTSPIKE_PRESET"] = "paper"

    # Run both
    run(cmd_eval, env)
    run(cmd_l3, env)

    # Export metrics for paper (JSON/CSV)
    data_dir = Path("docs/paper/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    base_json = data_dir / f"maze_{args.maze_size}x{args.maze_size}_eval_s{args.max_steps}.json"
    base_csv = data_dir / f"maze_{args.maze_size}x{args.maze_size}_eval_s{args.max_steps}.csv"
    var_json = data_dir / f"maze_{args.maze_size}x{args.maze_size}_l3_s{args.max_steps}.json"
    var_csv = data_dir / f"maze_{args.maze_size}x{args.maze_size}_l3_s{args.max_steps}.csv"
    run([
        py, "experiments/maze-query-hub-prototype/tools/export_paper_maze.py",
        "--summary", str(eval_sum), "--steps", str(eval_steps),
        "--out-json", str(base_json), "--out-csv", str(base_csv),
        "--compression-base", "mem",
    ], env)
    run([
        py, "experiments/maze-query-hub-prototype/tools/export_paper_maze.py",
        "--summary", str(l3_sum), "--steps", str(l3_steps),
        "--out-json", str(var_json), "--out-csv", str(var_csv),
        "--compression-base", "mem",
    ], env)

    # Compare baseline vs variant
    diff_out = data_dir / f"ab_eval_vs_l3_{args.maze_size}x{args.maze_size}_s{args.max_steps}.json"
    run([
        py, "experiments/maze-query-hub-prototype/baselines/compare_runs.py",
        "--base-summary", str(eval_sum), "--base-steps", str(eval_steps),
        "--var-summary", str(l3_sum), "--var-steps", str(l3_steps),
        "--out", str(diff_out),
    ], env)

    print("\n[done] Outputs:")
    print("  Eval  :", eval_sum)
    print("  EvalT :", eval_steps)
    print("  L3    :", l3_sum)
    print("  L3T   :", l3_steps)
    print("  Export:", base_json, ",", var_json)
    print("  Diff  :", diff_out)


if __name__ == "__main__":
    main()
