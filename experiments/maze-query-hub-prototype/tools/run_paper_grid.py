#!/usr/bin/env python3
"""
Batch runner for paper-scale re-tests (maze sizes and trial counts).

Runs the paper comparison (Evaluator vs L3) for a set of (size -> steps,seeds)
pairs, then exports paper-ready metrics and A/B diffs automatically.

Example (paper-like defaults: 15x15=100 seeds, 25x25=60, 51x51=40):

  PYTHONPATH=src INSIGHTSPIKE_LOG_DIR=results/logs MPLCONFIGDIR=results/mpl \
  python experiments/maze-query-hub-prototype/tools/run_paper_grid.py \
    --sizes 15 25 51 \
    --steps-per-size 15:250 25:250 51:250 \
    --seeds-per-size 15:100 25:60 51:40 \
    --out-root experiments/maze-query-hub-prototype/results/paper_grid \
    --namespace-prefix paper_v4 \
    --ultra-light --maze-snapshot-out docs/paper/data/maze_51x51.json

Notes
- Use --ultra-light for speed (steps-ultra-light + no-post-sp-diagnostics).
- Add --union / --force-per-hop if you need per-hop series (L3 path).
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple


def run(cmd: list[str], env: dict | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_kv_pairs(pairs: list[str]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for p in pairs:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        out[int(k)] = int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paper comparison grid (maze)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[15, 25, 51])
    ap.add_argument("--steps-per-size", type=str, nargs="*", default=["15:250", "25:500", "51:1500"], help="size:steps ...")
    ap.add_argument("--seeds-per-size", type=str, nargs="*", default=["15:100", "25:60", "51:40"], help="size:seeds ...")
    ap.add_argument("--ag", type=float, default=0.4)
    ap.add_argument("--rich-logs", action="store_true", help="Propagate --rich-logs to run_paper_comparison.")
    ap.add_argument("--union", action="store_true")
    ap.add_argument("--force-per-hop", action="store_true")
    ap.add_argument("--out-root", type=Path, default=Path("experiments/maze-query-hub-prototype/results/paper_grid"))
    ap.add_argument("--namespace-prefix", type=str, default="paper_v4")
    ap.add_argument("--sqlite", type=Path, default=None)
    ap.add_argument("--maze-snapshot-out", type=Path, default=None)
    args = ap.parse_args()

    steps_map = parse_kv_pairs(args.steps_per_size)
    seeds_map = parse_kv_pairs(args.seeds_per_size)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path("src").resolve()))
    env.setdefault("INSIGHTSPIKE_LOG_DIR", "results/logs")
    env.setdefault("MPLCONFIGDIR", "results/mpl")
    env["INSIGHTSPIKE_PRESET"] = "paper"

    for sz in args.sizes:
        max_steps = steps_map.get(sz, 250)
        seeds = seeds_map.get(sz, 32)
        ns = f"{args.namespace_prefix}_{sz}x{sz}_s{max_steps}"
        sqlite_path = args.sqlite or (out_root / f"mq_{sz}x{sz}.sqlite")
        cmd = [
            "python", "experiments/maze-query-hub-prototype/tools/run_paper_comparison.py",
            "--maze-size", str(sz),
            "--max-steps", str(max_steps),
            "--seeds", str(seeds),
            "--out-root", str(out_root),
            "--namespace", ns,
            "--ag", str(args.ag),
        ]
        if args.rich_logs:
            cmd.append("--rich-logs")
        if args.union:
            cmd.append("--union")
        if args.force_per_hop:
            cmd.append("--force-per-hop")
        if args.maze_snapshot_out:
            cmd += ["--maze-snapshot-out", str(args.maze_snapshot_out)]
        if args.sqlite:
            cmd += ["--sqlite", str(args.sqlite)]
        run(cmd, env)

    print("\n[done] Batch comparison completed. See:")
    print("  Root:", out_root)
    print("  Data:", Path("docs/paper/data").resolve())


if __name__ == "__main__":
    main()
