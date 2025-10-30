#!/usr/bin/env python3
"""Compute statistical summary (mean±CI, win-rate) for maze runs.

Runs both 'simple' and 'gedig' strategies across a size and seeds,
reports:
  - success_rate (± no CI since Bernoulli small-N; show proportion)
  - avg_steps (successes only) ±95% CI
  - graph_edges mean ±95% CI (proxy for compression)
  - win_rate (gedig vs simple) by steps (lower is win)
"""
from __future__ import annotations

import os
import argparse
from typing import List, Tuple, Dict
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.clean_maze_run import generate_maze, run_once, TrialResult  # type: ignore


def mean_ci(values: List[float]) -> Tuple[float, float]:
    import math
    if not values:
        return (0.0, 0.0)
    m = sum(values) / len(values)
    if len(values) == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(len(values))
    return (m, ci)


def main() -> None:
    ap = argparse.ArgumentParser(description='Maze statistical summary for simple vs geDIG')
    ap.add_argument('--size', type=int, default=25)
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--max-steps-factor', type=float, default=None)
    args = ap.parse_args()

    seeds = [i * 9973 + 42 for i in range(args.seeds)]

    # apply optional factor to ENV so run_once picks it up
    if args.max_steps_factor is not None:
        os.environ['MAZE_MAX_STEPS_FACTOR'] = str(args.max_steps_factor)

    simp_succ = 0; ged_succ = 0
    simp_steps: List[int] = []; ged_steps: List[int] = []
    simp_edges: List[int] = []; ged_edges: List[int] = []
    wins = 0; total = 0

    for s in seeds:
        maze = generate_maze(args.size, s)
        r_simple: TrialResult = run_once(maze, 'simple', s, max_steps=int((args.max_steps_factor or 4.0) * args.size * args.size))
        r_gedig: TrialResult = run_once(maze, 'gedig', s, max_steps=int((args.max_steps_factor or 4.0) * args.size * args.size))
        if r_simple.success:
            simp_succ += 1; simp_steps.append(r_simple.steps)
        if r_gedig.success:
            ged_succ += 1; ged_steps.append(r_gedig.steps)
        simp_edges.append(r_simple.graph_edges); ged_edges.append(r_gedig.graph_edges)
        # win by fewer steps (only if both solved)
        if r_simple.success and r_gedig.success:
            total += 1
            if r_gedig.steps < r_simple.steps:
                wins += 1

    from math import isfinite
    def fmt_mean_ci(vals: List[int]) -> str:
        if not vals:
            return 'n/a'
        m, c = mean_ci([float(v) for v in vals])
        return f"{m:.1f} ± {c:.1f}"

    print("=== Statistical Summary ===")
    print(f"Size: {args.size}, Seeds: {len(seeds)}")
    print(f"Success rate (simple): {simp_succ/len(seeds):.3f}")
    print(f"Success rate (geDIG): {ged_succ/len(seeds):.3f}")
    print(f"Avg steps (simple):   {fmt_mean_ci(simp_steps)}")
    print(f"Avg steps (geDIG):    {fmt_mean_ci(ged_steps)}")
    print(f"Edges (simple):       {fmt_mean_ci(simp_edges)}")
    print(f"Edges (geDIG):        {fmt_mean_ci(ged_edges)}")
    if total:
        print(f"Win rate by steps (geDIG vs simple): {wins/total:.3f} ({wins}/{total})")

if __name__ == '__main__':
    main()

