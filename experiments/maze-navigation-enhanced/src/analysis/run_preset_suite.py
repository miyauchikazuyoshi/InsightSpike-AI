#!/usr/bin/env python3
"""Run preset application, calibration, and stats in one go.

Example:
  PYTHONPATH=experiments/maze-navigation-enhanced/src \
  python experiments/maze-navigation-enhanced/src/analysis/run_preset_suite.py --preset 25x25
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import List

from utils.preset_loader import load_preset, apply_env
from analysis.calibrate_ktau import run_once, mean_ci  # type: ignore
from analysis.clean_maze_run import generate_maze  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description='Preset -> Calibrate -> Stats')
    ap.add_argument('--preset', type=str, default='25x25')
    ap.add_argument('--k-delta', type=float, nargs='+', default=[-0.02, 0.0, +0.02, +0.05])
    ap.add_argument('--tau-delta', type=float, nargs='+', default=[-0.04, -0.02, 0.0, +0.03])
    ap.add_argument('--bt-delta', type=float, nargs='+', default=[-0.08, -0.04, 0.0, +0.04])
    args = ap.parse_args()

    cfg = load_preset(preset_name=args.preset)
    apply_env(cfg)

    maze_cfg = cfg.get('maze') or {}
    size = int(maze_cfg.get('size', 25))
    seeds_n = int(maze_cfg.get('seeds', 16))
    factor = float(maze_cfg.get('max_steps_factor', 4.0))
    seeds = [i * 9973 + 42 for i in range(seeds_n)]

    gedig = cfg.get('gedig') or {}
    k0 = float(gedig.get('ig_weight', 0.10))
    tau0 = float(gedig.get('threshold', -0.15))
    bt0 = float(gedig.get('backtrack_threshold', -0.20))

    k_grid = [round(k0 + d, 4) for d in args.k_delta]
    tau_grid = [round(tau0 + d, 4) for d in args.tau_delta]
    bt_grid = [round(bt0 + d, 4) for d in args.bt_delta]

    best = None
    grid_results = {}
    for k in k_grid:
        for tau in tau_grid:
            for bt in bt_grid:
                trials = []
                for s in seeds:
                    trials.append(run_once(size, s, k=k, tau=tau, tau_bt=bt, max_steps_factor=factor))
                succ = sum(1 for t in trials if t.success)
                srate = succ / max(1, len(trials))
                succ_steps = [t.steps for t in trials if t.success]
                avg, ci = mean_ci([float(x) for x in succ_steps]) if succ_steps else (float('inf'), 0.0)
                key = f"k={k},tau={tau},bt={bt}"
                grid_results[key] = {'success_rate': round(srate, 4), 'avg_steps': None if avg == float('inf') else round(avg, 2), 'avg_steps_ci': round(ci, 2)}
                score = (srate, -avg)
                if best is None or score > best[0]:
                    best = (score, {'k': k, 'tau': tau, 'tau_bt': bt, 'success_rate': srate, 'avg_steps': avg})

    out_dir = Path(__file__).resolve().parents[2] / 'results' / 'calibration'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'grid_results.json', 'w') as f:
        json.dump(grid_results, f, indent=2)
    if best is not None:
        with open(out_dir / 'calibration.json', 'w') as f:
            json.dump(best[1], f, indent=2)
        print('✅ Best:', best[1])

    # Stats summary (simple vs geDIG with best params if available)
    from analysis.clean_maze_run import run_once as run_simple  # type: ignore
    simp_succ = 0; ged_succ = 0
    simp_steps: List[int] = []; ged_steps: List[int] = []
    simp_edges: List[int] = []; ged_edges: List[int] = []
    wins = 0; total = 0
    for s in seeds:
        maze = generate_maze(size, s)
        ms = int(size*size*factor)
        rs = run_simple(maze, 'simple', s, ms)
        rt = run_simple(maze, 'gedig', s, ms)
        if rs.success:
            simp_succ += 1; simp_steps.append(rs.steps)
        if rt.success:
            ged_succ += 1; ged_steps.append(rt.steps)
        simp_edges.append(rs.graph_edges); ged_edges.append(rt.graph_edges)
        if rs.success and rt.success:
            total += 1
            if rt.steps < rs.steps:
                wins += 1
    m_s, c_s = mean_ci([float(x) for x in simp_steps]) if simp_steps else (0.0, 0.0)
    m_g, c_g = mean_ci([float(x) for x in ged_steps]) if ged_steps else (0.0, 0.0)
    stats = {
        'size': size,
        'seeds': seeds_n,
        'success_rate_simple': round(simp_succ/max(1,len(seeds)), 4),
        'success_rate_gedig': round(ged_succ/max(1,len(seeds)), 4),
        'avg_steps_simple': round(m_s,2), 'avg_steps_simple_ci': round(c_s,2),
        'avg_steps_gedig': round(m_g,2), 'avg_steps_gedig_ci': round(c_g,2),
        'win_rate_steps': (wins, total, round(wins/max(1,total),3)),
    }
    with open(Path(__file__).resolve().parents[2] / 'results' / 'calibration' / 'stats_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print('=== Summary ===')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

