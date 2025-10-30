#!/usr/bin/env python3
"""Grid calibration for (k, τ) on maze experiment.

Searches over:
  - k: geDIG IG weight (MazeNavigator.gedig_ig_weight)
  - τ: geDIG decision threshold (MazeNavigator.gedig_threshold)
  - τ_bt: backtrack threshold (MazeNavigator.backtrack_threshold)

Objective (lexicographic):
  1) Maximize success rate
  2) Minimize average steps (successes only)

Writes best params to results/calibration/calibration.json
"""
from __future__ import annotations

import os
import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.clean_maze_run import generate_maze  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore


@dataclass
class Trial:
    success: bool
    steps: int


def run_once(size: int, seed: int, *, k: float, tau: float, tau_bt: float, max_steps_factor: float) -> Trial:
    maze = generate_maze(size, seed)
    start = (1, 1)
    goal = (size - 2, size - 2)
    max_steps = int(size * size * max_steps_factor)
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='gedig',
        gedig_threshold=tau,
        backtrack_threshold=tau_bt,
        gedig_ig_weight=k,
        simple_mode=False
    )
    steps = 0
    for _ in range(max_steps):
        _ = nav.step()
        steps += 1
        if nav.current_pos == goal:
            break
    return Trial(success=(nav.current_pos == goal), steps=steps)


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
    ap = argparse.ArgumentParser(description='Calibrate (k, τ) on maze experiment')
    ap.add_argument('--size', type=int, default=25)
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--k-grid', type=float, nargs='+', default=[0.08, 0.10, 0.12, 0.15])
    ap.add_argument('--tau-grid', type=float, nargs='+', default=[-0.22, -0.18, -0.15, -0.12])
    ap.add_argument('--tau-bt-grid', type=float, nargs='+', default=[-0.30, -0.25, -0.22, -0.18])
    ap.add_argument('--max-steps-factor', type=float, default=4.0)
    args = ap.parse_args()

    seeds = [(i + args.seed_offset) * 9973 + 42 for i in range(args.seeds)]

    results: Dict[str, Dict[str, float]] = {}
    best = None

    for k in args.k_grid:
        for tau in args.tau_grid:
            for tau_bt in args.tau_bt_grid:
                trials: List[Trial] = []
                for s in seeds:
                    trials.append(run_once(args.size, s, k=k, tau=tau, tau_bt=tau_bt, max_steps_factor=args.max_steps_factor))

                success_rate = sum(1 for t in trials if t.success) / max(1, len(trials))
                succ_steps = [t.steps for t in trials if t.success]
                avg_steps, ci_steps = mean_ci(succ_steps) if succ_steps else (float('inf'), 0.0)
                key = f"k={k:.3f},tau={tau:.3f},bt={tau_bt:.3f}"
                results[key] = {
                    'success_rate': round(success_rate, 4),
                    'avg_steps': (None if avg_steps == float('inf') else round(avg_steps, 2)),
                    'avg_steps_ci': round(ci_steps, 2),
                }

                score = (success_rate, -avg_steps)
                if best is None or score > best[0]:
                    best = (score, {'k': k, 'tau': tau, 'tau_bt': tau_bt, 'success_rate': success_rate, 'avg_steps': avg_steps})

    out_dir = Path(__file__).resolve().parents[2] / 'results' / 'calibration'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'grid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    if best is not None:
        with open(out_dir / 'calibration.json', 'w') as f:
            json.dump(best[1], f, indent=2)
        print('✅ Best:', best[1])
    else:
        print('No valid runs')


if __name__ == '__main__':
    main()

