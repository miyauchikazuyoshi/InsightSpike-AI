#!/usr/bin/env python3
"""
Auto-sweep L1 gating and backtrack threshold for maze navigator (geDIG wiring).

Goals
 - Find practical settings that: reach goal reliably, minimize steps, trigger BT in stuck zones,
   and keep L1 candidate count K small and stable (≈2–3).

What it sweeps (lightweight defaults; configurable via CLI):
 - sizes: [15, 25]
 - seeds: default [7, 11, 17, 23, 29, 31]
 - L1 tau: [0.80, 0.85, 0.90]
 - filter_unvisited: [1] (can extend)
 - spatial_gate: [6, 8, 10]
 - cand_topk: [3, 5]
 - backtrack_threshold: [-0.025, -0.03, -0.035, -0.04]

Scoring (lower is better):
 - If success rate < 1.0 for a (size, setting), skip from recommendation
 - score = avg_steps + penalty
   - penalty += 20 if avg_BT == 0 (no BT fired) for that size
   - penalty += 5 * max(0, K_mean - 3.5) + 5 * max(0, 2.0 - K_mean)  (soft keep K≈2–3.5)

Outputs
 - Prints a table per size and a final recommendation summary
 - Writes JSON to experiments/maze-navigation-enhanced/results/auto_sweep/<timestamp>/summary.json

Usage
  poetry run python experiments/maze-navigation-enhanced/src/experiments/auto_tune_backtrack_and_l1.py \
    --sizes 15 25 --seeds 7 11 17 23 29 31 --max-steps 600

Note
 - Uses environment knobs already supported by GraphManager / GeDIGEvaluator. geDIG wiring threshold is fixed at -0.04.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
from datetime import datetime

import numpy as np

# Make imports work when executed from repo root
HERE = Path(__file__).resolve()
EXPERIMENTS_SRC = HERE.parent.parent  # .../experiments/maze-navigation-enhanced/src
if str(EXPERIMENTS_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_SRC))

from navigation.maze_navigator import MazeNavigator  # type: ignore


def generate_test_maze(size: int, seed: int) -> np.ndarray:
    import random
    random.seed(seed)
    np.random.seed(seed)

    maze = np.ones((size, size), dtype=int)

    def carve(x: int, y: int) -> None:
        maze[y, x] = 0

    carve(1, 1)

    def neighbors(cx: int, cy: int):
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                yield nx, ny, dx, dy

    stack = [(1, 1)]
    visited = {stack[0]}
    while stack:
        x, y = stack[-1]
        nbs = [(nx, ny, dx, dy) for nx, ny, dx, dy in neighbors(x, y) if (nx, ny) not in visited]
        if not nbs:
            stack.pop(); continue
        nx, ny, dx, dy = random.choice(nbs)
        maze[y + dy // 2, x + dx // 2] = 0
        maze[ny, nx] = 0
        visited.add((nx, ny))
        stack.append((nx, ny))

    maze[size - 2, size - 2] = 0
    return maze


@dataclass
class RunMetrics:
    steps: int
    goal: bool
    bt: int
    dead_end: int
    shortcut: int
    k_min: int
    k_max: int
    k_mean: float


def run_once(size: int, seed: int, max_steps: int, gedig_threshold: float) -> RunMetrics:
    maze = generate_test_maze(size, seed)
    start = (1, 1)
    goal = (size - 2, size - 2)
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='gedig',
        gedig_threshold=gedig_threshold,
        backtrack_threshold=float(os.environ.get('MAZE_BACKTRACK_THRESHOLD', '-0.035')),
        simple_mode=True,
        backtrack_debounce=True,
    )
    nav.run(max_steps=max_steps)

    # Event counts
    from collections import Counter
    cnt = Counter(e['type'] for e in nav.event_log)
    bt = int(cnt.get('backtrack_trigger', 0))
    de = int(cnt.get('dead_end_detected', 0))
    sc = int(cnt.get('shortcut_candidate', 0))
    # K stats
    k_values: List[int] = []
    try:
        k_values.extend(int(log['l1_count']) for log in nav.graph_manager.edge_logs if isinstance(log, dict) and 'l1_count' in log)
    except Exception:
        pass
    if not k_values:
        try:
            k_values = list(getattr(nav.graph_manager, '_l1_candidate_counts', []))  # type: ignore[attr-defined]
        except Exception:
            k_values = []
    k_min = min(k_values) if k_values else 0
    k_max = max(k_values) if k_values else 0
    k_mean = float(np.mean(k_values)) if k_values else 0.0

    return RunMetrics(
        steps=len(nav.path),
        goal=bool(nav.is_goal_reached),
        bt=bt,
        dead_end=de,
        shortcut=sc,
        k_min=int(k_min),
        k_max=int(k_max),
        k_mean=float(k_mean),
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description='Auto-tune backtrack threshold and L1 gating for geDIG maze navigator')
    ap.add_argument('--sizes', type=int, nargs='*', default=[15, 25])
    ap.add_argument('--seeds', type=int, nargs='*', default=[7, 11, 17, 23, 29, 31])
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--gedig-threshold', type=float, default=-0.04)
    ap.add_argument('--taus', type=float, nargs='*', default=[0.80, 0.85, 0.90])
    ap.add_argument('--cand-topk', type=int, nargs='*', default=[3, 5])
    ap.add_argument('--spatial-gates', type=int, nargs='*', default=[6, 8, 10])
    ap.add_argument('--bt-thresholds', type=float, nargs='*', default=[-0.025, -0.03, -0.035, -0.04])
    ap.add_argument('--filter-unvisited', type=int, nargs='*', default=[1])
    args = ap.parse_args()

    # Fixed weights (recommended)
    os.environ['MAZE_L1_WEIGHTED'] = '1'
    os.environ['MAZE_L1_WEIGHTS'] = os.environ.get('MAZE_L1_WEIGHTS', '1,1,0,0,1.5,1.0,0,0')
    os.environ['MAZE_L1_UNIT_NORM'] = '1'
    os.environ['MAZE_L1_NORM_SEARCH'] = '1'
    os.environ['MAZE_GEDIG_LOCAL_NORM'] = '1'
    os.environ['MAZE_USE_HOP_DECISION'] = '1'
    os.environ['MAZE_HOP_DECISION_LEVEL'] = '1'
    os.environ['MAZE_HOP_DECISION_MAX'] = '2'
    os.environ['MAZE_WIRING_TOPK'] = '1'
    os.environ['MAZE_WIRING_MIN_ACCEPT'] = '0'
    os.environ['MAZE_WIRING_FORCE_PREV'] = '1'
    os.environ['MAZE_GEDIG_IG_MODE'] = os.environ.get('MAZE_GEDIG_IG_MODE', 'z')
    os.environ['MAZE_GEDIG_LAMBDA'] = os.environ.get('MAZE_GEDIG_LAMBDA', '3')
    os.environ['MAZE_GEDIG_SP_GAIN'] = os.environ.get('MAZE_GEDIG_SP_GAIN', '1')

    # Query hub optional (kept as-is)
    os.environ['MAZE_USE_QUERY_HUB'] = os.environ.get('MAZE_USE_QUERY_HUB', '1')
    os.environ['MAZE_QUERY_HUB_PERSIST'] = os.environ.get('MAZE_QUERY_HUB_PERSIST', '0')
    os.environ['MAZE_QUERY_HUB_CONNECT_CURRENT'] = os.environ.get('MAZE_QUERY_HUB_CONNECT_CURRENT', '1')

    results: Dict[str, Any] = {
        'run_started_at': datetime.now().isoformat(timespec='seconds'),
        'sizes': args.sizes,
        'seeds': args.seeds,
        'gedig_threshold': args.gedig_threshold,
        'grid': {
            'taus': args.taus,
            'cand_topk': args.cand_topk,
            'spatial_gates': args.spatial_gates,
            'bt_thresholds': args.bt_thresholds,
            'filter_unvisited': args.filter_unvisited,
        },
        'results': {},
        'best': {}
    }

    print('=== Auto tuning sweep ===')
    best_overall = None  # (score, payload)

    for size in args.sizes:
        print(f"\n--- size={size} ---")
        size_best = None
        for fu in args.filter_unvisited:
            os.environ['MAZE_L1_FILTER_UNVISITED'] = str(int(fu))
            for tau in args.taus:
                os.environ['MAZE_L1_NORM_TAU'] = f"{tau}"
                for topk in args.cand_topk:
                    os.environ['MAZE_L1_CAND_TOPK'] = str(int(topk))
                    for gate in args.spatial_gates:
                        os.environ['MAZE_SPATIAL_GATE'] = str(int(gate))
                        for bt_th in args.bt_thresholds:
                            os.environ['MAZE_BACKTRACK_THRESHOLD'] = f"{bt_th}"
                            key = f"fu={fu}|tau={tau}|topk={topk}|gate={gate}|bt={bt_th}"
                            runs: List[RunMetrics] = []
                            for seed in args.seeds:
                                m = run_once(size, seed, args.max_steps, args.gedig_threshold)
                                runs.append(m)
                            # Aggregate
                            success = sum(1 for r in runs if r.goal) / len(runs)
                            if success < 1.0:
                                score = float('inf')
                            else:
                                avg_steps = float(np.mean([r.steps for r in runs]))
                                avg_bt = float(np.mean([r.bt for r in runs]))
                                k_means = [r.k_mean for r in runs if r.k_mean > 0]
                                k_mean = float(np.mean(k_means)) if k_means else 0.0
                                # Penalties
                                penalty = 0.0
                                if avg_bt <= 0.1:
                                    penalty += 20.0
                                if k_mean > 0:
                                    penalty += 5.0 * max(0.0, k_mean - 3.5) + 5.0 * max(0.0, 2.0 - k_mean)
                                score = avg_steps + penalty
                            # Save
                            results['results'].setdefault(str(size), {})[key] = {
                                'success_rate': success,
                                'avg_steps': None if success < 1.0 else float(np.mean([r.steps for r in runs])),
                                'avg_bt': float(np.mean([r.bt for r in runs])),
                                'avg_dead_end': float(np.mean([r.dead_end for r in runs])),
                                'avg_shortcut': float(np.mean([r.shortcut for r in runs])),
                                'k_mean_mean': float(np.mean([r.k_mean for r in runs])) if runs else 0.0,
                                'k_min_min': int(min([r.k_min for r in runs])) if runs else 0,
                                'k_max_max': int(max([r.k_max for r in runs])) if runs else 0,
                                'score': score,
                            }
                            # Track size best
                            if score < float('inf'):
                                payload = (size, fu, tau, topk, gate, bt_th, results['results'][str(size)][key])
                                if (size_best is None) or (score < size_best[0]):
                                    size_best = (score, payload)
                                if (best_overall is None) or (score < best_overall[0]):
                                    best_overall = (score, payload)
        # Print size best
        if size_best:
            sc, (sz, fu, tau, topk, gate, bt_th, stats) = size_best
            print(f"best(size={sz}) fu={fu} tau={tau} topk={topk} gate={gate} bt={bt_th} "
                  f"-> steps={stats['avg_steps']:.1f} BT={stats['avg_bt']:.2f} K~{stats['k_mean_mean']:.2f} score={sc:.1f}")
            results['best'][str(size)] = {
                'fu': fu, 'tau': tau, 'topk': topk, 'gate': gate, 'bt': bt_th, **stats
            }
        else:
            print(f"No valid config (success<1.0) for size={size}")

    print("\n=== Overall recommendation ===")
    if best_overall:
        sc, (sz, fu, tau, topk, gate, bt_th, stats) = best_overall
        print(f"size={sz} fu={fu} tau={tau} topk={topk} gate={gate} bt={bt_th} "
              f"-> steps={stats['avg_steps']:.1f} BT={stats['avg_bt']:.2f} K~{stats['k_mean_mean']:.2f} score={sc:.1f}")
        results['best_overall'] = {
            'size': sz, 'fu': fu, 'tau': tau, 'topk': topk, 'gate': gate, 'bt': bt_th, **stats
        }
    else:
        print("No overall valid config.")
        results['best_overall'] = None

    # Save JSON
    out_root = HERE.parents[3] / 'results' / 'auto_sweep'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'summary.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == '__main__':
    main()

