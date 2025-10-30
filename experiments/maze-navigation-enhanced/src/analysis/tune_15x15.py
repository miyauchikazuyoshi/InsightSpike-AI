#!/usr/bin/env python3
"""
15x15 tuning script

Grid-searches key parameters on 15x15 mazes and reports:
 - success_rate
 - avg_steps (success only) and ratio vs BFS shortest
 - backtrack_step_rate and trigger_count
 - avg graph_edges (structure compactness proxy)

Writes a JSON report under results/tuning_15x15/<timestamp>.json and prints a
ranked top-k table. Designed to be fast and reproducible with a fixed seed set.

Usage (examples):
  PYTHONPATH=experiments/maze-navigation-enhanced/src \
  python experiments/maze-navigation-enhanced/src/analysis/tune_15x15.py \
    --seeds 12 --topk 4 --fast --grid-small

  # More thorough grid
  PYTHONPATH=experiments/maze-navigation-enhanced/src \
  python experiments/maze-navigation-enhanced/src/analysis/tune_15x15.py \
    --seeds 24 --topk 3 4 --gedig-th -0.20 -0.18 -0.15 -0.12 \
    --bt-th -0.30 -0.25 -0.22 -0.18
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator  # type: ignore
from analysis.clean_maze_run import bfs_shortest_path_len, generate_maze  # type: ignore


@dataclass
class TrialResult:
    success: bool
    steps: int
    graph_edges: int
    backtrack_steps: int
    backtrack_triggers: int


def run_once(
    maze: np.ndarray,
    *,
    seed: int,
    gedig_threshold: float,
    backtrack_threshold: float,
    wiring_top_k: int,
    max_steps: int,
    fast_mode: bool,
) -> TrialResult:
    start = (1, 1)
    goal = (maze.shape[1] - 2, maze.shape[0] - 2)
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='gedig',
        gedig_threshold=gedig_threshold,
        backtrack_threshold=backtrack_threshold,
        simple_mode=fast_mode,
        backtrack_debounce=True,
        wiring_top_k=wiring_top_k,
        enable_diameter_metrics=(False if fast_mode else True),
        dense_metric_interval=(25 if fast_mode else 1),
        snapshot_skip_idle=(True if fast_mode else False),
        max_graph_snapshots=(0 if fast_mode else None),
        enable_flush=False,
        vector_index=None,
        ann_backend=None,
    )

    steps = 0
    for _ in range(max_steps):
        steps += 1
        _ = nav.step()
        if nav.current_pos == goal:
            break

    # Metrics
    gstats = nav.graph_manager.get_graph_statistics()
    edges = gstats.get('num_edges', 0)
    # Event-based backtrack counters
    bt_steps = 0
    bt_triggers = 0
    try:
        for ev in getattr(nav, 'event_log', []):
            et = ev.get('type')
            if et == 'backtrack_step':
                bt_steps += 1
            elif et == 'backtrack_trigger':
                bt_triggers += 1
    except Exception:
        pass

    return TrialResult(
        success=(nav.current_pos == goal),
        steps=steps,
        graph_edges=edges,
        backtrack_steps=bt_steps,
        backtrack_triggers=bt_triggers,
    )


def mean_ci(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (math.inf, 0.0)
    m = sum(values) / len(values)
    if len(values) == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(len(values))
    return (m, ci)


def main() -> None:
    ap = argparse.ArgumentParser(description='15x15 geDIG tuning')
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--fast', action='store_true', help='Reduce instrumentation for speed')
    ap.add_argument('--max-steps-factor', type=float, default=None, help='Override default step budget (factor * size^2)')
    ap.add_argument('--topk', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--gedig-th', type=float, nargs='+', default=[-0.20, -0.18, -0.15, -0.12])
    ap.add_argument('--bt-th', type=float, nargs='+', default=[-0.30, -0.25, -0.22, -0.18])
    ap.add_argument('--grid-small', action='store_true', help='Use a smaller grid for quick smoke')
    ap.add_argument('--out', type=str, default=None, help='Output JSON path (default: results/tuning_15x15/<ts>.json)')
    args = ap.parse_args()

    # Environment toggles (stable, fast & robust defaults)
    os.environ.setdefault('NAV_DISABLE_DIAMETER', '1')
    os.environ.setdefault('MAZE_BT_PLAN_FREEZE', '1')
    os.environ.setdefault('MAZE_BT_REPLAN_STUCK_N', '2')
    os.environ.setdefault('MAZE_BACKTRACK_COOLDOWN', '80')
    os.environ.setdefault('MAZE_BT_DYNAMIC', '1')
    os.environ.setdefault('MAZE_GEDIG_LOCAL_NORM', '1')
    os.environ.setdefault('MAZE_GEDIG_SP_GAIN', '1')

    size = 15
    # Budget
    factor = args.max_steps_factor
    if factor is None:
        factor = 3.0 if args.fast or os.environ.get('MAZE_FAST_MODE','0') == '1' else 4.0
    max_steps = int(size * size * factor)

    # Grid selection
    topk_grid = args.topk
    gth_grid = args.gedig_th
    bth_grid = args.bt_th
    if args.grid_small:
        # Quick smoke grid
        topk_grid = sorted(set([min(topk_grid), max(topk_grid)]))
        gth_grid = sorted(set([gth_grid[0], gth_grid[-1]]))
        bth_grid = sorted(set([bth_grid[0], bth_grid[-1]]))

    # Seeds
    seeds = [((i + args.seed_offset) * 9973 + 42) for i in range(args.seeds)]

    # Output dir
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'tuning_15x15'))
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = args.out or os.path.join(out_dir, f'tune15_{ts}.json')

    report: Dict[str, Dict] = {
        'meta': {
            'size': size,
            'seeds': len(seeds),
            'max_steps': max_steps,
            'fast_mode': bool(args.fast),
            'grid': {'topk': topk_grid, 'gedig_th': gth_grid, 'bt_th': bth_grid},
        },
        'combos': {},
        'ranking': [],
    }

    print(f"Tuning 15x15: seeds={len(seeds)} max_steps={max_steps} fast={args.fast}")
    total = len(topk_grid) * len(gth_grid) * len(bth_grid)
    done = 0
    for topk in topk_grid:
        for gth in gth_grid:
            for bth in bth_grid:
                done += 1
                tag = f"topk{topk}_g{gth:.2f}_bt{bth:.2f}"
                print(f"[{done}/{total}] {tag}")

                succ = 0
                steps_succ: List[int] = []
                edges_all: List[int] = []
                bt_steps_all: List[int] = []
                bfs_all: List[int] = []

                for s in seeds:
                    maze = generate_maze(size, s)
                    bfs_len = bfs_shortest_path_len(maze, (1, 1), (size - 2, size - 2))
                    if bfs_len is not None:
                        bfs_all.append(bfs_len)
                    r = run_once(
                        maze,
                        seed=s,
                        gedig_threshold=gth,
                        backtrack_threshold=bth,
                        wiring_top_k=topk,
                        max_steps=max_steps,
                        fast_mode=args.fast,
                    )
                    if r.success:
                        succ += 1
                        steps_succ.append(r.steps)
                    edges_all.append(r.graph_edges)
                    bt_steps_all.append(r.backtrack_steps)

                succ_rate = succ / max(1, len(seeds))
                m_steps, ci_steps = mean_ci([float(x) for x in steps_succ]) if steps_succ else (math.inf, 0.0)
                bfs_mean = (sum(bfs_all) / len(bfs_all)) if bfs_all else math.inf
                ratio = None
                if math.isfinite(m_steps) and math.isfinite(bfs_mean):
                    m_moves = max(0.0, m_steps - 1.0)
                    bfs_moves = max(1.0, bfs_mean - 1.0)
                    ratio = m_moves / bfs_moves
                edges_mean = (sum(edges_all) / len(edges_all)) if edges_all else math.nan
                bt_rate = (sum(bt_steps_all) / sum(steps_succ)) if steps_succ and sum(steps_succ) > 0 else 0.0

                combo = {
                    'success_rate': succ_rate,
                    'avg_steps': None if not math.isfinite(m_steps) else round(m_steps, 2),
                    'ci_steps': None if not math.isfinite(ci_steps) else round(ci_steps, 2),
                    'ratio_vs_bfs': None if ratio is None else round(ratio, 3),
                    'avg_edges': None if not edges_all else round(edges_mean, 1),
                    'backtrack_step_rate': round(bt_rate, 4),
                    'n_success': succ,
                }
                report['combos'][tag] = combo

    # Ranking: primary=success_rate(desc), secondary=ratio_vs_bfs(asc), tertiary=backtrack_rate(asc)
    def _key(kv: Tuple[str, Dict]):
        tag, c = kv
        sr = c.get('success_rate') or 0.0
        ratio = c.get('ratio_vs_bfs')
        br = c.get('backtrack_step_rate') or 0.0
        ratio = (ratio if ratio is not None else float('inf'))
        return (-sr, ratio, br)

    ranked = sorted(report['combos'].items(), key=_key)
    report['ranking'] = [{**{'tag': tag}, **c} for tag, c in ranked[:10]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out_path}")

    # Pretty print top-5
    print("\nTop-5 configs:")
    for r in report['ranking'][:5]:
        print(
            f"  {r['tag']}: success={r['success_rate']:.1%}, "
            f"ratio={r.get('ratio_vs_bfs')}, bt_rate={r.get('backtrack_step_rate')}"
        )


if __name__ == '__main__':
    main()

