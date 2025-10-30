#!/usr/bin/env python3
"""
Unified experiment runner (preset-driven)

Reads thresholds/top‑k from `configs/<preset>.yaml` via the preset loader and
runs a reproducible set of seeds, producing a standard JSON summary.

Usage:
  PYTHONPATH=experiments/maze-navigation-enhanced/src \
  python experiments/maze-navigation-enhanced/src/analysis/unified_runner.py \
    --preset 15x15 --seeds 20 --compare-simple --fast
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preset_loader import load_preset, apply_env  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore
from analysis.clean_maze_run import bfs_shortest_path_len, generate_maze  # type: ignore


def run_once(maze: np.ndarray, *, strategy: str, gth: float, bth: float, topk: int, max_steps: int, fast: bool) -> Dict:
    start = (1, 1)
    goal = (maze.shape[1] - 2, maze.shape[0] - 2)
    nav = MazeNavigator(
        maze=maze, start_pos=start, goal_pos=goal,
        wiring_strategy=strategy,
        gedig_threshold=gth, backtrack_threshold=bth,
        simple_mode=fast, backtrack_debounce=True,
        wiring_top_k=topk,
        enable_diameter_metrics=(False if fast else True),
        dense_metric_interval=(25 if fast else 1),
        snapshot_skip_idle=(True if fast else False),
        max_graph_snapshots=(0 if fast else None),
        enable_flush=False,
        vector_index=None, ann_backend=None,
    )
    steps = 0
    for _ in range(max_steps):
        steps += 1
        _ = nav.step()
        if nav.current_pos == goal:
            break
    gstats = nav.graph_manager.get_graph_statistics()
    ev = getattr(nav, 'event_log', [])
    bt_tr = len([e for e in ev if e.get('type') == 'backtrack_trigger'])
    bt_st = len([e for e in ev if e.get('type') == 'backtrack_step'])
    return {
        'success': bool(nav.current_pos == goal),
        'steps': steps,
        'graph_edges': gstats.get('num_edges', 0),
        'bt_triggers': bt_tr,
        'bt_steps': bt_st,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Unified preset-driven runner')
    ap.add_argument('--preset', type=str, default='15x15', help='15x15, 25x25, 50x50 or custom file name (without .yaml)')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--compare-simple', action='store_true', help='Also run simple baseline')
    args = ap.parse_args()

    cfg = load_preset(preset_name=args.preset)
    apply_env(cfg)  # publish MAZE_* overrides if present

    maze_size = int(((cfg.get('maze') or {}).get('size') or 15))
    max_steps_factor = float(((cfg.get('maze') or {}).get('max_steps_factor') or (3.0 if args.fast else 4.0)))
    gth = float(((cfg.get('gedig') or {}).get('threshold') or -0.15))
    bth = float(((cfg.get('gedig') or {}).get('backtrack_threshold') or -0.30))
    env_map = (cfg.get('env') or {})
    topk = int((env_map.get('MAZE_L1_CAND_TOPK') or 4))

    # Stable environment toggles
    os.environ.setdefault('NAV_DISABLE_DIAMETER', '1')
    os.environ.setdefault('MAZE_BT_PLAN_FREEZE', '1')
    os.environ.setdefault('MAZE_BT_REPLAN_STUCK_N', '2')
    os.environ.setdefault('MAZE_BACKTRACK_COOLDOWN', '80')
    os.environ.setdefault('MAZE_BT_DYNAMIC', '1')
    os.environ.setdefault('MAZE_GEDIG_LOCAL_NORM', '1')
    os.environ.setdefault('MAZE_GEDIG_SP_GAIN', '1')

    max_steps = int(maze_size * maze_size * max_steps_factor)
    seeds = [((i + args.seed_offset) * 9973 + 42) for i in range(args.seeds)]

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'unified'))
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'{args.preset}_{ts}.json')

    def _agg(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return (math.inf, 0.0)
        m = sum(vals) / len(vals)
        if len(vals) == 1:
            return (m, 0.0)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        sd = math.sqrt(var)
        return (m, 1.96 * sd / math.sqrt(len(vals)))

    results: Dict[str, Dict] = {}
    for strat in (['gedig'] + (['simple'] if args.compare_simple else [])):
        succ = 0
        steps_succ: List[float] = []
        bfs_all: List[float] = []
        bt_steps: List[int] = []
        edges_all: List[int] = []
        for s in seeds:
            maze = generate_maze(maze_size, s)
            bfs_len = bfs_shortest_path_len(maze, (1, 1), (maze_size - 2, maze_size - 2))
            if bfs_len is not None:
                bfs_all.append(float(bfs_len))
            r = run_once(maze, strategy=strat, gth=gth, bth=bth, topk=topk, max_steps=max_steps, fast=args.fast)
            if r['success']:
                succ += 1
                steps_succ.append(float(r['steps']))
            bt_steps.append(int(r['bt_steps']))
            edges_all.append(int(r['graph_edges']))

        sr = succ / max(1, len(seeds))
        m_steps, ci_steps = _agg(steps_succ) if steps_succ else (math.inf, 0.0)
        bfs_mean = (sum(bfs_all) / len(bfs_all)) if bfs_all else math.inf
        ratio = None
        if math.isfinite(m_steps) and math.isfinite(bfs_mean):
            m_moves = max(0.0, m_steps - 1.0)
            bfs_moves = max(1.0, bfs_mean - 1.0)
            ratio = m_moves / bfs_moves
        edges_mean = (sum(edges_all) / len(edges_all)) if edges_all else math.nan
        bt_rate = (sum(bt_steps) / sum(steps_succ)) if steps_succ and sum(steps_succ) > 0 else 0.0

        results[strat] = {
            'success_rate': sr,
            'avg_steps': None if not math.isfinite(m_steps) else round(m_steps, 2),
            'ci_steps': None if not math.isfinite(ci_steps) else round(ci_steps, 2),
            'ratio_vs_bfs': None if ratio is None else round(ratio, 3),
            'avg_edges': None if not edges_all else round(edges_mean, 1),
            'backtrack_step_rate': round(bt_rate, 4),
            'n_success': succ,
        }

    report = {
        'preset': args.preset,
        'config': {
            'size': maze_size,
            'max_steps_factor': max_steps_factor,
            'threshold': gth,
            'backtrack_threshold': bth,
            'topk': topk,
            'fast': bool(args.fast),
            'seeds': len(seeds),
        },
        'results': results,
        'timestamp': ts,
    }

    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

