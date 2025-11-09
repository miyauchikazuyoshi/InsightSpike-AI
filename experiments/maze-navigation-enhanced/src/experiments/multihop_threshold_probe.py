#!/usr/bin/env python3
"""
Multihop Threshold Probe

Runs a generated maze with geDIG wiring and logs hop0/1/2 metrics for each chosen edge.
Computes simple percentile-based threshold suggestions per hop.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.find_optimal_threshold import generate_test_maze  # type: ignore
from navigation.maze_navigator import MazeNavigator  # type: ignore


def run_probe(size: int, seed: int, max_steps: int) -> dict:
    # Configure lightweight, observation-friendly flags
    os.environ.setdefault('MAZE_GEDIG_LOCAL_NORM', '1')
    os.environ.setdefault('MAZE_L1_NORM_SEARCH', '1')
    os.environ.setdefault('MAZE_L1_WEIGHTED', '1')
    # Observation-rich defaults: expand non-local candidates
    os.environ.setdefault('MAZE_L1_NORM_TAU', '0.9')
    os.environ.setdefault('MAZE_WIRING_WINDOW', '24')
    os.environ.setdefault('MAZE_SPATIAL_GATE', '0')
    os.environ.setdefault('MAZE_L1_WEIGHTS', '0.3,0.3,0,0,3,2,0,0')
    # Observe hop metrics for 0,1,2
    os.environ.setdefault('MAZE_LOG_HOPS', '0,1,2')
    os.environ.setdefault('MAZE_MAX_HOPS', '2')

    maze = generate_test_maze(size, seed)
    start = (1, 1)
    goal = (size - 2, size - 2)

    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='simple',
        gedig_threshold=0.0,
        backtrack_threshold=-0.2,
        simple_mode=False,
        backtrack_debounce=True,
        force_multihop=False,
        global_recall_enabled=True,
        recall_score_threshold=0.03,
    )
    nav.run(max_steps=max_steps)

    # Reconstruct locally and collect approximate structural deltas per hop.
    # We use ΔGED_norm_approx ≈ 1 / (2*(n_after + e_after)) to ensure non-zero scale per hop.
    hop_series: dict[int, list[float]] = {0: [], 1: [], 2: []}
    edge_logs = getattr(nav.graph_manager, 'edge_creation_log', [])
    import networkx as nx
    core = None
    try:
        src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../', 'src'))
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        from insightspike.algorithms.gedig_core import GeDIGCore  # type: ignore
        core = GeDIGCore(enable_multihop=True, max_hops=2, adaptive_hops=False, use_refactored_reward=True)
    except Exception:
        core = None
    if core is not None:
        G = nx.Graph()
        for ev in edge_logs:
            s = ev.get('source'); t = ev.get('target')
            if not isinstance(s, int) or not isinstance(t, int):
                continue
            if not G.has_node(s): G.add_node(s)
            if not G.has_node(t): G.add_node(t)
            local = {s, t}
            local.update(G.neighbors(s))
            local.update(G.neighbors(t))
            g_before = G.subgraph(local).copy()
            g_after = g_before.copy(); g_after.add_edge(s, t)
        try:
            from insightspike.algorithms.linkset_adapter import build_linkset_info as _build_ls  # type: ignore
        except Exception:
            _build_ls = None  # type: ignore
        ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
        if ls is not None:
            res = core.calculate(g_prev=g_before, g_now=g_after, linkset_info=ls)
        else:
            res = core.calculate(g_prev=g_before, g_now=g_after)
            if res.hop_results:
                for h in [0, 1, 2]:
                    if h in res.hop_results:
                        hr = res.hop_results[h]
                        n = int(getattr(hr, 'node_count', 0)); e = int(getattr(hr, 'edge_count', 0))
                        denom = max(1, 2 * (n + e))
                        approx = 1.0 / denom
                        hop_series[h].append(float(approx))
            G.add_edge(s, t)

    # Compute distribution stats and percentile-based suggestions
    summary: dict[int, dict[str, float]] = {}
    for h in [0, 1, 2]:
        vals = np.array(hop_series[h], dtype=float)
        if vals.size == 0:
            summary[h] = {}
            continue
        pct = {
            'p05': float(np.percentile(vals, 5)),
            'p10': float(np.percentile(vals, 10)),
            'p25': float(np.percentile(vals, 25)),
            'p50': float(np.percentile(vals, 50)),
            'p75': float(np.percentile(vals, 75)),
            'p90': float(np.percentile(vals, 90)),
        }
        # Suggest: backtrack threshold near p10 (more conservative), insight near p05 (sharper)
        pct['suggest_backtrack'] = pct['p10']
        pct['suggest_insight'] = pct['p05']
        summary[h] = pct

    return {
        'size': size,
        'seed': seed,
        'max_steps': max_steps,
        'hop_series_len': {h: len(hop_series[h]) for h in [0, 1, 2]},
        'thresholds': summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()

    res = run_probe(args.size, args.seed, args.max_steps)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
