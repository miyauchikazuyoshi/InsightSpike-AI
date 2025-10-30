"""
Quick test for cycle detection in GraphManager.

Build a small maze with some loops, run MazeNavigator with geDIG wiring,
and print edge_creation_log entries that include detected cycles with ΔSP.
"""
from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

import numpy as np

import sys
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
for p in (BASE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from navigation.maze_navigator import MazeNavigator  # type: ignore


def build_loopy_maze(w: int = 15, h: int = 15, seed: int = 42) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    rng = np.random.default_rng(seed)
    maze = np.ones((h, w), dtype=int)
    # carve simple corridors (grid-like)
    for y in range(1, h, 2):
        maze[y, 1:w-1] = 0
    for x in range(1, w, 2):
        maze[1:h-1, x] = 0
    # add random openings to create additional loops
    for _ in range(max(10, (w*h)//20)):
        x = int(rng.integers(1, w-1))
        y = int(rng.integers(1, h-1))
        maze[y, x] = 0
    start = (1, 1)
    goal = (w-2, h-2)
    maze[start[1], start[0]] = 0
    maze[goal[1], goal[0]] = 0
    return maze, start, goal


def main() -> None:
    # Soften wiring to ensure edges get added and cycles appear
    os.environ['MAZE_WIRING_MIN_ACCEPT'] = '1'     # accept best even if over threshold
    os.environ['MAZE_WIRING_TOPK'] = '2'           # accept up to 2 edges per step
    os.environ['MAZE_WIRING_FORCE_PREV'] = '0'     # avoid trajectory auto-edges to highlight geDIG edges
    os.environ['MAZE_L1_NORM_SEARCH'] = '1'        # enable candidate recall
    os.environ['MAZE_L1_NORM_TAU'] = '999'         # be permissive
    os.environ['MAZE_SPATIAL_GATE'] = '0'          # no spatial restriction
    os.environ['MAZE_LOG_HOPS'] = ''               # keep hop logging off for speed

    maze, start, goal = build_loopy_maze()
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy='gedig',
        gedig_threshold=-0.05,  # default negative (minimize F)
        backtrack_threshold=-0.1,
        verbosity=0,
        progress_interval=200,
        enable_diameter_metrics=False,
    )
    nav.run(max_steps=300)

    # Extract cycle logs
    gm = nav.graph_manager
    cycles: List[Dict[str, Any]] = []
    for rec in getattr(gm, 'edge_creation_log', []):
        cyc = rec.get('cycle') if isinstance(rec, dict) else None
        if cyc and isinstance(cyc, dict):
            cycles.append({
                'from': rec.get('source'),
                'to': rec.get('target'),
                'delta_sp': cyc.get('delta_sp'),
                'path_len': (len(cyc.get('node_path') or []) or None),
            })

    print(f"[cycle-test] total edges: {len(getattr(gm, 'edge_creation_log', []))}")
    print(f"[cycle-test] cycles detected: {len(cycles)}")
    for i, c in enumerate(cycles[:10], 1):
        print(f"  #{i}: from={c['from']} to={c['to']} ΔSP={c['delta_sp']:.1f} path_len={c['path_len']}")
    if len(cycles) > 10:
        print(f"  ... ({len(cycles)-10} more)")


if __name__ == '__main__':
    main()
