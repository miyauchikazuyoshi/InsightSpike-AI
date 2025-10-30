"""
Quick progress runner for MazeNavigator to observe step rate and timeout step.

Usage:
  python experiments/maze-navigation-enhanced/scripts/quick_run_progress.py \
    --size 50 --seed 11 --max-steps 1200 --strategy gedig

Prints progress every N steps, so we can see at which step the process times out.
This script avoids heavy plotting/export and uses fast-mode navigator settings.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from navigation.maze_navigator import MazeNavigator  # type: ignore


def generate_maze(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    maze = np.ones((size, size), dtype=int)
    def carve(x: int, y: int):
        maze[y, x] = 0
    carve(1, 1)
    def nbs(x: int, y: int):
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < size-1 and 1 <= ny < size-1:
                yield nx, ny, dx, dy
    stack = [(1, 1)]
    seen = {(1, 1)}
    while stack:
        x, y = stack[-1]
        cand = [(nx, ny, dx, dy) for nx, ny, dx, dy in nbs(x, y) if (nx, ny) not in seen]
        if not cand:
            stack.pop(); continue
        nx, ny, dx, dy = cand[rng.integers(0, len(cand))]
        maze[y + dy//2, x + dx//2] = 0
        maze[ny, nx] = 0
        seen.add((nx, ny))
        stack.append((nx, ny))
    # Start/goal
    maze[1, 1] = 0; maze[size-2, size-2] = 0
    return maze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=50)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--max-steps', type=int, default=1200)
    ap.add_argument('--strategy', type=str, default='gedig')
    ap.add_argument('--print-every', type=int, default=25)
    args = ap.parse_args()

    # Fast mode environment (reduce overhead)
    os.environ.setdefault('MAZE_LOG_HOPS', '')
    os.environ.setdefault('MAZE_SPATIAL_GATE', '0')
    os.environ.setdefault('MAZE_WIRING_TOPK', '2')
    os.environ.setdefault('MAZE_WIRING_MIN_ACCEPT', '1')
    os.environ.setdefault('MAZE_WIRING_FORCE_PREV', '0')

    maze = generate_maze(args.size, args.seed)
    start = (1, 1); goal = (args.size-2, args.size-2)

    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=args.strategy,
        gedig_threshold=-0.05,
        backtrack_threshold=-0.2,
        simple_mode=True,
        backtrack_debounce=True,
        enable_diameter_metrics=False,
        dense_metric_interval=50,
        snapshot_skip_idle=True,
        max_graph_snapshots=0,
        vector_index=None,
        ann_backend=None,
        verbosity=0,
        progress_interval=10**9,
    )

    t0 = time.time()
    for step in range(1, args.max_steps+1):
        reached = nav.step()
        if step % args.print_every == 0 or reached:
            dt = time.time() - t0
            print(f"[progress] step={step} dt={dt:.1f}s goal={reached}")
            sys.stdout.flush()
        if reached:
            break
    print(f"[done] steps={step} goal={reached}")


if __name__ == '__main__':
    main()

