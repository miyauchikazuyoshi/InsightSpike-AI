#!/usr/bin/env python3
"""
Clean maze experiment runner (Poetry-friendly)

Runs geDIG vs Simple on multiple maze sizes and seeds, computes:
- success_rate
- avg_steps (successes only)
- shortest_path_ratio (avg_steps / BFS shortest)
- redundancy, graph nodes/edges
Writes aggregated JSON to results/scaling/scaling_summary.json
"""

from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator  # type: ignore


def bfs_shortest_path_len(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> int | None:
    from collections import deque
    h, w = maze.shape
    q = deque([(start, 0)])
    visited = set([start])
    while q:
        (x, y), d = q.popleft()
        if (x, y) == goal:
            return d + 1  # steps as number of visited cells in path
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                p = (nx, ny)
                if p not in visited:
                    visited.add(p)
                    q.append((p, d + 1))
    return None


def generate_maze(size: int, seed: int) -> np.ndarray:
    """Simple recursive backtracker with a few loop openings to avoid degenerate cases."""
    rng = np.random.default_rng(seed)
    maze = np.ones((size, size), dtype=int)

    def carve_passage(x: int, y: int) -> None:
        maze[y, x] = 0

    carve_passage(1, 1)

    def neighbors(cx: int, cy: int):
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                yield nx, ny, dx, dy

    stack: List[Tuple[int, int]] = [(1, 1)]
    visited = {(1, 1)}
    while stack:
        x, y = stack[-1]
        nbs = [(nx, ny, dx, dy) for nx, ny, dx, dy in neighbors(x, y) if (nx, ny) not in visited]
        if not nbs:
            stack.pop()
            continue
        nx, ny, dx, dy = nbs[rng.integers(0, len(nbs))]
        maze[y + dy // 2, x + dx // 2] = 0
        maze[ny, nx] = 0
        visited.add((nx, ny))
        stack.append((nx, ny))

    # Add a few random loops to avoid excessive dead-ends
    fast_mode = bool(int(os.environ.get('MAZE_FAST_MODE', '0')))
    loops_target = (max(1, size // 4) if fast_mode else max(2, size // 3))
    attempts = 0
    loops = 0
    max_attempts = (size * size // (2 if fast_mode else 1))
    while loops < loops_target and attempts < max_attempts:
        attempts += 1
        x = int(rng.integers(2, size - 2))
        y = int(rng.integers(2, size - 2))
        if maze[y, x] == 1:
            open_cnt = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and maze[ny, nx] == 0:
                    open_cnt += 1
            if open_cnt >= 2:
                maze[y, x] = 0
                loops += 1

    # Ensure start/goal are open
    maze[1, 1] = 0
    maze[size - 2, size - 2] = 0
    return maze


@dataclass
class TrialResult:
    success: bool
    steps: int
    unique_positions: int
    redundancy: float
    graph_nodes: int
    graph_edges: int


def run_once(maze: np.ndarray, strategy: str, seed: int, max_steps: int) -> TrialResult:
    start = (1, 1)
    goal = (maze.shape[1] - 2, maze.shape[0] - 2)
    # Use non-simple mode to avoid collapsing behaviors
    # Fast mode via env flag for container runs
    fast_mode = bool(int(os.environ.get('MAZE_FAST_MODE', '0')))
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=strategy,
        gedig_threshold=-0.15,
        backtrack_threshold=-0.3,
        simple_mode=fast_mode,
        backtrack_debounce=True,
        wiring_top_k=(3 if fast_mode else 4),
        # Trim heavy instrumentation in fast mode
        enable_diameter_metrics=(False if fast_mode else True),
        dense_metric_interval=(25 if fast_mode else 1),
        snapshot_skip_idle=(True if fast_mode else False),
        max_graph_snapshots=(0 if fast_mode else None),
        enable_flush=False,
        vector_index=None,
        ann_backend=None,
    )
    path: List[Tuple[int, int]] = []
    for _ in range(max_steps):
        _ = nav.step()
        path.append(nav.current_pos)
        if nav.current_pos == goal:
            break
    gstats = nav.graph_manager.get_graph_statistics()
    unique = len(set(path)) if path else 0
    red = (len(path) / unique) if unique else 0.0
    return TrialResult(
        success=(nav.current_pos == goal),
        steps=len(path),
        unique_positions=unique,
        redundancy=red,
        graph_nodes=gstats.get('num_nodes', 0),
        graph_edges=gstats.get('num_edges', 0),
    )


def mean_ci(values: List[float]) -> Tuple[float, float]:
    import math
    if not values:
        return (0.0, 0.0)
    m = sum(values) / len(values)
    # Normal approx 95% CI with sample std
    if len(values) == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(len(values))
    return (m, ci)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Clean geDIG maze runner")
    parser.add_argument('--sizes', type=int, nargs='+', default=[15, 25])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--seed-offset', type=int, default=0, help='Offset index for seed generation to split runs')
    parser.add_argument('--fast', action='store_true', help='Enable faster run (reduced instrumentation)')
    parser.add_argument('--checkpoint', action='store_true', help='Write per-size progress after each seed')
    parser.add_argument('--max-steps-factor', type=float, default=None, help='Override default step budget as factor * size^2')
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'scaling'))
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, Dict[str, float | int | Dict[str, float]]] = {}

    for size in args.sizes:
        print(f"\n=== Size {size}x{size} ===")
        # Pre-build a set of mazes tied to seeds to keep BFS stable per trial
        seeds = [(i + args.seed_offset) * 9973 + 42 for i in range(args.seeds)]

        # Compute BFS shortest on one representative maze (per seed) and average
        bfs_lens: List[int] = []
        simple_steps: List[int] = []
        gedig_steps: List[int] = []
        simple_edges: List[int] = []
        gedig_edges: List[int] = []
        simple_success = 0
        gedig_success = 0

        # Compute max steps budget
        env_factor = os.environ.get('MAZE_MAX_STEPS_FACTOR')
        factor = None
        try:
            factor = float(env_factor) if env_factor is not None else None
        except Exception:
            factor = None
        if args.max_steps_factor is not None:
            factor = args.max_steps_factor
        if factor is None:
            factor = 3.0 if os.environ.get('MAZE_FAST_MODE','0') == '1' else 4.0

        for s in seeds:
            maze = generate_maze(size, s)
            sp = bfs_shortest_path_len(maze, (1, 1), (size - 2, size - 2))
            if sp is not None:
                bfs_lens.append(sp)

            max_steps = int(size * size * factor)
            # Simple
            r_simple = run_once(maze, 'simple', s, max_steps)
            if r_simple.success:
                simple_success += 1
                simple_steps.append(r_simple.steps)
            simple_edges.append(r_simple.graph_edges)

            # geDIG
            r_gedig = run_once(maze, 'gedig', s, max_steps)
            if r_gedig.success:
                gedig_success += 1
                gedig_steps.append(r_gedig.steps)
            gedig_edges.append(r_gedig.graph_edges)

            # Optional per-seed checkpoint for this size
            if args.checkpoint:
                import math
                bfs_mean = (sum(bfs_lens) / len(bfs_lens)) if bfs_lens else math.nan
                simp_mean = (sum(simple_steps) / len(simple_steps)) / 1.0 if simple_steps else math.inf
                ged_mean = (sum(gedig_steps) / len(gedig_steps)) / 1.0 if gedig_steps else math.inf
                progress = {
                    'size': size,
                    'processed_seeds': len(bfs_lens),
                    'success_rate_simple': simple_success / max(1, len(bfs_lens)),
                    'success_rate_gedig': gedig_success / max(1, len(bfs_lens)),
                    'avg_steps_simple': (simp_mean if math.isfinite(simp_mean) else None),
                    'avg_steps_gedig': (ged_mean if math.isfinite(ged_mean) else None),
                    'bfs_shortest_mean': bfs_mean,
                }
                prog_name = f"size_{size}_progress"
                if args.seed_offset:
                    prog_name += f"_offset{args.seed_offset}"
                with open(os.path.join(out_dir, f"{prog_name}.json"), 'w') as pf:
                    json.dump(progress, pf, indent=2)

        import math
        # Averages and CIs
        bfs_mean = (sum(bfs_lens) / len(bfs_lens)) if bfs_lens else math.nan
        simp_mean, simp_ci = mean_ci(simple_steps) if simple_steps else (math.inf, 0.0)
        ged_mean, ged_ci = mean_ci(gedig_steps) if gedig_steps else (math.inf, 0.0)
        # Unify to move-count definition: moves = cells - 1
        def moves_ratio(mean_cells: float, bfs_cells: float) -> float:
            if not (math.isfinite(mean_cells) and math.isfinite(bfs_cells)):
                return math.nan
            mean_moves = max(0.0, mean_cells - 1.0)
            bfs_moves = max(1.0, bfs_cells - 1.0)  # avoid division by zero
            return mean_moves / bfs_moves
        simp_ratio = moves_ratio(simp_mean, bfs_mean)
        ged_ratio = moves_ratio(ged_mean, bfs_mean)

        edge_red = None
        if simple_edges and gedig_edges:
            se = sum(simple_edges) / len(simple_edges)
            ge = sum(gedig_edges) / len(gedig_edges)
            if se > 0:
                edge_red = (se - ge) / se * 100.0

        summary[str(size)] = {
            'seeds': len(seeds),
            'success_rate_simple': simple_success / len(seeds),
            'success_rate_gedig': gedig_success / len(seeds),
            'avg_steps_simple': round(simp_mean, 2) if math.isfinite(simp_mean) else None,
            'avg_steps_gedig': round(ged_mean, 2) if math.isfinite(ged_mean) else None,
            'ci_steps_simple': round(simp_ci, 2) if simp_ci else 0.0,
            'ci_steps_gedig': round(ged_ci, 2) if ged_ci else 0.0,
            'bfs_shortest_mean': bfs_mean,
            'ratio_simple_vs_bfs': round(simp_ratio, 3) if math.isfinite(simp_ratio) else None,
            'ratio_gedig_vs_bfs': round(ged_ratio, 3) if math.isfinite(ged_ratio) else None,
            'edge_reduction_percent': round(edge_red, 2) if edge_red is not None else None,
        }

        print(f"Simple  success={simple_success/len(seeds):.1%}, steps={summary[str(size)]['avg_steps_simple']}±{summary[str(size)]['ci_steps_simple']}")
        print(f"geDIG   success={gedig_success/len(seeds):.1%}, steps={summary[str(size)]['avg_steps_gedig']}±{summary[str(size)]['ci_steps_gedig']}")
        if edge_red is not None:
            print(f"Edge reduction (geDIG vs Simple): {edge_red:.1f}%")

    out_path = os.path.join(out_dir, 'scaling_summary.json')
    spec = {
        'ged_spec_version': os.environ.get('GED_SPEC_VERSION', 'v2-draft'),
        'normalization': {
            'c_v': 1.0,
            'c_e': 1.0,
            'replacement_strategy': 'delete+insert (upper bound)',
            'Cmax_formula': 'max(|V_t|,|V_{t-1}|) + max(|E_t|,|E_{t-1}|)',
            'Z_h': 'L_before(h)'
        },
        'ratio_definition': 'moves = cells - 1',
    }
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'timestamp': time.strftime('%Y%m%d_%H%M%S'), 'spec': spec}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
