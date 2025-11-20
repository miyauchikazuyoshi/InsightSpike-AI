#!/usr/bin/env python3
"""Quick 50x50 (default) maze exploration experiment using MazeNavigator.

Usage (defaults run a 50x50 maze):
    python examples/maze50_experiment.py --size 50 --wall-prob 0.18 --max-steps 4000 --seed 42 \
            --simple-mode 0 --force-multihop 1

Progress Logging:
    - Add `--verbosity 1` (or higher) to enable periodic progress prints emitted by `MazeNavigator.run()`.
    - Control frequency with `--progress-interval` (default 200 steps). Only active when verbosity>0.

Comparison:
    - verbosity==0 (default): manual step loop (silent until summary)
    - verbosity>0: uses `nav.run(max_steps)` providing START/GOAL/TIMEOUT events & interval progress lines

Outputs a concise summary (steps, goal flag, visited coverage, event counts).
"""
from __future__ import annotations
import os, sys, argparse, time, math, random
import numpy as np
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXP_PATH = os.path.join(ROOT, 'experiments', 'maze-navigation-enhanced', 'src')
if EXP_PATH not in sys.path:
    sys.path.insert(0, EXP_PATH)

from navigation.maze_navigator import MazeNavigator  # type: ignore
import subprocess

def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(['git','rev-parse','--short','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'NA'


def analyze_maze(maze: np.ndarray) -> dict:
    """Return structural diagnostics for the generated maze.

    Metrics:
      - open_cells / wall_cells / wall_ratio (empirical)
      - reachable_cells from start (0,0)
      - goal_reachable (bool)
      - shortest_path_len (if reachable)
      - degree_hist (distribution of open-cell Manhattan degrees 0..4)
      - dead_end_cells (count: open cells with degree==1 excluding start/goal if >1 path)
    """
    from collections import deque, Counter as C
    h, w = maze.shape
    start = (0, 0); goal = (h-1, w-1)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    open_mask = (maze == 0)
    open_cells = int(open_mask.sum())
    wall_cells = h*w - open_cells
    wall_ratio = wall_cells / (h*w)
    # BFS reachability + parent for shortest path
    reachable = set(); parent = {}
    if open_mask[start]:
        q = deque([start]); reachable.add(start); parent[start] = None
        while q:
            r,c = q.popleft()
            for dr,dc in dirs:
                nr,nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and open_mask[nr,nc] and (nr,nc) not in reachable:
                    reachable.add((nr,nc)); parent[(nr,nc)] = (r,c); q.append((nr,nc))
    goal_reachable = goal in reachable
    shortest_len = None
    if goal_reachable:
        cur = goal; path_len = 0
        while cur is not None:
            path_len += 1; cur = parent[cur]
        shortest_len = path_len - 1  # edges count
    # Degree / dead-ends over reachable (or all open if unreachable)
    target_iter = reachable if reachable else {(r,c) for r in range(h) for c in range(w) if open_mask[r,c]}
    deg_hist = C()
    dead_end_cells = 0
    for r,c in target_iter:
        deg = 0
        for dr,dc in dirs:
            nr,nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr,nc]:
                deg += 1
        deg_hist[deg] += 1
        if deg == 1 and (r,c) not in {start, goal}:
            dead_end_cells += 1
    return {
        'size': f'{h}x{w}',
        'open_cells': open_cells,
        'wall_cells': wall_cells,
        'wall_ratio': wall_ratio,
        'reachable_cells': len(reachable),
        'goal_reachable': goal_reachable,
        'shortest_path_len': shortest_len,
        'degree_hist': dict(sorted(deg_hist.items())),
        'dead_end_cells': dead_end_cells,
    }


def ascii_maze(maze: np.ndarray, path: set | None = None) -> str:
    """Return an ASCII representation. Optional path cells highlighted with '*'."""
    h, w = maze.shape
    chars = []
    path = path or set()
    for r in range(h):
        row = []
        for c in range(w):
            if (r,c) == (0,0):
                row.append('S')
            elif (r,c) == (h-1,w-1):
                row.append('G')
            elif maze[r,c] == 1:
                row.append('#')
            else:
                row.append('*' if (r,c) in path else '.')
        chars.append(''.join(row))
    return '\n'.join(chars)


def generate_maze(size: int, wall_prob: float, rng: np.random.Generator, ensure_path: bool = True, max_tries: int = 40) -> np.ndarray:
    """Generate a random maze; optionally enforce start->goal connectivity via BFS retry."""
    def is_connected(grid: np.ndarray) -> bool:
        from collections import deque
        if grid[0,0] == 1 or grid[-1,-1] == 1:
            return False
        h, w = grid.shape
        q = deque([(0,0)])
        seen = {(0,0)}
        while q:
            r, c = q.popleft()
            if (r, c) == (h-1, w-1):
                return True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr,nc) not in seen:
                    seen.add((nr,nc)); q.append((nr,nc))
        return False

    for attempt in range(max_tries):
        maze = (rng.random((size,size)) < wall_prob).astype(int)
        maze[0,0] = 0; maze[-1,-1] = 0
        if not ensure_path or is_connected(maze):
            return maze
    # Fallback: last generated even if disconnected
    return maze


def run_experiment(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    maze = generate_maze(args.size, args.wall_prob, rng, ensure_path=not args.allow_disconnected)

    if args.diagnostics:
        diag = analyze_maze(maze)
        print("=== Maze Diagnostics ===")
        print(f"size={diag['size']} wall_ratio={diag['wall_ratio']:.3f} open={diag['open_cells']} walls={diag['wall_cells']}")
        print(f"reachable={diag['reachable_cells']} goal_reachable={diag['goal_reachable']} shortest_path_len={diag['shortest_path_len']}")
        print(f"dead_end_cells={diag['dead_end_cells']} degree_hist={diag['degree_hist']}")
        if not diag['goal_reachable']:
            print("[WARN] Goal not reachable BEFORE navigation (generation fallback). Consider lower --wall-prob or allow more retries.)")
        if args.ascii:
            print("--- Maze ASCII ---")
            print(ascii_maze(maze))

    # Prepare sweep list
    if args.sweep_visit_weight:
        sweep_list = []
        for token in args.sweep_visit_weight.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                sweep_list.append(float(token))
            except ValueError:
                pass
        if not sweep_list:
            sweep_list = [args.visit_weight_scale] if args.visit_weight_scale else [1.0]
    else:
        sweep_list = [args.visit_weight_scale] if args.visit_weight_scale else [1.0]

    aggregate_rows = []
    run_index = 0
    for scale in sweep_list:
        run_index += 1
        # Optional vector index wiring (query strategy)
        vector_index = None
        if getattr(args, 'use_vector_index', False):
            try:
                from indexes.vector_index import InMemoryIndex  # type: ignore
                vector_index = InMemoryIndex()
            except Exception:
                vector_index = None
        nav = MazeNavigator(
            maze=maze,
            start_pos=(0,0),
            goal_pos=(args.size-1, args.size-1),
            simple_mode=bool(args.simple_mode),
            force_multihop=bool(args.force_multihop),
            dynamic_escalation=True,
            enable_flush=False,
            verbosity=args.verbosity,
            progress_interval=args.progress_interval,
            enable_diameter_metrics=not args.disable_diameter,
            diameter_node_cap=args.diameter_node_cap,
            diameter_time_budget_ms=args.diameter_time_budget_ms,
            backtrack_threshold=args.backtrack_threshold,
            wiring_strategy=getattr(args,'wiring_strategy','simple'),
            vector_index=vector_index,
            wiring_top_k=getattr(args,'query_top_k',4),
        )
        if getattr(args,'wiring_strategy','simple') == 'query':
            try:
                nav.query_wiring_max_dist = getattr(args,'query_max_dist',6.0)
            except Exception:
                pass
        if getattr(args,'nn_degeneracy_trigger',False):
            nav.nn_degeneracy_enabled = True
            nav.nn_deg_var_thresh = args.nn_deg_var_thresh
            nav.nn_deg_range_thresh = args.nn_deg_range_thresh
            nav.nn_deg_min_unvisited_ratio = args.nn_deg_min_unvisited
            nav.nn_deg_min_window_no_growth = args.nn_deg_no_growth_window
        # Apply visit weight scale
        if scale is not None:
            try:
                nav.set_visit_weight_scale(scale)
            except Exception:
                pass
        # Graph growth logging controls
        if args.log_graph_growth:
            nav._log_graph_growth = True
            if args.graph_growth_interval:
                nav._graph_growth_interval = max(1, args.graph_growth_interval)
        # Frontier tuning
        if args.frontier_window is not None:
            nav._frontier_jump_window = args.frontier_window
        if args.frontier_novelty_threshold is not None:
            nav._frontier_novelty_threshold = args.frontier_novelty_threshold
        if args.frontier_cooldown is not None:
            nav._frontier_cooldown = args.frontier_cooldown

        t0 = time.time()
        if args.verbosity > 0:
            nav.run(args.max_steps)
        else:
            for _ in range(args.max_steps):
                if nav.step():
                    break
        dt = time.time() - t0

        stats = nav.get_statistics()
        g = nav.graph_manager.graph
        nodes = g.number_of_nodes(); edges = g.number_of_edges()
        steps_eff = max(1, nav.step_count + 1)
        nodes_per_k = nodes / steps_eff * 1000.0
        edges_per_k = edges / steps_eff * 1000.0
        visited = len(set(nav.path))
        open_cells = int((maze == 0).sum())
        coverage = visited / max(1, open_cells)
        novelty = stats.get('novelty_ratio', None)
        revisit = stats.get('revisit_ratio', None)
        aggregate_rows.append({
            'scale': scale,
            'nodes': nodes,
            'edges': edges,
            'nodes_per_k': nodes_per_k,
            'edges_per_k': edges_per_k,
            'steps': nav.step_count,
            'coverage': coverage,
            'novelty': novelty,
            'revisit': revisit,
            'goal': nav.is_goal_reached
        })
    # Formatting: 条件演算子をフォーマット指定子内部に置くと ValueError になるため分離
    nov_s = f"{novelty:.3f}" if novelty is not None else "NA"
    rev_s = f"{revisit:.3f}" if revisit is not None else "NA"
    print(f"[RUN {run_index}] scale={scale} steps={nav.step_count} goal={nav.is_goal_reached} nodes={nodes} edges={edges} cov={coverage:.3f} nov={nov_s} rev={rev_s} n/k={nodes_per_k:.1f} e/k={edges_per_k:.1f} dt={dt:.2f}s")

    # Summary table
    print("\n=== Sweep Summary (visit weight scale) ===")
    hdr = ["scale","steps","nodes","edges","n_per_k","e_per_k","coverage","novelty","revisit","goal"]
    print("\t".join(hdr))
    for row in aggregate_rows:
        print("\t".join([
            f"{row['scale']}",
            str(row['steps']),
            str(row['nodes']),
            str(row['edges']),
            f"{row['nodes_per_k']:.1f}",
            f"{row['edges_per_k']:.1f}",
            f"{row['coverage']:.3f}",
            f"{row['novelty']:.3f}" if row['novelty'] is not None else 'NA',
            f"{row['revisit']:.3f}" if row['revisit'] is not None else 'NA',
            str(row['goal'])
        ]))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=50)
    p.add_argument('--wall-prob', type=float, default=0.18)
    p.add_argument('--max-steps', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--simple-mode', type=int, default=0, help='1 to enable simple_mode')
    p.add_argument('--force-multihop', type=int, default=1, help='1 to force multihop evaluation')
    p.add_argument('--allow-disconnected', action='store_true', help='Allow disconnected maze (skip connectivity check)')
    p.add_argument('--verbosity', type=int, default=0, help='>0 to enable periodic progress logging')
    p.add_argument('--progress-interval', type=int, default=200, help='Progress print interval when verbosity>0')
    p.add_argument('--diagnostics', type=int, default=1, help='1 to print pre-run maze diagnostics')
    p.add_argument('--ascii', type=int, default=0, help='1 to print ASCII maze (and final path if goal reached)')
    # Performance / metrics gating
    p.add_argument('--disable-diameter', action='store_true', help='Disable expensive diameter metric collection')
    p.add_argument('--diameter-node-cap', type=int, default=900, help='Skip diameter when nodes exceed this cap')
    p.add_argument('--diameter-time-budget-ms', type=float, default=40.0, help='Per-capture time budget for diameter calc')
    # Frontier jump (experimental)
    p.add_argument('--frontier-window', type=int, default=None, help='Override frontier jump sliding window length')
    p.add_argument('--frontier-novelty-threshold', type=float, default=None, help='Override novelty threshold for frontier jump (new cells per step)')
    p.add_argument('--frontier-cooldown', type=int, default=None, help='Override cooldown (steps) between frontier jumps')
    # Exploration weighting
    p.add_argument('--visit-weight-scale', type=float, default=None, help='Scale factor for visit count weight (index 5). >1.0 to favor unexplored')
    p.add_argument('--sweep-visit-weight', type=str, default=None, help='Comma separated list of visit weight scales to sweep, e.g. 1.0,1.5,2.0')
    # Graph growth logging
    p.add_argument('--log-graph-growth', action='store_true', help='Emit graph_growth analysis events periodically')
    p.add_argument('--graph-growth-interval', type=int, default=100, help='Interval for graph growth logging events')
    # Backtrack tuning
    p.add_argument('--backtrack-threshold', type=float, default=-0.2, help='Backtrack trigger threshold (geDIG value <= threshold). Raise (e.g. -0.01 or -0.0002) to make backtracking more likely.')
    # Query wiring / NN 退化オプション
    p.add_argument('--wiring-strategy', type=str, default='simple', choices=['simple','query','ultra','complex','ultra50','ultra50hd'], help='Wiring strategy (enable query for NN degeneracy logic).')
    p.add_argument('--use-vector-index', action='store_true', help='Use in-memory vector index (required for query wiring).')
    p.add_argument('--nn-degeneracy-trigger', action='store_true', help='Enable NN distance distribution degeneracy backtrack trigger.')
    p.add_argument('--nn-deg-var-thresh', type=float, default=1e-4, help='Variance threshold below which distances considered collapsed.')
    p.add_argument('--nn-deg-range-thresh', type=float, default=5e-4, help='Range threshold (max-min) for distance collapse.')
    p.add_argument('--nn-deg-min-unvisited', type=float, default=0.2, help='Min ratio of unvisited among topK; below implies degeneracy.')
    p.add_argument('--nn-deg-no-growth-window', type=int, default=5, help='Window of steps with no graph growth required to trigger.')
    # Query wiring tuning
    p.add_argument('--query-top-k', type=int, default=4, help='Top-K neighbors to wire in query mode (wiring_top_k).')
    p.add_argument('--query-max-dist', type=float, default=6.0, help='Distance cap for query wiring edge acceptance.')
    return p.parse_args(argv)
if __name__ == '__main__':
    args = parse_args()
    # Run metadata header (B step: メタ付与)
    try:
        sweep_raw = getattr(args, 'sweep_visit_weight', None)
        print(f"[META] commit={_git_commit_hash()} seed={args.seed} sweep={sweep_raw or args.visit_weight_scale} size={args.size} wall_prob={args.wall_prob} max_steps={args.max_steps}")
    except Exception:
        pass
    run_experiment(args)
