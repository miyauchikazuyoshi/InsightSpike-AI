#!/usr/bin/env python3
"""
Final geDIG test - comparing simple vs optimized geDIG with proper thresholds.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator
import random


def generate_simple_maze(size: int) -> np.ndarray:
    """Generate a simple maze using recursive backtracking."""
    maze = np.ones((size, size), dtype=int)
    
    def carve(x, y):
        maze[y, x] = 0
    
    carve(1, 1)
    
    def neighbors(cx, cy):
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
            stack.pop()
            continue
        
        nx, ny, dx, dy = random.choice(nbs)
        maze[y + dy // 2, x + dx // 2] = 0
        maze[ny, nx] = 0
        visited.add((nx, ny))
        stack.append((nx, ny))
    
    # Add loops
    loops_target = size // 2
    attempts = 0
    loops = 0
    
    while loops < loops_target and attempts < size * 10:
        attempts += 1
        x = random.randint(2, size - 3)
        y = random.randint(2, size - 3)
        
        if maze[y, x] == 1:
            open_cnt = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] 
                          if 0 <= y + dy < size and 0 <= x + dx < size 
                          and maze[y + dy, x + dx] == 0)
            if open_cnt >= 2:
                maze[y, x] = 0
                loops += 1
    
    maze[size - 2, size - 2] = 0
    
    return maze


def run_single_experiment(
    maze_size: int,
    seed: int,
    wiring_strategy: str,
    max_steps: int = 1000
) -> Dict:
    """Run a single maze experiment."""
    
    random.seed(seed)
    np.random.seed(seed)
    maze = generate_simple_maze(maze_size)
    
    start = (1, 1)
    goal = (maze_size - 2, maze_size - 2)
    
    # Create navigator
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=wiring_strategy,
        gedig_threshold=-0.15,  # Proper negative threshold
        backtrack_threshold=-0.3,
        simple_mode=False,
        backtrack_debounce=True
    )
    
    # Run navigation
    start_time = time.time()
    path = []
    backtrack_events = []
    
    for step in range(max_steps):
        action = nav.step()
        path.append(nav.current_pos)
        
        # Track backtrack events
        events = getattr(nav, 'event_log', [])
        for event in events:
            if event.get('type') == 'backtrack_trigger' and event not in backtrack_events:
                backtrack_events.append(event)
        
        if nav.current_pos == goal:
            break
    
    end_time = time.time()
    
    # Get graph stats
    graph_stats = nav.graph_manager.get_graph_statistics()
    
    # Analyze edge logs for gedig_optimized
    edge_analysis = {}
    if wiring_strategy.startswith('gedig') and hasattr(nav.graph_manager, 'edge_logs'):
        edge_logs = nav.graph_manager.edge_logs
        if edge_logs:
            gedig_values = [log['gedig'] for log in edge_logs]
            edge_analysis = {
                'edges_created': len(edge_logs),
                'min_gedig': min(gedig_values),
                'max_gedig': max(gedig_values),
                'avg_gedig': sum(gedig_values) / len(gedig_values)
            }
    
    result = {
        'maze_size': maze_size,
        'seed': seed,
        'wiring_strategy': wiring_strategy,
        'goal_reached': nav.current_pos == goal,
        'steps': len(path),
        'unique_positions': len(set(path)),
        'redundancy': len(path) / len(set(path)) if set(path) else 0,
        'time_seconds': end_time - start_time,
        'backtrack_count': len(backtrack_events),
        'graph_edges': graph_stats['num_edges'],
        'graph_density': graph_stats['density'],
        'edge_analysis': edge_analysis
    }
    
    return result


def _bfs_shortest_len(maze: np.ndarray, start: tuple[int,int], goal: tuple[int,int]) -> int | None:
    from collections import deque
    h, w = maze.shape
    dq = deque([(start, 0)])
    seen = {start}
    while dq:
        (x, y), d = dq.popleft()
        if (x, y) == goal:
            return d + 1
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0 and (nx,ny) not in seen:
                seen.add((nx,ny))
                dq.append(((nx,ny), d+1))
    return None


def run_final_comparison():
    """Run final comparison experiment."""
    
    results = []
    
    print("=" * 70)
    print("FINAL geDIG COMPARISON - Simple vs Optimized geDIG")
    print("=" * 70)
    
    maze_sizes = [15, 25]
    seeds = [42, 123, 456]
    
    for maze_size in maze_sizes:
        print(f"\n{maze_size}x{maze_size} mazes:")
        print("-" * 40)
        
        for seed in seeds:
            print(f"  Seed {seed}:")
            
            # Run simple baseline
            result_simple = run_single_experiment(
                maze_size=maze_size,
                seed=seed,
                wiring_strategy='simple',
                max_steps=maze_size * maze_size * 4
            )
            results.append(result_simple)
            print(f"    Simple:          {'✓' if result_simple['goal_reached'] else '✗'} "
                  f"{result_simple['steps']:>4} steps, {result_simple['graph_edges']:>3} edges")
            
            # Run geDIG (standard)
            result_gedig = run_single_experiment(
                maze_size=maze_size,
                seed=seed,
                wiring_strategy='gedig',
                max_steps=maze_size * maze_size * 4
            )
            results.append(result_gedig)
            
            edge_info = ""
            if result_gedig['edge_analysis']:
                ea = result_gedig['edge_analysis']
                edge_info = f" (geDIG: {ea['avg_gedig']:.3f})"
            
            print(f"    geDIG_optimized: {'✓' if result_gedig['goal_reached'] else '✗'} "
                  f"{result_gedig['steps']:>4} steps, {result_gedig['graph_edges']:>3} edges{edge_info}")
            
            # Show improvement
            if result_simple['goal_reached'] and result_gedig['goal_reached']:
                improvement = (result_simple['steps'] - result_gedig['steps']) / result_simple['steps'] * 100
                backtrack_diff = result_gedig['backtrack_count'] - result_simple['backtrack_count']
                print(f"    → Improvement: {improvement:+.1f}% steps, {backtrack_diff:+d} backtracks")
    
    # Summary statistics (with BFS shortest path ratio)
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = {}
    for maze_size in [15, 25]:
        # compute representative BFS on a fresh maze per seed and average
        bfs_values = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            m = generate_simple_maze(maze_size)
            sp = _bfs_shortest_len(m, (1,1), (maze_size-2, maze_size-2))
            if sp is not None:
                bfs_values.append(sp)
        bfs_mean = (sum(bfs_values)/len(bfs_values)) if bfs_values else None
        for strategy in ['simple', 'gedig_optimized']:
            strategy_results = [r for r in results 
                               if r['maze_size'] == maze_size 
                               and r['wiring_strategy'] == strategy]
            
            if not strategy_results:
                continue
            
            successful = [r for r in strategy_results if r['goal_reached']]
            
            if successful:
                avg_steps = sum(r['steps'] for r in successful) / len(successful)
                avg_edges = sum(r['graph_edges'] for r in strategy_results) / len(strategy_results)
                avg_backtracks = sum(r['backtrack_count'] for r in strategy_results) / len(strategy_results)
                success_rate = len(successful) / len(strategy_results) * 100
                
                print(f"{maze_size}x{maze_size} {strategy:15}: "
                      f"Success: {success_rate:>5.1f}%, "
                      f"Avg steps: {avg_steps:>6.1f}, "
                      f"Avg edges: {avg_edges:>5.1f}, "
                      f"Avg backtracks: {avg_backtracks:>4.1f}")
                if bfs_mean:
                    ratio = avg_steps / bfs_mean
                    print(f"  -> Shortest-path ratio vs BFS: {ratio:.2f}x (BFS≈{bfs_mean:.1f})")
                # collect summary
                summary.setdefault(str(maze_size), {})[strategy] = {
                    'success_rate': round(success_rate/100.0, 3),
                    'avg_steps': round(avg_steps, 2),
                    'avg_edges': round(avg_edges, 1),
                    'avg_backtracks': round(avg_backtracks, 2),
                    'bfs_mean': bfs_mean,
                    'ratio_vs_bfs': (round(avg_steps/bfs_mean, 3) if bfs_mean else None)
                }
    
    # Save results and summary under this experiment's results directory
    out_base = Path(__file__).resolve().parents[2] / 'results' / 'final_gedig_test'
    out_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(out_base / f'results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_base / f'summary_{timestamp}.json', 'w') as f:
        json.dump({'summary': summary, 'timestamp': timestamp}, f, indent=2)
    print(f"\nResults saved to: {out_base}/results_{timestamp}.json")
    print(f"Summary saved to: {out_base}/summary_{timestamp}.json")


if __name__ == '__main__':
    run_final_comparison()
