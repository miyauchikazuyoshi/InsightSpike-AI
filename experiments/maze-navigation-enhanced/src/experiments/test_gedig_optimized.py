#!/usr/bin/env python3
"""
Test optimized geDIG implementation in maze navigation.
Compares three strategies:
1. simple - baseline sequential wiring
2. gedig - original geDIG with graph copying (slow)
3. gedig_optimized - optimized geDIG without graph copying
"""

import os
import sys
import json
import time
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
    
    # Start carving from (1, 1)
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
    
    # Add some loops for complexity
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
    
    # Ensure goal is accessible
    maze[size - 2, size - 2] = 0
    
    return maze


def run_single_experiment(
    maze_size: int,
    seed: int,
    wiring_strategy: str,
    gedig_threshold: float = -0.1,
    backtrack_threshold: float = -0.3,
    max_steps: int = 1000,
    timeout_seconds: float = 120.0
) -> Dict:
    """Run a single maze experiment with specified parameters."""
    
    # Create maze
    random.seed(seed)
    np.random.seed(seed)
    maze = generate_simple_maze(maze_size)
    
    # Find valid start and goal
    start = (1, 1)
    goal = (maze_size - 2, maze_size - 2)
    
    # Ensure start and goal are valid
    if maze[start[0], start[1]] == 1:
        for i in range(1, maze_size - 1):
            for j in range(1, maze_size - 1):
                if maze[i, j] == 0:
                    start = (i, j)
                    break
            if maze[start[0], start[1]] == 0:
                break
    
    if maze[goal[0], goal[1]] == 1:
        for i in range(maze_size - 2, 0, -1):
            for j in range(maze_size - 2, 0, -1):
                if maze[i, j] == 0:
                    goal = (i, j)
                    break
            if maze[goal[0], goal[1]] == 0:
                break
    
    print(f"      {wiring_strategy}: ", end='', flush=True)
    
    # Create navigator
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=wiring_strategy,
        gedig_threshold=gedig_threshold,
        backtrack_threshold=backtrack_threshold,
        simple_mode=True,
        backtrack_debounce=True
    )
    
    # Run navigation with timeout
    start_time = time.time()
    path = []
    
    for step in range(max_steps):
        action = nav.step()
        path.append(nav.current_pos)
        
        if nav.current_pos == goal:
            break
        
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            print(f"TIMEOUT after {step} steps", end='')
            break
    
    end_time = time.time()
    
    # Collect graph statistics
    graph_stats = nav.graph_manager.get_graph_statistics()
    
    # Collect results
    events = getattr(nav, 'event_log', [])
    gedig_history = getattr(nav, 'gedig_history', [])
    
    result = {
        'maze_size': maze_size,
        'seed': seed,
        'wiring_strategy': wiring_strategy,
        'gedig_threshold': gedig_threshold,
        'goal_reached': nav.current_pos == goal,
        'steps': len(path),
        'unique_positions': len(set(path)),
        'redundancy': len(path) / len(set(path)) if set(path) else 0,
        'time_seconds': end_time - start_time,
        'backtrack_count': len([e for e in events if e.get('type') == 'backtrack_trigger']),
        'graph_nodes': graph_stats['num_nodes'],
        'graph_edges': graph_stats['num_edges'],
        'graph_density': graph_stats['density'],
        'mean_gedig': np.mean(gedig_history) if gedig_history else 0,
        'min_gedig': np.min(gedig_history) if gedig_history else 0
    }
    
    status = '✓' if result['goal_reached'] else '✗'
    print(f"{status} {result['steps']} steps, {result['graph_edges']} edges, {result['time_seconds']:.1f}s")
    
    return result


def run_optimization_comparison(
    maze_sizes: List[int] = [15, 25],
    seeds: List[int] = [42, 123, 456],
    output_dir: str = 'results/optimized_gedig_test'
) -> None:
    """Compare simple vs gedig vs gedig_optimized strategies."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("=" * 70)
    print("OPTIMIZED geDIG EXPERIMENT - Comparing Three Strategies")
    print("=" * 70)
    
    for maze_size in maze_sizes:
        print(f"\nTesting {maze_size}x{maze_size} mazes...")
        
        # Adjust max steps based on maze size
        max_steps = maze_size * maze_size * 4
        
        for seed in seeds:
            print(f"  Seed {seed}:")
            
            # Test all three strategies
            for strategy in ['simple', 'gedig', 'gedig_optimized']:
                result = run_single_experiment(
                    maze_size=maze_size,
                    seed=seed,
                    wiring_strategy=strategy,
                    max_steps=max_steps,
                    timeout_seconds=60.0 if strategy == 'gedig' else 120.0
                )
                results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate summary
    generate_summary(results, output_dir, timestamp)


def generate_summary(results: List[Dict], output_dir: str, timestamp: str) -> None:
    """Generate summary statistics."""
    
    summary = []
    
    for maze_size in sorted(set(r['maze_size'] for r in results)):
        for strategy in ['simple', 'gedig', 'gedig_optimized']:
            strategy_results = [r for r in results 
                               if r['maze_size'] == maze_size 
                               and r['wiring_strategy'] == strategy]
            
            if not strategy_results:
                continue
            
            success_rate = sum(1 for r in strategy_results if r['goal_reached']) / len(strategy_results)
            successful_runs = [r for r in strategy_results if r['goal_reached']]
            
            stats = {
                'maze_size': maze_size,
                'strategy': strategy,
                'total_runs': len(strategy_results),
                'success_rate': success_rate,
                'avg_steps': np.mean([r['steps'] for r in successful_runs]) if successful_runs else 0,
                'avg_time': np.mean([r['time_seconds'] for r in strategy_results]),
                'avg_edges': np.mean([r['graph_edges'] for r in strategy_results]),
                'avg_density': np.mean([r['graph_density'] for r in strategy_results]),
                'avg_redundancy': np.mean([r['redundancy'] for r in successful_runs]) if successful_runs else 0
            }
            
            summary.append(stats)
    
    # Save summary
    summary_file = os.path.join(output_dir, f'summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY RESULTS")
    print("=" * 90)
    print(f"{'Maze':<8} {'Strategy':<15} {'Success':<10} {'Avg Steps':<12} {'Avg Time':<12} {'Avg Edges':<12} {'Density':<10}")
    print("-" * 90)
    
    for s in summary:
        print(f"{s['maze_size']}x{s['maze_size']:<5} {s['strategy']:<15} "
              f"{s['success_rate']*100:>6.1f}% "
              f"{s['avg_steps']:>11.1f} "
              f"{s['avg_time']:>10.1f}s "
              f"{s['avg_edges']:>11.1f} "
              f"{s['avg_density']:>9.4f}")
    
    # Calculate improvements
    print("\n" + "=" * 90)
    print("PERFORMANCE COMPARISON")
    print("=" * 90)
    
    for maze_size in sorted(set(s['maze_size'] for s in summary)):
        simple = next((s for s in summary if s['maze_size'] == maze_size and s['strategy'] == 'simple'), None)
        gedig = next((s for s in summary if s['maze_size'] == maze_size and s['strategy'] == 'gedig'), None)
        optimized = next((s for s in summary if s['maze_size'] == maze_size and s['strategy'] == 'gedig_optimized'), None)
        
        print(f"\n{maze_size}x{maze_size} maze:")
        
        if simple and gedig:
            print(f"  geDIG vs simple:")
            print(f"    - Time: {gedig['avg_time']/simple['avg_time']:.1f}x slower")
            print(f"    - Edges: {gedig['avg_edges']:.0f} vs {simple['avg_edges']:.0f}")
            if simple['avg_steps'] > 0:
                step_improvement = (simple['avg_steps'] - gedig['avg_steps']) / simple['avg_steps'] * 100
                print(f"    - Step efficiency: {step_improvement:+.1f}%")
        
        if simple and optimized:
            print(f"  geDIG_optimized vs simple:")
            print(f"    - Time: {optimized['avg_time']/simple['avg_time']:.1f}x")
            print(f"    - Edges: {optimized['avg_edges']:.0f} vs {simple['avg_edges']:.0f}")
            if simple['avg_steps'] > 0:
                step_improvement = (simple['avg_steps'] - optimized['avg_steps']) / simple['avg_steps'] * 100
                print(f"    - Step efficiency: {step_improvement:+.1f}%")
        
        if gedig and optimized:
            print(f"  Optimization speedup:")
            print(f"    - {gedig['avg_time']/optimized['avg_time']:.1f}x faster")
            print(f"    - Same effectiveness: {optimized['success_rate']*100:.0f}% success")


if __name__ == '__main__':
    # Run optimization comparison
    run_optimization_comparison(
        maze_sizes=[15],  # Just small maze to avoid timeout
        seeds=[42, 123],  # Two seeds
        output_dir='results/optimized_gedig_test'
    )