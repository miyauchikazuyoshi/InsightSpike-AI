#!/usr/bin/env python3
"""
Real geDIG experiment - Testing actual geDIG wiring strategy
This experiment properly tests geDIG by using wiring_strategy='gedig'
instead of 'simple' which was used in previous experiments.
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add parent directory to path
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
    gedig_threshold: float = -0.05,  # Much more permissive threshold
    backtrack_threshold: float = -0.3,
    max_steps: int = 1000,
    timeout_seconds: float = 30.0
) -> Dict:
    """Run a single maze experiment with specified parameters."""
    
    # Create maze using simple maze generation
    random.seed(seed)
    np.random.seed(seed)
    maze = generate_simple_maze(maze_size)
    
    # Find valid start and goal
    start = (1, 1)
    goal = (maze_size - 2, maze_size - 2)
    
    # Ensure start and goal are valid
    if maze[start[0], start[1]] == 1:  # Wall
        # Find nearest valid position
        for i in range(1, maze_size - 1):
            for j in range(1, maze_size - 1):
                if maze[i, j] == 0:
                    start = (i, j)
                    break
            if maze[start[0], start[1]] == 0:
                break
    
    if maze[goal[0], goal[1]] == 1:  # Wall
        # Find nearest valid position
        for i in range(maze_size - 2, 0, -1):
            for j in range(maze_size - 2, 0, -1):
                if maze[i, j] == 0:
                    goal = (i, j)
                    break
            if maze[goal[0], goal[1]] == 0:
                break
    
    # Create navigator with specified wiring strategy
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=wiring_strategy,  # Key parameter!
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
            print(f"      Timeout after {step} steps")
            break
    
    end_time = time.time()
    
    # Collect results
    events = getattr(nav, 'event_log', [])
    gedig_history = getattr(nav, 'gedig_history', [])
    
    result = {
        'maze_size': maze_size,
        'seed': seed,
        'wiring_strategy': wiring_strategy,
        'gedig_threshold': gedig_threshold,
        'backtrack_threshold': backtrack_threshold,
        'goal_reached': nav.current_pos == goal,
        'steps': len(path),
        'unique_positions': len(set(path)),
        'redundancy': len(path) / len(set(path)) if set(path) else 0,
        'time_seconds': end_time - start_time,
        'backtrack_count': len([e for e in events if e.get('type') == 'backtrack_trigger']),
        'backtrack_plans': getattr(nav, '_backtrack_plan_count', 0),
        'mean_gedig': np.mean(gedig_history) if gedig_history else 0,
        'min_gedig': np.min(gedig_history) if gedig_history else 0,
        'gedig_below_threshold': sum(1 for g in gedig_history if g < gedig_threshold) if gedig_history else 0,
        'path': path[:100]  # Store first 100 steps for analysis
    }
    
    return result


def run_comparison_experiment(
    maze_sizes: List[int] = [15, 25, 50],
    seeds: List[int] = [42, 123, 456, 789, 101],
    output_dir: str = 'results/real_gedig_test'
) -> None:
    """Compare simple vs gedig wiring strategies."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("=" * 60)
    print("REAL geDIG EXPERIMENT - Comparing Wiring Strategies")
    print("=" * 60)
    
    for maze_size in maze_sizes:
        print(f"\nTesting {maze_size}x{maze_size} mazes...")
        
        # Adjust max steps based on maze size
        max_steps = maze_size * maze_size * 2
        
        for seed in seeds:
            print(f"  Seed {seed}:")
            
            # Test with simple strategy (baseline)
            print("    Running 'simple' strategy...", end='')
            result_simple = run_single_experiment(
                maze_size=maze_size,
                seed=seed,
                wiring_strategy='simple',
                max_steps=max_steps,
                timeout_seconds=30.0  # 30 second timeout per maze
            )
            results.append(result_simple)
            print(f" {'✓' if result_simple['goal_reached'] else '✗'} ({result_simple['steps']} steps)")
            
            # Test with gedig strategy (our method)
            print("    Running 'gedig' strategy...", end='')
            result_gedig = run_single_experiment(
                maze_size=maze_size,
                seed=seed,
                wiring_strategy='gedig',
                max_steps=max_steps,
                timeout_seconds=30.0  # 30 second timeout per maze
            )
            results.append(result_gedig)
            print(f" {'✓' if result_gedig['goal_reached'] else '✗'} ({result_gedig['steps']} steps)")
            
            # Show improvement
            if result_simple['goal_reached'] and result_gedig['goal_reached']:
                improvement = (result_simple['steps'] - result_gedig['steps']) / result_simple['steps'] * 100
                print(f"    → geDIG improvement: {improvement:.1f}%")
    
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
        for strategy in ['simple', 'gedig']:
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
                'avg_redundancy': np.mean([r['redundancy'] for r in successful_runs]) if successful_runs else 0,
                'avg_backtrack_count': np.mean([r['backtrack_count'] for r in strategy_results]),
                'avg_backtrack_plans': np.mean([r['backtrack_plans'] for r in strategy_results]),
                'avg_gedig_below_threshold': np.mean([r['gedig_below_threshold'] for r in strategy_results])
            }
            
            summary.append(stats)
    
    # Save summary
    summary_file = os.path.join(output_dir, f'summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Maze':<8} {'Strategy':<10} {'Success':<10} {'Avg Steps':<12} {'Redundancy':<12} {'Backtracks':<12}")
    print("-" * 80)
    
    for s in summary:
        print(f"{s['maze_size']}x{s['maze_size']:<5} {s['strategy']:<10} "
              f"{s['success_rate']*100:>6.1f}% "
              f"{s['avg_steps']:>11.1f} "
              f"{s['avg_redundancy']:>11.2f} "
              f"{s['avg_backtrack_plans']:>11.1f}")
    
    # Calculate overall improvement
    print("\n" + "=" * 80)
    print("GEDIG vs SIMPLE COMPARISON")
    print("=" * 80)
    
    for maze_size in sorted(set(s['maze_size'] for s in summary)):
        simple_stats = next((s for s in summary if s['maze_size'] == maze_size and s['strategy'] == 'simple'), None)
        gedig_stats = next((s for s in summary if s['maze_size'] == maze_size and s['strategy'] == 'gedig'), None)
        
        if simple_stats and gedig_stats and simple_stats['avg_steps'] > 0:
            step_improvement = (simple_stats['avg_steps'] - gedig_stats['avg_steps']) / simple_stats['avg_steps'] * 100
            success_diff = (gedig_stats['success_rate'] - simple_stats['success_rate']) * 100
            
            print(f"{maze_size}x{maze_size} maze:")
            print(f"  Success rate: simple={simple_stats['success_rate']*100:.1f}%, gedig={gedig_stats['success_rate']*100:.1f}% "
                  f"(+{success_diff:+.1f}%)")
            print(f"  Step efficiency: {step_improvement:+.1f}% improvement")
            print(f"  Backtrack plans: simple={simple_stats['avg_backtrack_plans']:.1f}, "
                  f"gedig={gedig_stats['avg_backtrack_plans']:.1f}")


if __name__ == '__main__':
    # Run the real geDIG experiment - minimal test first
    run_comparison_experiment(
        maze_sizes=[15],  # Just 15x15 
        seeds=[42, 123],  # Two seeds only
        output_dir='results/real_gedig_test'
    )