#!/usr/bin/env python3
"""
Run comparison experiments on complex mazes to show geDIG advantages
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import random
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from experiments.maze_layouts import (
    create_complex_maze, create_ultra_maze, create_perfect_maze,
    COMPLEX_DEFAULT_START, COMPLEX_DEFAULT_GOAL,
    ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL,
    PERFECT_DEFAULT_START, PERFECT_DEFAULT_GOAL
)
from navigation.maze_navigator import MazeNavigator
# Baseline functions inline to avoid import issues
def neighbors(maze, p):
    x, y = p
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0] and maze[ny,nx]==0:
            yield (nx,ny)

def run_random(maze, start, goal, max_steps, rng):
    path=[start]
    current=start
    visited=set([start])
    for step in range(1, max_steps+1):
        nbrs=list(neighbors(maze,current))
        if not nbrs:
            break
        current=rng.choice(nbrs)
        path.append(current)
        visited.add(current)
        if current==goal:
            break
    return path, current==goal

def run_dfs(maze, start, goal, max_steps, rng):
    stack=[start]
    parent={start: None}
    expanded_order=[]
    steps=0
    while stack and steps < max_steps:
        node=stack.pop()
        expanded_order.append(node)
        steps+=1
        if node==goal:
            break
        nbrs=list(neighbors(maze,node))
        rng.shuffle(nbrs)
        for n in nbrs:
            if n not in parent:
                parent[n]=node
                stack.append(n)
    if expanded_order and expanded_order[-1]==goal:
        path=[]
        cur=goal
        while cur is not None:
            path.append(cur)
            cur=parent[cur]
        path.reverse()
        success=True
    else:
        path=expanded_order
        success=(expanded_order and expanded_order[-1]==goal)
    return path, success

def _max_steps_for(maze, default_max):
    import os
    try:
        f = os.environ.get('MAZE_MAX_STEPS_FACTOR')
        if f is not None:
            return int((maze.shape[0] * maze.shape[1]) * float(f))
    except Exception:
        pass
    return default_max


def run_gedig_navigator(maze, start, goal, strategy='gedig', max_steps=2000):
    """Run geDIG-based navigator (proper settings for paper)."""
    # Allow env override for max steps factor
    max_steps = _max_steps_for(maze, max_steps)
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=strategy,
        gedig_threshold=-0.15,
        backtrack_threshold=-0.3,
        simple_mode=False,               # use full decision pipeline
        backtrack_debounce=True,
        wiring_top_k=3,                  # smaller candidate set for speed
        enable_diameter_metrics=False,   # trim instrumentation
        dense_metric_interval=25,
        snapshot_skip_idle=True,
        max_graph_snapshots=0,
        enable_flush=False,
        vector_index=None,
        ann_backend=None,
    )
    
    path = []
    start_time = time.perf_counter()
    
    for step in range(max_steps):
        action = nav.step()
        path.append(nav.current_pos)
        if nav.current_pos == goal:
            break
    
    elapsed = time.perf_counter() - start_time
    graph_stats = nav.graph_manager.get_graph_statistics()
    
    return {
        'success': nav.current_pos == goal,
        'steps': len(path),
        'unique_positions': len(set(path)),
        'redundancy': len(path) / max(1, len(set(path))),
        'time': elapsed,
        'edges': graph_stats['num_edges'],
        'nodes': graph_stats['num_nodes']
    }

def run_baseline(maze, start, goal, algorithm='random', max_steps=2000, seed=42):
    """Run baseline algorithm"""
    rng = random.Random(seed)
    
    start_time = time.perf_counter()
    if algorithm == 'random':
        path, success = run_random(maze, start, goal, max_steps, rng)
    elif algorithm == 'dfs':
        path, success = run_dfs(maze, start, goal, max_steps, rng)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    elapsed = time.perf_counter() - start_time
    
    return {
        'success': success,
        'steps': len(path),
        'unique_positions': len(set(path)),
        'redundancy': len(path) / max(1, len(set(path))),
        'time': elapsed,
        'edges': None,
        'nodes': None
    }

def main():
    """Run comprehensive comparison"""
    
    print("=" * 80)
    print("Complex Maze Navigation Comparison")
    print("=" * 80)
    
    # Maze configurations
    maze_configs = [
        {
            'name': 'Complex (25×25)',
            'maze_fn': create_complex_maze,
            'start': COMPLEX_DEFAULT_START,
            'goal': COMPLEX_DEFAULT_GOAL,
            'max_steps': 2000
        },
        {
            'name': 'Ultra (25×25)',
            'maze_fn': lambda: create_ultra_maze(seed=42),
            'start': ULTRA_DEFAULT_START,
            'goal': ULTRA_DEFAULT_GOAL,
            'max_steps': 3000
        },
        {
            'name': 'Perfect (25×25)',
            'maze_fn': lambda: create_perfect_maze(seed=42),
            'start': PERFECT_DEFAULT_START,
            'goal': PERFECT_DEFAULT_GOAL,
            'max_steps': 2000
        }
    ]
    
    # Algorithms to test
    algorithms = [
        ('Random Walk', 'random', None),
        ('DFS', 'dfs', None),
        ('Episodic Memory', None, 'simple'),
        ('geDIG', None, 'gedig')
    ]
    
    results = []
    
    for config in maze_configs:
        print(f"\n{config['name']} Maze")
        print("-" * 60)
        
        maze = config['maze_fn']()
        
        for algo_name, baseline_algo, nav_strategy in algorithms:
            print(f"  {algo_name:20}: ", end='', flush=True)
            
            # Run 3 trials for stability
            trial_results = []
            for seed in [42, 123, 456]:
                random.seed(seed)
                np.random.seed(seed)
                
                if baseline_algo:
                    result = run_baseline(
                        maze, config['start'], config['goal'],
                        algorithm=baseline_algo,
                        max_steps=config['max_steps'],
                        seed=seed
                    )
                else:
                    result = run_gedig_navigator(
                        maze, config['start'], config['goal'],
                        strategy=nav_strategy,
                        max_steps=config['max_steps']
                    )
                
                trial_results.append(result)
            
            # Average results
            avg_result = {
                'maze': config['name'],
                'algorithm': algo_name,
                'success_rate': sum(r['success'] for r in trial_results) / len(trial_results) * 100,
                'avg_steps': np.mean([r['steps'] for r in trial_results if r['success']]) if any(r['success'] for r in trial_results) else float('inf'),
                'avg_redundancy': np.mean([r['redundancy'] for r in trial_results]),
                'avg_time': np.mean([r['time'] for r in trial_results]),
                'avg_edges': np.mean([r['edges'] for r in trial_results if r['edges'] is not None]) if any(r['edges'] is not None for r in trial_results) else None
            }
            
            results.append(avg_result)
            
            # Print summary
            if avg_result['success_rate'] > 0:
                print(f"✓ {avg_result['success_rate']:.0f}% success, {avg_result['avg_steps']:.0f} steps, redundancy={avg_result['avg_redundancy']:.2f}")
            else:
                print(f"✗ 0% success")
    
    # Summary table
    print("\n" + "=" * 80)
    print("Summary Results")
    print("=" * 80)
    
    for maze_name in [c['name'] for c in maze_configs]:
        print(f"\n{maze_name}:")
        print(f"{'Algorithm':<20} {'Success':<10} {'Steps':<10} {'Redundancy':<12} {'Edges':<10}")
        print("-" * 70)
        
        maze_results = [r for r in results if r['maze'] == maze_name]
        for r in maze_results:
            steps_str = f"{r['avg_steps']:.0f}" if r['avg_steps'] != float('inf') else "N/A"
            edges_str = f"{r['avg_edges']:.0f}" if r['avg_edges'] is not None else "-"
            print(f"{r['algorithm']:<20} {r['success_rate']:>6.0f}% {steps_str:>10} {r['avg_redundancy']:>11.2f} {edges_str:>10}")
    
    # Save results under this experiment's results directory
    out_base = Path(__file__).resolve().parent / 'results' / 'complex_comparison'
    out_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = str(out_base / f'results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("Key Findings")
    print("=" * 80)
    
    for maze_name in [c['name'] for c in maze_configs]:
        maze_results = [r for r in results if r['maze'] == maze_name]
        
        random_result = next((r for r in maze_results if r['algorithm'] == 'Random Walk'), None)
        gedig_result = next((r for r in maze_results if r['algorithm'] == 'geDIG'), None)
        
        if random_result and gedig_result:
            if gedig_result['avg_steps'] != float('inf') and random_result['avg_steps'] != float('inf'):
                improvement = (random_result['avg_steps'] - gedig_result['avg_steps']) / random_result['avg_steps'] * 100
                print(f"{maze_name}: geDIG achieves {improvement:.1f}% step reduction vs Random Walk")
            elif gedig_result['success_rate'] > random_result['success_rate']:
                print(f"{maze_name}: geDIG succeeds where Random Walk fails ({gedig_result['success_rate']:.0f}% vs {random_result['success_rate']:.0f}%)")

if __name__ == '__main__':
    main()
