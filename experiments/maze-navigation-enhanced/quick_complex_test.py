#!/usr/bin/env python3
"""
Quick complex maze test to show geDIG advantages
"""

import sys
import os
import numpy as np
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from experiments.maze_layouts import create_ultra_maze, ULTRA_DEFAULT_START, ULTRA_DEFAULT_GOAL
from navigation.maze_navigator import MazeNavigator

def run_quick_comparison():
    """Quick comparison on ultra-complex maze"""
    
    print("=" * 60)
    print("Ultra Complex Maze (25×25) - Quick Test")
    print("=" * 60)
    
    # Create ultra-complex maze
    maze = create_ultra_maze(seed=42)
    start = ULTRA_DEFAULT_START
    goal = ULTRA_DEFAULT_GOAL
    
    print(f"Start: {start}, Goal: {goal}")
    print(f"Maze complexity: {np.sum(maze == 0)} open cells out of 625")
    
    results = {}
    
    # Test each strategy
    for strategy_name, strategy in [('Simple', 'simple'), ('geDIG', 'gedig')]:
        print(f"\n{strategy_name} Strategy:")
        print("-" * 40)
        
        random.seed(42)
        np.random.seed(42)
        
        nav = MazeNavigator(
            maze=maze,
            start_pos=start,
            goal_pos=goal,
            wiring_strategy=strategy,
            gedig_threshold=-0.08,
            backtrack_threshold=-0.2,
            simple_mode=True
        )
        
        path = []
        start_time = time.perf_counter()
        max_steps = 1000
        
        for step in range(max_steps):
            action = nav.step()
            path.append(nav.current_pos)
            
            if step % 100 == 0:
                print(f"  Step {step}: at {nav.current_pos}")
            
            if nav.current_pos == goal:
                print(f"  ✓ Goal reached in {step+1} steps!")
                break
        else:
            print(f"  ✗ Failed to reach goal in {max_steps} steps")
        
        elapsed = time.perf_counter() - start_time
        graph_stats = nav.graph_manager.get_graph_statistics()
        
        results[strategy_name] = {
            'success': nav.current_pos == goal,
            'steps': len(path),
            'unique_positions': len(set(path)),
            'redundancy': len(path) / max(1, len(set(path))),
            'time': elapsed,
            'edges': graph_stats['num_edges'],
            'nodes': graph_stats['num_nodes']
        }
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Graph: {graph_stats['num_nodes']} nodes, {graph_stats['num_edges']} edges")
        print(f"  Redundancy: {results[strategy_name]['redundancy']:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if results['Simple']['success'] and results['geDIG']['success']:
        step_improvement = (results['Simple']['steps'] - results['geDIG']['steps']) / results['Simple']['steps'] * 100
        edge_reduction = (results['Simple']['edges'] - results['geDIG']['edges']) / results['Simple']['edges'] * 100
        
        print(f"Step reduction: {step_improvement:.1f}%")
        print(f"Edge reduction: {edge_reduction:.1f}%")
    elif results['geDIG']['success'] and not results['Simple']['success']:
        print("geDIG succeeds where Simple fails!")
    else:
        print("Both strategies struggled with this maze")

if __name__ == '__main__':
    run_quick_comparison()