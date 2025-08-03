#!/usr/bin/env python3
"""
Multi-hop geDIG Navigator Comparison - 3 Runs
=============================================

Tests 1-hop, 2-hop, and 3-hop evaluation multiple times to verify stability.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import json
from multihop_gedig_comparison import MultiHopGeDIGNavigator

def run_multiple_experiments(num_runs=3):
    """Run multiple experiments to check stability"""
    
    all_results = []
    
    for run in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run + 1} / {num_runs}")
        print(f"{'='*70}")
        
        # Generate maze with different seed each time
        np.random.seed(42 + run)  # Different seed for each run
        size = 50
        maze = np.ones((size, size), dtype=int)
        
        # Create maze with recursive backtracker
        def carve_passages(cx, cy):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            np.random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size and maze[ny, nx] == 1:
                    maze[cy + dy // 2, cx + dx // 2] = 0
                    maze[ny, nx] = 0
                    carve_passages(nx, ny)
        
        maze[1, 1] = 0
        carve_passages(1, 1)
        maze[size-2, size-2] = 0
        
        # Add some loops
        for _ in range(size):
            x = np.random.randint(2, size-2)
            y = np.random.randint(2, size-2)
            if maze[y, x] == 1:
                neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                              if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
                if neighbors >= 2:
                    maze[y, x] = 0
        
        goal = (size-2, size-2)
        run_results = {}
        
        # Test different hop counts
        for hop_count in [1, 2, 3]:
            print(f"\n--- {hop_count}-hop (Run {run + 1}) ---")
            navigator = MultiHopGeDIGNavigator(maze, hop_count=hop_count)
            result = navigator.navigate(goal, max_steps=5000)
            run_results[f"{hop_count}-hop"] = result
        
        all_results.append(run_results)
    
    # Analyze results
    print("\n" + "="*70)
    print("SUMMARY OF ALL RUNS")
    print("="*70)
    
    # Calculate statistics
    stats = {}
    for method in ["1-hop", "2-hop", "3-hop"]:
        method_results = [run[method] for run in all_results]
        
        stats[method] = {
            'success_rate': sum(r['success'] for r in method_results) / num_runs * 100,
            'avg_steps': np.mean([r['steps'] for r in method_results]),
            'std_steps': np.std([r['steps'] for r in method_results]),
            'avg_coverage': np.mean([r['coverage'] for r in method_results]),
            'std_coverage': np.std([r['coverage'] for r in method_results]),
            'avg_efficiency': np.mean([r['efficiency'] for r in method_results]),
            'std_efficiency': np.std([r['efficiency'] for r in method_results])
        }
    
    # Display detailed results
    print("\n| Run | Method | Success | Steps | Coverage | Efficiency |")
    print("|-----|--------|---------|-------|----------|------------|")
    
    for i, run_results in enumerate(all_results):
        for method, result in run_results.items():
            print(f"| {i+1:3} | {method:6} | {str(result['success']):7} | {result['steps']:5} | "
                  f"{result['coverage']:7.1f}% | {result['efficiency']:9.1f}% |")
        if i < len(all_results) - 1:
            print("|-----|--------|---------|-------|----------|------------|")
    
    # Display statistics
    print("\n" + "="*70)
    print("STATISTICS ACROSS ALL RUNS")
    print("="*70)
    
    print("\n| Method | Success% | Avg Steps | Std Steps | Avg Coverage% | Avg Efficiency% |")
    print("|--------|----------|-----------|-----------|---------------|-----------------|")
    
    for method, stat in stats.items():
        print(f"| {method:6} | {stat['success_rate']:7.0f}% | {stat['avg_steps']:9.1f} | "
              f"{stat['std_steps']:9.1f} | {stat['avg_coverage']:13.1f} | "
              f"{stat['avg_efficiency']:15.1f} |")
    
    # Create box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Steps box plot
    steps_data = [[run[method]['steps'] for run in all_results] for method in ["1-hop", "2-hop", "3-hop"]]
    axes[0].boxplot(steps_data, labels=["1-hop", "2-hop", "3-hop"])
    axes[0].set_ylabel('Steps')
    axes[0].set_title('Steps to Goal Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Coverage box plot
    coverage_data = [[run[method]['coverage'] for run in all_results] for method in ["1-hop", "2-hop", "3-hop"]]
    axes[1].boxplot(coverage_data, labels=["1-hop", "2-hop", "3-hop"])
    axes[1].set_ylabel('Coverage (%)')
    axes[1].set_title('Coverage Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Efficiency box plot
    efficiency_data = [[run[method]['efficiency'] for run in all_results] for method in ["1-hop", "2-hop", "3-hop"]]
    axes[2].boxplot(efficiency_data, labels=["1-hop", "2-hop", "3-hop"])
    axes[2].set_ylabel('Efficiency (%)')
    axes[2].set_title('Efficiency Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multihop_stability_analysis.png', dpi=150)
    plt.close()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'multihop_3runs_results_{timestamp}.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_runs': num_runs,
            'statistics': stats,
            'all_runs': [[{k: v for k, v in result.items() if k != 'coverage_over_time'} 
                         for method, result in run.items()] 
                        for run in all_results]
        }, f, indent=2)
    
    print(f"\nResults saved to multihop_3runs_results_{timestamp}.json")
    print("Box plots saved to multihop_stability_analysis.png")
    
    return all_results, stats


if __name__ == "__main__":
    all_results, stats = run_multiple_experiments(num_runs=3)