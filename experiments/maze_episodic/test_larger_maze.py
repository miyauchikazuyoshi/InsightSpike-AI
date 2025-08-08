#!/usr/bin/env python3
"""
Test Pure Episodic Navigator on Larger Mazes
"""

import numpy as np
import time
import os
from pure_episodic_navigator import PureEpisodicNavigator, create_complex_maze, visualize_maze_with_path

def test_multiple_sizes():
    """Test on progressively larger mazes"""
    print("="*70)
    print("PURE EPISODIC NAVIGATION - MULTI-SIZE TEST")
    print("No visit counts, only episodic memory + multi-hop evaluation")
    print("="*70)
    
    sizes = [20, 25, 30, 40]
    results = {}
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Testing {size}×{size} maze")
        print('='*60)
        
        # Create maze
        maze = create_complex_maze(size, seed=42)
        
        # Statistics
        walkable = np.sum(maze == 0)
        optimal_estimate = 2 * (size - 2)
        
        print(f"Maze statistics:")
        print(f"  Walkable cells: {walkable}")
        print(f"  Density: {walkable/(size*size)*100:.1f}%")
        print(f"  Optimal estimate: ~{optimal_estimate} steps")
        
        # Create navigator
        nav = PureEpisodicNavigator(maze, message_depth=3)
        
        # Set max steps based on size
        max_steps = min(size * 100, 3000)
        
        # Navigate
        start_time = time.time()
        result = nav.navigate(max_steps=max_steps)
        
        # Store result
        results[size] = result
        
        # Analysis
        if result['success']:
            efficiency = result['steps'] / optimal_estimate
            print(f"\n✓ SUCCESS in {result['steps']} steps!")
            print(f"  Efficiency: {efficiency:.2f}x optimal")
            print(f"  Wall hit rate: {result['wall_hits']/result['steps']*100:.1f}%")
            
            # Save visualization
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path,
                f'visualizations/pure_episodic_{size}x{size}_result.png'
            )
        else:
            print(f"\n✗ Failed after {result['steps']} steps")
            print(f"  Explored: {len(nav.visited)} cells")
            print(f"  Coverage: {len(nav.visited)/walkable*100:.1f}%")
        
        # Stop if failed
        if not result['success'] and size >= 25:
            print("\nStopping test as maze is too large for current implementation.")
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Size':<10} {'Success':<10} {'Steps':<10} {'Efficiency':<12} {'Time (s)':<10}")
    print("-" * 60)
    
    for size, result in results.items():
        success = "✓" if result['success'] else "✗"
        steps = result['steps']
        efficiency = f"{steps/(2*(size-2)):.2f}x" if result['success'] else "N/A"
        time_taken = f"{result['time']:.2f}"
        
        print(f"{size}×{size:<5} {success:<10} {steps:<10} {efficiency:<12} {time_taken:<10}")
    
    # Hop distribution analysis
    print("\nHop Selection Patterns:")
    for size, result in results.items():
        if result['success']:
            print(f"\n{size}×{size}:")
            total = sum(result['hop_selections'].values())
            for hop, count in result['hop_selections'].items():
                print(f"  {hop}: {count/total*100:.1f}%")
    
    return results


def analyze_scaling():
    """Analyze how performance scales with maze size"""
    results = test_multiple_sizes()
    
    # Extract successful results
    successful = {size: r for size, r in results.items() if r['success']}
    
    if len(successful) >= 2:
        print("\n" + "="*70)
        print("SCALING ANALYSIS")
        print("="*70)
        
        sizes = sorted(successful.keys())
        steps = [successful[s]['steps'] for s in sizes]
        
        # Simple linear regression
        if len(sizes) >= 2:
            # Calculate slope
            n = len(sizes)
            xy_sum = sum(s * st for s, st in zip(sizes, steps))
            x_sum = sum(sizes)
            y_sum = sum(steps)
            x2_sum = sum(s**2 for s in sizes)
            
            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum**2)
            
            print(f"\nScaling factor: ~{slope:.1f} steps per unit maze size")
            print(f"Predicted steps for 50×50: ~{int(slope * 50)} steps")


if __name__ == "__main__":
    analyze_scaling()