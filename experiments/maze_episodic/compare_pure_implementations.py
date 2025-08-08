#!/usr/bin/env python3
"""
Compare original PureEpisodicNavigator with Integrated Index version
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_navigator import PureEpisodicNavigator
from pure_episodic_integrated import PureEpisodicIntegrated
from insightspike.environments.proper_maze_generator import ProperMazeGenerator


def compare_navigators(maze_size=(15, 15), max_steps=1000):
    """Compare original and integrated implementations"""
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=maze_size)
    
    print(f"\n{'='*60}")
    print(f"Comparing on {maze_size[0]}x{maze_size[1]} maze")
    print(f"{'='*60}")
    
    results = {}
    
    # Test original implementation
    print("\n1. Testing Original PureEpisodicNavigator (O(n²))...")
    navigator_orig = PureEpisodicNavigator(maze)
    
    orig_search_times = []
    start_total = time.time()
    
    for step in range(max_steps):
        if navigator_orig.position == navigator_orig.goal:
            results['original'] = {
                'success': True,
                'steps': step,
                'episodes': len(navigator_orig.episodes),
                'total_time': time.time() - start_total,
                'search_times': orig_search_times
            }
            print(f"  Success in {step} steps!")
            break
            
        # Measure search time
        start_search = time.time()
        action = navigator_orig.decide_action()
        search_time = (time.time() - start_search) * 1000
        orig_search_times.append(search_time)
        
        if action:
            success = navigator_orig.move(action)
            if success:
                navigator_orig.add_episode(navigator_orig.position, action, reward=0)
            
        if step % 100 == 0:
            print(f"  Step {step}: episodes={len(navigator_orig.episodes)}, "
                  f"avg_search={np.mean(orig_search_times[-10:]):.2f}ms")
    else:
        results['original'] = {
            'success': False,
            'steps': max_steps,
            'episodes': len(navigator_orig.episodes),
            'total_time': time.time() - start_total,
            'search_times': orig_search_times
        }
        print(f"  Failed after {max_steps} steps")
    
    # Test integrated implementation
    print("\n2. Testing Integrated Index Navigator (O(1))...")
    navigator_new = PureEpisodicIntegrated(maze)
    
    result_new = navigator_new.navigate(max_steps=max_steps)
    results['integrated'] = result_new
    
    if result_new['success']:
        print(f"  Success in {result_new['steps']} steps!")
    else:
        print(f"  Failed after {max_steps} steps")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS:")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Original':<20} {'Integrated':<20} {'Improvement':<20}")
    print("-"*85)
    
    # Success
    orig_success = results['original']['success']
    new_success = results['integrated']['success']
    print(f"{'Success':<25} {str(orig_success):<20} {str(new_success):<20} {'-':<20}")
    
    # Steps
    orig_steps = results['original']['steps']
    new_steps = results['integrated']['steps']
    print(f"{'Steps to goal':<25} {orig_steps:<20} {new_steps:<20} {'-':<20}")
    
    # Episodes
    orig_episodes = results['original']['episodes']
    new_episodes = results['integrated']['total_episodes']
    print(f"{'Total episodes':<25} {orig_episodes:<20} {new_episodes:<20} {'-':<20}")
    
    # Average search time
    orig_avg_search = np.mean(results['original']['search_times']) if results['original']['search_times'] else 0
    new_avg_search = results['integrated']['avg_search_time']
    speedup = orig_avg_search / new_avg_search if new_avg_search > 0 else 0
    print(f"{'Avg search time (ms)':<25} {orig_avg_search:<20.2f} {new_avg_search:<20.2f} {f'{speedup:.1f}x faster':<20}")
    
    # Total time
    orig_total = results['original']['total_time']
    new_total = results['integrated']['total_time']
    total_speedup = orig_total / new_total if new_total > 0 else 0
    print(f"{'Total time (s)':<25} {orig_total:<20.2f} {new_total:<20.2f} {f'{total_speedup:.1f}x faster':<20}")
    
    # Plot search time comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['original']['search_times'][:200], label='Original (O(n²))', alpha=0.7)
    plt.plot(results['integrated']['search_times'][:200], label='Integrated (O(1))', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Cumulative search time
    orig_cumsum = np.cumsum(results['original']['search_times'])
    new_cumsum = np.cumsum(results['integrated']['search_times'])
    plt.plot(orig_cumsum[:200], label='Original (O(n²))', alpha=0.7)
    plt.plot(new_cumsum[:200], label='Integrated (O(1))', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Search Time (ms)')
    plt.title('Cumulative Search Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{maze_size[0]}x{maze_size[1]}.png', dpi=150)
    print(f"\nSaved comparison plot to comparison_{maze_size[0]}x{maze_size[1]}.png")
    
    return results


def test_scaling():
    """Test how performance scales with maze size"""
    sizes = [(10, 10), (15, 15), (20, 20)]
    
    scaling_results = []
    
    for size in sizes:
        print(f"\n\n{'#'*60}")
        print(f"Testing {size[0]}x{size[1]} maze")
        print(f"{'#'*60}")
        
        try:
            results = compare_navigators(maze_size=size, max_steps=500)
            
            scaling_results.append({
                'size': size[0] * size[1],
                'original_avg_search': np.mean(results['original']['search_times']),
                'integrated_avg_search': results['integrated']['avg_search_time'],
                'original_total': results['original']['total_time'],
                'integrated_total': results['integrated']['total_time']
            })
        except Exception as e:
            print(f"Error on {size}: {e}")
            continue
    
    # Plot scaling analysis
    if scaling_results:
        plt.figure(figsize=(10, 6))
        
        sizes = [r['size'] for r in scaling_results]
        orig_times = [r['original_avg_search'] for r in scaling_results]
        new_times = [r['integrated_avg_search'] for r in scaling_results]
        
        plt.plot(sizes, orig_times, 'ro-', label='Original (O(n²))', markersize=8)
        plt.plot(sizes, new_times, 'bo-', label='Integrated (O(1))', markersize=8)
        
        plt.xlabel('Maze Size (total cells)')
        plt.ylabel('Average Search Time (ms)')
        plt.title('Search Time Scaling Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add theoretical O(n²) curve
        n = np.array(sizes)
        theoretical_n2 = orig_times[0] * (n / n[0])**2
        plt.plot(sizes, theoretical_n2, 'r--', alpha=0.5, label='Theoretical O(n²)')
        
        plt.legend()
        plt.savefig('scaling_analysis.png', dpi=150)
        print("\nSaved scaling analysis to scaling_analysis.png")


if __name__ == "__main__":
    # First do a single comparison
    compare_navigators(maze_size=(15, 15))
    
    # Then test scaling
    # test_scaling()  # Uncomment for full scaling test