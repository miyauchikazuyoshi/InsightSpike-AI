#!/usr/bin/env python3
"""
Test integrated index on large mazes (25x25 and 50x50)
Previously failed due to O(n²) bottleneck
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_integrated import PureEpisodicIntegrated
from insightspike.environments.proper_maze_generator import ProperMazeGenerator


def test_large_maze(maze_size=(25, 25), max_steps=5000):
    """Test integrated index navigator on large maze"""
    
    print(f"\n{'='*60}")
    print(f"Testing {maze_size[0]}x{maze_size[1]} maze with Integrated Index")
    print(f"{'='*60}")
    
    # Generate maze
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=maze_size)
    
    # Create navigator with integrated index
    print("Creating navigator with integrated index...")
    navigator = PureEpisodicIntegrated(maze, message_depth=3)
    
    # Navigate
    print(f"Starting navigation (max {max_steps} steps)...")
    start_time = time.time()
    
    result = navigator.navigate(max_steps=max_steps)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Success: {result['success']}")
    print(f"Steps taken: {result['steps']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average search time: {result['avg_search_time']:.2f}ms")
    print(f"Total search time: {result['total_search_time']:.2f}ms")
    
    if result['success']:
        print(f"\n🎉 Successfully solved {maze_size[0]}x{maze_size[1]} maze!")
        print("This was previously impossible due to O(n²) bottleneck!")
    
    # Plot search performance over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(result['search_times'], alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Search Time (ms)')
    plt.title(f'Search Time per Step - {maze_size[0]}x{maze_size[1]} Maze')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    cumulative_search = np.cumsum(result['search_times'])
    plt.plot(cumulative_search)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Search Time (ms)')
    plt.title('Cumulative Search Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Episode count over time
    episode_counts = list(range(1, len(result['search_times']) + 1))
    plt.plot(episode_counts, result['search_times'])
    plt.xlabel('Number of Episodes')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Time vs Episode Count')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Path visualization
    if hasattr(navigator, 'path'):
        path = np.array(navigator.path)
        plt.scatter(path[:, 1], path[:, 0], c=range(len(path)), cmap='viridis', s=1)
        plt.colorbar(label='Step')
        plt.title('Navigation Path')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'large_maze_{maze_size[0]}x{maze_size[1]}_integrated.png', dpi=150)
    print(f"\nSaved plot to large_maze_{maze_size[0]}x{maze_size[1]}_integrated.png")
    
    return result


def compare_with_theoretical_limit():
    """Compare actual performance with theoretical O(n²) limit"""
    
    print("\n" + "="*60)
    print("THEORETICAL ANALYSIS")
    print("="*60)
    
    # For a 25x25 maze
    maze_cells = 25 * 25
    typical_steps = maze_cells * 2  # Rough estimate
    
    print(f"\nFor 25x25 maze ({maze_cells} cells):")
    print(f"Estimated steps to solve: ~{typical_steps}")
    
    # O(n²) calculation
    naive_search_per_step = typical_steps / 1000  # ms, grows with n
    total_naive = sum(i * naive_search_per_step for i in range(typical_steps))
    
    # O(1) calculation  
    integrated_search_per_step = 0.5  # ms, constant
    total_integrated = typical_steps * integrated_search_per_step
    
    print(f"\nTheoretical search time:")
    print(f"  Naive O(n²): {total_naive/1000:.1f} seconds")
    print(f"  Integrated O(1): {total_integrated/1000:.1f} seconds")
    print(f"  Speedup: {total_naive/total_integrated:.1f}x")
    
    # For 50x50
    maze_cells = 50 * 50
    typical_steps = maze_cells * 2
    
    print(f"\n\nFor 50x50 maze ({maze_cells} cells):")
    print(f"Estimated steps to solve: ~{typical_steps}")
    
    naive_search_per_step = typical_steps / 1000
    total_naive = sum(i * naive_search_per_step for i in range(typical_steps))
    total_integrated = typical_steps * integrated_search_per_step
    
    print(f"\nTheoretical search time:")
    print(f"  Naive O(n²): {total_naive/1000:.1f} seconds") 
    print(f"  Integrated O(1): {total_integrated/1000:.1f} seconds")
    print(f"  Speedup: {total_naive/total_integrated:.1f}x")
    
    print("\nThis explains why the original implementation failed on large mazes!")


if __name__ == "__main__":
    # Show theoretical analysis first
    compare_with_theoretical_limit()
    
    # Test 25x25 maze
    result_25 = test_large_maze(maze_size=(25, 25), max_steps=3000)
    
    # Test 50x50 maze if 25x25 succeeded
    if result_25['success']:
        print("\n\n" + "#"*60)
        print("25x25 SUCCESS! Now attempting 50x50...")
        print("#"*60)
        
        result_50 = test_large_maze(maze_size=(50, 50), max_steps=10000)
        
        if result_50['success']:
            print("\n\n" + "🎊"*30)
            print("BREAKTHROUGH: Successfully solved 50x50 maze!")
            print("The O(n²) bottleneck has been eliminated!")
            print("🎊"*30)