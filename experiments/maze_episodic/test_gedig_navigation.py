#!/usr/bin/env python3
"""
Test geDIG-aware navigation
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig import PureEpisodicGeDIG
from insightspike.environments.proper_maze_generator import ProperMazeGenerator


def test_gedig_navigation(maze_size=(25, 25), max_steps=3000):
    """Test geDIG-aware navigator"""
    
    print(f"\n{'='*60}")
    print(f"Testing geDIG-aware Navigation on {maze_size[0]}x{maze_size[1]} maze")
    print(f"{'='*60}")
    
    # Generate maze
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=maze_size)
    
    # Create navigator
    navigator = PureEpisodicGeDIG(maze, message_depth=3)
    print(f"Start: {navigator.position}, Goal: {navigator.goal}")
    
    # Navigate
    print("\nStarting navigation with geDIG-aware edge selection...")
    result = navigator.navigate(max_steps=max_steps)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Success: {result['success']}")
    print(f"Steps taken: {result['steps']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Average search time: {result['avg_search_time']:.2f}ms")
    
    # Print hop selections
    print("\nSearch mode usage:")
    for mode, count in result['hop_selections'].items():
        print(f"  {mode}: {count} times")
    
    # Print index statistics
    stats = result['index_stats']
    print("\nIndex statistics:")
    print(f"  Episodes: {stats['episodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Average degree: {stats['avg_degree']:.2f}")
    print(f"  Edge acceptance rate: {stats['edge_acceptance_rate']:.2f}")
    print(f"  geDIG calculations: {stats['gedig_calculations']}")
    
    if result['success']:
        print(f"\n🎉 Successfully solved using geDIG-aware navigation!")
    
    # Visualize results
    visualize_results(navigator, result, maze_size)
    
    return result


def visualize_results(navigator, result, maze_size):
    """Visualize navigation results"""
    
    plt.figure(figsize=(15, 10))
    
    # Search time analysis
    plt.subplot(2, 3, 1)
    search_times = result['search_times']
    plt.plot(search_times, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Time per Step')
    plt.grid(True, alpha=0.3)
    
    # Cumulative search time
    plt.subplot(2, 3, 2)
    plt.plot(np.cumsum(search_times))
    plt.xlabel('Step')
    plt.ylabel('Cumulative Search Time (ms)')
    plt.title('Total Search Time')
    plt.grid(True, alpha=0.3)
    
    # Hop selection distribution
    plt.subplot(2, 3, 3)
    modes = list(result['hop_selections'].keys())
    counts = list(result['hop_selections'].values())
    plt.bar(modes, counts)
    plt.xlabel('Search Mode')
    plt.ylabel('Usage Count')
    plt.title('Search Mode Distribution')
    
    # Path visualization
    plt.subplot(2, 3, 4)
    path = np.array(result['path'])
    plt.scatter(path[:, 1], path[:, 0], c=range(len(path)), cmap='viridis', s=2)
    plt.colorbar(label='Step')
    plt.title('Navigation Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    
    # Edge acceptance over time
    plt.subplot(2, 3, 5)
    # Sample edge statistics every 100 episodes
    edge_counts = []
    episode_counts = []
    
    for i in range(0, len(navigator.index.metadata), 100):
        edges = 0
        for j in range(min(i, len(navigator.index.metadata))):
            if j in navigator.index.graph:
                edges += len(list(navigator.index.graph.neighbors(j)))
        edge_counts.append(edges / 2)  # Each edge counted twice
        episode_counts.append(i)
    
    plt.plot(episode_counts, edge_counts)
    plt.xlabel('Episodes')
    plt.ylabel('Total Edges')
    plt.title('Graph Growth')
    plt.grid(True, alpha=0.3)
    
    # Maze with path
    plt.subplot(2, 3, 6)
    plt.imshow(navigator.maze, cmap='binary')
    plt.plot(path[:, 1], path[:, 0], 'r-', linewidth=1, alpha=0.7)
    plt.plot(navigator.position[1], navigator.position[0], 'go', markersize=8)
    plt.plot(navigator.goal[1], navigator.goal[0], 'ro', markersize=8)
    plt.title(f'{maze_size[0]}x{maze_size[1]} Maze Solution')
    
    plt.tight_layout()
    plt.savefig(f'gedig_navigation_{maze_size[0]}x{maze_size[1]}.png', dpi=150)
    print(f"\nSaved visualization to gedig_navigation_{maze_size[0]}x{maze_size[1]}.png")


def compare_approaches():
    """Compare different approaches"""
    
    print("\n" + "="*60)
    print("APPROACH COMPARISON")
    print("="*60)
    
    print("\n1. Original Pure Episodic: O(n²) search bottleneck")
    print("   - All episodes compared on every search")
    print("   - Failed on large mazes due to computational cost")
    
    print("\n2. Integrated Index (O(1)): Fast but simple")
    print("   - Pre-normalized vectors for O(1) search")
    print("   - Graph based only on similarity")
    print("   - 25x25 success, 50x50 failed (navigation issue)")
    
    print("\n3. geDIG-aware (Current): Smart edge selection")
    print("   - Combines similarity with geDIG evaluation")
    print("   - GED considers spatial/temporal/action distance")
    print("   - IG rewards diverse, informative connections")
    print("   - Multiple search modes for different situations")
    
    print("\nKey innovation: Edges selected by BOTH similarity AND geDIG value")
    print("This creates a more intelligent graph structure for navigation!")


if __name__ == "__main__":
    # Show comparison
    compare_approaches()
    
    # Test on 25x25
    result_25 = test_gedig_navigation(maze_size=(25, 25), max_steps=3000)
    
    # If successful, try 50x50
    if result_25['success']:
        print("\n\n" + "#"*60)
        print("25x25 success! Now attempting 50x50...")
        print("#"*60)
        
        result_50 = test_gedig_navigation(maze_size=(50, 50), max_steps=10000)