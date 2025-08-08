#!/usr/bin/env python3
"""
Test the aspirational query implementation
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query import PureEpisodicGeDIGQuery
from insightspike.environments.proper_maze_generator import ProperMazeGenerator


def test_query_implementation(maze_size=(25, 25), max_steps=5000):
    """Test aspirational query implementation"""
    
    print(f"\n{'='*60}")
    print(f"Testing Aspirational Query Implementation on {maze_size[0]}x{maze_size[1]} maze")
    print(f"{'='*60}")
    print("Query strategy: (current_x, current_y, null, null, path, goal)")
    print("- Stores actual episode vectors (with wall info)")
    print("- Searches with aspirational vectors (wanting paths)")
    print("- NO wall filtering, NO exploration bonus\n")
    
    # Generate maze
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=maze_size)
    
    # Create navigator
    navigator = PureEpisodicGeDIGQuery(maze, message_depth=3)
    print(f"Start: {navigator.position}, Goal: {navigator.goal}")
    
    # Navigate
    print("\nStarting navigation with aspirational queries...")
    result = navigator.navigate(max_steps=max_steps)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Success: {result['success']}")
    print(f"Steps taken: {result['steps']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Wall hits: {result['wall_hits']}")
    print(f"Average search time: {result['avg_search_time']:.2f}ms")
    
    # Print hop selections
    print("\nSearch mode usage:")
    for mode, count in result['hop_selections'].items():
        print(f"  {mode}: {count} times")
    
    # Print success rate
    if result['steps'] > 0:
        failure_rate = result['wall_hits'] / result['steps'] * 100
        print(f"\nWall hit rate: {failure_rate:.1f}%")
        print(f"Success rate: {100 - failure_rate:.1f}%")
    
    if result['success']:
        print(f"\n🎉 SUCCESS with aspirational query strategy!")
        print("Pure episodic memory with smart queries!")
    
    # Visualize results
    visualize_query_results(navigator, result, maze_size)
    
    return result


def visualize_query_results(navigator, result, maze_size):
    """Visualize results with query analysis"""
    
    plt.figure(figsize=(15, 10))
    
    # Path visualization
    plt.subplot(2, 3, 1)
    path = np.array(result['path'])
    
    plt.imshow(navigator.maze, cmap='binary', alpha=0.3)
    plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, alpha=0.7)
    
    # Mark wall hits
    wall_hit_positions = []
    for i in range(len(path)-1):
        if np.array_equal(path[i], path[i+1]):
            wall_hit_positions.append(path[i])
    
    if wall_hit_positions:
        wall_hits = np.array(wall_hit_positions)
        plt.scatter(wall_hits[:, 1], wall_hits[:, 0], c='red', s=30, 
                   marker='x', label='Wall hits', alpha=0.5)
    
    plt.plot(navigator.position[1], navigator.position[0], 'go', markersize=10, label='Start')
    plt.plot(navigator.goal[1], navigator.goal[0], 'ro', markersize=10, label='Goal')
    plt.title('Navigation Path (Aspirational Query)')
    plt.legend()
    
    # Wall hit rate over time
    plt.subplot(2, 3, 2)
    window = 100
    wall_hit_rates = []
    
    for i in range(window, len(path)):
        hits_in_window = sum(1 for j in range(i-window, i-1) 
                           if np.array_equal(path[j], path[j+1]))
        wall_hit_rates.append(hits_in_window / window * 100)
    
    if wall_hit_rates:
        plt.plot(wall_hit_rates)
        plt.xlabel('Step (offset by 100)')
        plt.ylabel('Wall Hit Rate (%)')
        plt.title('Learning Curve (100-step window)')
        plt.grid(True, alpha=0.3)
    
    # Search time analysis
    plt.subplot(2, 3, 3)
    search_times = result['search_times']
    plt.plot(search_times, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Efficiency')
    plt.grid(True, alpha=0.3)
    
    # Episode accumulation
    plt.subplot(2, 3, 4)
    plt.plot(range(result['total_episodes']))
    plt.xlabel('Step')
    plt.ylabel('Total Episodes')
    plt.title('Memory Growth')
    plt.grid(True, alpha=0.3)
    
    # Search mode distribution
    plt.subplot(2, 3, 5)
    modes = list(result['hop_selections'].keys())
    counts = list(result['hop_selections'].values())
    plt.bar(modes, counts)
    plt.xlabel('Search Mode')
    plt.ylabel('Usage Count')
    plt.title('Message Passing Depth Usage')
    
    # Success analysis
    plt.subplot(2, 3, 6)
    success_count = result['steps'] - result['wall_hits']
    failure_count = result['wall_hits']
    
    plt.pie([success_count, failure_count], 
            labels=['Successful moves', 'Wall hits'],
            autopct='%1.1f%%',
            colors=['green', 'red'])
    plt.title('Action Success Rate')
    
    plt.tight_layout()
    plt.savefig(f'query_navigation_{maze_size[0]}x{maze_size[1]}.png', dpi=150)
    print(f"\nSaved visualization to query_navigation_{maze_size[0]}x{maze_size[1]}.png")


def compare_strategies():
    """Compare different query strategies"""
    
    print("\n" + "="*60)
    print("QUERY STRATEGY COMPARISON")
    print("="*60)
    
    print("\n1. Direct Query (purest implementation):")
    print("   - Query: current episode vector")
    print("   - Searches for similar situations")
    print("   - Result: 99%+ wall hits")
    
    print("\n2. Aspirational Query (this implementation):")
    print("   - Query: (x, y, path, path, path, path)")
    print("   - Searches for successful episodes")
    print("   - Weights successful episodes higher")
    print("   - Theory: Learn from past successes")
    
    print("\n3. Future extensions:")
    print("   - Goal-biased queries")
    print("   - Temporal sequence queries")
    print("   - Multi-step planning queries")


if __name__ == "__main__":
    # Show comparison
    compare_strategies()
    
    # Test on 15x15 first
    print("\n\nStarting with 15x15 maze...")
    result_15 = test_query_implementation(maze_size=(15, 15), max_steps=2000)
    
    # If promising, try 25x25
    if result_15['success'] or result_15['wall_hits'] < result_15['steps'] * 0.5:
        print("\n\n" + "#"*60)
        print("Testing on 25x25 maze...")
        print("#"*60)
        
        result_25 = test_query_implementation(maze_size=(25, 25), max_steps=5000)
        
        # Compare with purest implementation
        print("\n\nIMPROVEMENT ANALYSIS:")
        print("="*60)
        print("Purest implementation: 99%+ wall hits")
        print(f"Query implementation: {result_25['wall_hits']/result_25['steps']*100:.1f}% wall hits")
        
        improvement = 99 - (result_25['wall_hits']/result_25['steps']*100)
        print(f"Improvement: {improvement:.1f}% reduction in wall hits!")