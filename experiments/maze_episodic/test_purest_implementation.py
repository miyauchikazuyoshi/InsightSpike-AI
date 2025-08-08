#!/usr/bin/env python3
"""
Test the PUREST implementation without any interventions
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_purest import PureEpisodicGeDIGPurest
from insightspike.environments.proper_maze_generator import ProperMazeGenerator


def test_purest_implementation(maze_size=(25, 25), max_steps=5000):
    """Test purest implementation without interventions"""
    
    print(f"\n{'='*60}")
    print(f"Testing PUREST Implementation on {maze_size[0]}x{maze_size[1]} maze")
    print(f"{'='*60}")
    print("NO wall filtering - can try to walk into walls")
    print("NO exploration bonus - pure episodic memory decisions")
    print("ONLY memory-based navigation\n")
    
    # Generate maze
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=maze_size)
    
    # Create navigator
    navigator = PureEpisodicGeDIGPurest(maze, message_depth=3)
    print(f"Start: {navigator.position}, Goal: {navigator.goal}")
    
    # Navigate
    print("\nStarting PUREST navigation...")
    result = navigator.navigate(max_steps=max_steps)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Success: {result['success']}")
    print(f"Steps taken: {result['steps']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Wall hits: {result['wall_hits']} (learning from failures!)")
    print(f"Average search time: {result['avg_search_time']:.2f}ms")
    
    # Print hop selections
    print("\nSearch mode usage:")
    for mode, count in result['hop_selections'].items():
        print(f"  {mode}: {count} times")
    
    # Print failure rate
    if result['steps'] > 0:
        failure_rate = result['wall_hits'] / result['steps'] * 100
        print(f"\nWall hit rate: {failure_rate:.1f}%")
        print("(This shows the system is learning from failures)")
    
    if result['success']:
        print(f"\n🎉 SUCCESS with PUREST implementation!")
        print("No cheating, no interventions, pure episodic memory!")
    
    # Visualize results
    visualize_purest_results(navigator, result, maze_size)
    
    return result


def visualize_purest_results(navigator, result, maze_size):
    """Visualize results with wall hit analysis"""
    
    plt.figure(figsize=(15, 10))
    
    # Path visualization with wall hits
    plt.subplot(2, 3, 1)
    path = np.array(result['path'])
    
    # Count wall hit positions
    wall_hit_positions = []
    for i in range(len(path)-1):
        if np.array_equal(path[i], path[i+1]):
            wall_hit_positions.append(path[i])
    
    plt.imshow(navigator.maze, cmap='binary', alpha=0.3)
    plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=1, alpha=0.5)
    
    if wall_hit_positions:
        wall_hits = np.array(wall_hit_positions)
        plt.scatter(wall_hits[:, 1], wall_hits[:, 0], c='red', s=20, 
                   marker='x', label='Wall hits')
    
    plt.plot(navigator.position[1], navigator.position[0], 'go', markersize=8)
    plt.plot(navigator.goal[1], navigator.goal[0], 'ro', markersize=8)
    plt.title('Navigation Path (red X = wall hits)')
    plt.legend()
    
    # Wall hit frequency over time
    plt.subplot(2, 3, 2)
    window = 100
    wall_hit_rate = []
    
    for i in range(window, len(path)):
        hits_in_window = sum(1 for j in range(i-window, i-1) 
                           if np.array_equal(path[j], path[j+1]))
        wall_hit_rate.append(hits_in_window / window * 100)
    
    if wall_hit_rate:
        plt.plot(wall_hit_rate)
        plt.xlabel('Step (offset by 100)')
        plt.ylabel('Wall Hit Rate (%)')
        plt.title('Wall Hit Rate Over Time (100-step window)')
        plt.grid(True, alpha=0.3)
    
    # Search time analysis
    plt.subplot(2, 3, 3)
    search_times = result['search_times']
    plt.plot(search_times, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Time per Step')
    plt.grid(True, alpha=0.3)
    
    # Episode accumulation
    plt.subplot(2, 3, 4)
    plt.plot(range(len(search_times)))
    plt.xlabel('Step')
    plt.ylabel('Total Episodes')
    plt.title('Episode Accumulation')
    plt.grid(True, alpha=0.3)
    
    # Hop selection distribution
    plt.subplot(2, 3, 5)
    modes = list(result['hop_selections'].keys())
    counts = list(result['hop_selections'].values())
    plt.bar(modes, counts)
    plt.xlabel('Search Mode')
    plt.ylabel('Usage Count')
    plt.title('Search Mode Distribution')
    
    # Success/Failure episode ratio
    plt.subplot(2, 3, 6)
    success_count = sum(1 for ep in navigator.index.metadata 
                       if ep.get('success', False))
    failure_count = len(navigator.index.metadata) - success_count
    
    plt.pie([success_count, failure_count], 
            labels=['Successful moves', 'Wall hits'],
            autopct='%1.1f%%')
    plt.title('Action Success Rate')
    
    plt.tight_layout()
    plt.savefig(f'purest_navigation_{maze_size[0]}x{maze_size[1]}.png', dpi=150)
    print(f"\nSaved visualization to purest_navigation_{maze_size[0]}x{maze_size[1]}.png")


def compare_implementations():
    """Compare different implementation approaches"""
    
    print("\n" + "="*60)
    print("IMPLEMENTATION COMPARISON")
    print("="*60)
    
    print("\n1. Original (with interventions):")
    print("   - Wall filtering: if vis.get(action) == 'path'")
    print("   - Exploration bonus: +0.3 for unvisited")
    print("   - Result: Efficient but not pure")
    
    print("\n2. PUREST (no interventions):")
    print("   - NO wall filtering")
    print("   - NO exploration bonus")
    print("   - Can attempt wall moves")
    print("   - Learns from failures")
    print("   - Result: True episodic memory system")
    
    print("\nKey difference:")
    print("The PUREST version must LEARN that walls block movement")
    print("through failed attempts, just like real episodic memory!")


if __name__ == "__main__":
    # Show comparison
    compare_implementations()
    
    # Test on 15x15 first
    print("\n\nStarting with 15x15 maze...")
    result_15 = test_purest_implementation(maze_size=(15, 15), max_steps=2000)
    
    # If successful or interesting, try 25x25
    if result_15['success'] or result_15['wall_hits'] > 50:
        print("\n\n" + "#"*60)
        print("Testing on 25x25 maze...")
        print("#"*60)
        
        result_25 = test_purest_implementation(maze_size=(25, 25), max_steps=5000)