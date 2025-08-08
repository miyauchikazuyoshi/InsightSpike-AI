#!/usr/bin/env python3
"""Test 5-hop implementation"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query_5hop import PureEpisodicGeDIGQuery5Hop
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Test on progressively larger mazes
for size in [(15, 15), (25, 25)]:
    print("\n" + "="*60)
    print(f"Testing 5-Hop Deep Message Passing on {size[0]}x{size[1]} maze")
    print("="*60)
    
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=size, seed=42)
    
    navigator = PureEpisodicGeDIGQuery5Hop(maze, message_depth=5)
    print(f"Start: {navigator.position}, Goal: {navigator.goal}")
    print("Features: Goal-biased queries, 5-hop propagation, adaptive combination")
    
    # Navigate with appropriate steps
    max_steps = 2000 if size == (15, 15) else 5000
    result = navigator.navigate(max_steps=max_steps)
    
    print(f"\nFinal Results:")
    print(f"Success: {result['success']}")
    print(f"Steps taken: {result['steps']}")
    print(f"Wall hits: {result['wall_hits']} ({result['wall_hits']/result['steps']*100:.1f}%)")
    print(f"Final position: {navigator.position}")
    
    print("\nDeep Multi-hop usage:")
    for mode, count in sorted(result['hop_selections'].items()):
        if count > 0:
            print(f"  {mode}: {count} times ({count/result['steps']*100:.1f}%)")
    
    # Visualize if successful or interesting
    if result['success'] or result['wall_hits'] < result['steps'] * 0.3:
        plt.figure(figsize=(10, 10))
        
        # Plot maze and path
        plt.imshow(maze, cmap='binary', alpha=0.3)
        
        path = np.array(result['path'])
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, alpha=0.7, label='Path')
        
        # Mark start and goal
        plt.plot(navigator._find_start()[1], navigator._find_start()[0], 'go', 
                markersize=15, label='Start')
        plt.plot(navigator.goal[1], navigator.goal[0], 'ro', 
                markersize=15, label='Goal')
        
        # Mark wall hits
        wall_hit_positions = []
        for i in range(len(path)-1):
            if np.array_equal(path[i], path[i+1]):
                wall_hit_positions.append(path[i])
        
        if wall_hit_positions and len(wall_hit_positions) < 100:
            wall_hits = np.array(wall_hit_positions)
            plt.scatter(wall_hits[:, 1], wall_hits[:, 0], c='red', s=20, 
                       marker='x', alpha=0.5, label=f'Wall hits ({len(wall_hit_positions)})')
        
        plt.title(f'5-Hop Navigation on {size[0]}x{size[1]} Maze\n'
                  f'Success: {result["success"]}, Steps: {result["steps"]}, '
                  f'Wall hit rate: {result["wall_hits"]/result["steps"]*100:.1f}%')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        
        filename = f'5hop_navigation_{size[0]}x{size[1]}.png'
        plt.savefig(filename, dpi=150)
        print(f"\nSaved visualization to {filename}")
        plt.close()
    
    if result['success']:
        print("\n🎉 BREAKTHROUGH! Successfully solved with pure episodic memory!")
        print("No wall filtering, no exploration bonus - just deep multi-hop reasoning!")
        break