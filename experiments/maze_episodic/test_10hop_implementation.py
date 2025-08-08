#!/usr/bin/env python3
"""Test 10-hop ultra-deep implementation"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query_10hop import PureEpisodicGeDIGQuery10Hop
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Test on 15x15 first
print("\n" + "="*60)
print("Testing 10-Hop Ultra-Deep Message Passing")
print("="*60)
print("Features:")
print("- Up to 10-hop propagation")
print("- Adaptive depth selection")
print("- Propagation caching")
print("- Enhanced goal bias")
print("- No wall filtering, no exploration bonus!")

generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

navigator = PureEpisodicGeDIGQuery10Hop(maze, message_depth=10)
print(f"\nMaze: 15x15")
print(f"Start: {navigator.position}, Goal: {navigator.goal}")

# Navigate
start_time = time.time()
result = navigator.navigate(max_steps=5000)
elapsed = time.time() - start_time

print(f"\n{'='*60}")
print("FINAL RESULTS:")
print(f"{'='*60}")
print(f"Success: {result['success']}")
print(f"Steps taken: {result['steps']}")
print(f"Wall hits: {result['wall_hits']} ({result['wall_hits']/result['steps']*100:.1f}%)")
print(f"Time elapsed: {elapsed:.1f}s")
print(f"Avg search time: {result['avg_search_time']:.2f}ms")

# Show hop distribution
print("\nDeep hop usage:")
total_hops = sum(result['hop_selections'].values())
for hop, count in sorted(result['hop_selections'].items()):
    if count > 0:
        print(f"  {hop}: {count} ({count/total_hops*100:.1f}%)")

print(f"\nCache performance:")
print(f"  Hits: {result['cache_hits']}")
print(f"  Misses: {result['cache_misses']}")
print(f"  Hit rate: {result['cache_hits']/(result['cache_hits']+result['cache_misses'])*100:.1f}%")

if result['success']:
    print("\n" + "🎉"*30)
    print("HISTORIC ACHIEVEMENT!")
    print("Solved maze with PURE episodic memory!")
    print("Using ultra-deep 10-hop message passing!")
    print("No cheats, just deep reasoning!")
    print("🎉"*30)
    
    # Visualize the successful path
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='binary', alpha=0.3)
    
    path = np.array(result['path'])
    plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, alpha=0.8, label='Solution path')
    
    plt.plot(navigator._find_start()[1], navigator._find_start()[0], 'go', 
            markersize=15, label='Start')
    plt.plot(navigator.goal[1], navigator.goal[0], 'ro', 
            markersize=15, label='Goal')
    
    plt.title(f'10-Hop Pure Episodic Navigation Success!\n'
              f'Steps: {result["steps"]}, Wall hits: {result["wall_hits"]} '
              f'({result["wall_hits"]/result["steps"]*100:.1f}%)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    
    filename = '10hop_success_15x15.png'
    plt.savefig(filename, dpi=150)
    print(f"\nSaved success visualization to {filename}")
    
    # Try on larger maze
    print("\n\nTrying on 25x25 maze...")
    maze_25 = generator.generate_dfs_maze(size=(25, 25), seed=42)
    navigator_25 = PureEpisodicGeDIGQuery10Hop(maze_25, message_depth=10)
    result_25 = navigator_25.navigate(max_steps=10000)
    
    if result_25['success']:
        print("\n🎉 ALSO SOLVED 25x25! 🎉")
else:
    final_dist = abs(navigator.position[0]-navigator.goal[0])+abs(navigator.position[1]-navigator.goal[1])
    print(f"\nFinal position: {navigator.position}")
    print(f"Distance to goal: {final_dist}")
    
    # Check if we got very close
    if final_dist <= 3:
        print("\n⭐ Got VERY close to the goal! Almost there!")
    elif final_dist <= 5:
        print("\n✨ Made significant progress toward the goal!")