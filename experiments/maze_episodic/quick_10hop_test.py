#!/usr/bin/env python3
"""Quick test of 10-hop - limited steps"""

import numpy as np
import sys
import os
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query_10hop import PureEpisodicGeDIGQuery10Hop
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Quick test
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

print("="*60)
print("Quick 10-Hop Test (1500 steps limit)")
print("="*60)

navigator = PureEpisodicGeDIGQuery10Hop(maze, message_depth=10)
print(f"Start: {navigator.position}, Goal: {navigator.goal}")

# Limited navigation
start_time = time.time()
result = navigator.navigate(max_steps=1500)
elapsed = time.time() - start_time

print(f"\nResults after {result['steps']} steps ({elapsed:.1f}s):")
print(f"Success: {result['success']}")
print(f"Wall hits: {result['wall_hits']} ({result['wall_hits']/result['steps']*100:.1f}%)")
print(f"Position: {navigator.position}")
print(f"Distance to goal: {abs(navigator.position[0]-navigator.goal[0])+abs(navigator.position[1]-navigator.goal[1])}")

# Show hop usage
print("\nHop usage:")
active_hops = {k: v for k, v in result['hop_selections'].items() if v > 0}
for hop, count in sorted(active_hops.items()):
    print(f"  {hop}: {count}")

print(f"\nCache: {result['cache_hits']} hits, {result['cache_misses']} misses")

if result['success']:
    print("\n🎉 SUCCESS with 10-hop pure episodic memory!")