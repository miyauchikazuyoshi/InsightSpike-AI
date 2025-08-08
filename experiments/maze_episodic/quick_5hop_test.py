#!/usr/bin/env python3
"""Quick test of 5-hop implementation"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query_5hop import PureEpisodicGeDIGQuery5Hop
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Quick test on 15x15
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

print("="*60)
print("Testing 5-Hop Deep Message Passing (15x15 maze)")
print("="*60)

navigator = PureEpisodicGeDIGQuery5Hop(maze, message_depth=5)
print(f"Start: {navigator.position}, Goal: {navigator.goal}")

# Navigate for limited steps
result = navigator.navigate(max_steps=1000)

print(f"\nResults after 1000 steps:")
print(f"Success: {result['success']}")
print(f"Wall hits: {result['wall_hits']} ({result['wall_hits']/1000*100:.1f}%)")
print(f"Position: {navigator.position}")
print(f"Distance to goal: {abs(navigator.position[0]-navigator.goal[0])+abs(navigator.position[1]-navigator.goal[1])}")

print("\nHop usage distribution:")
total = sum(result['hop_selections'].values())
for mode, count in sorted(result['hop_selections'].items()):
    if count > 0:
        print(f"  {mode}: {count} ({count/total*100:.1f}%)")