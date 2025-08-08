#!/usr/bin/env python3
"""Extended test to see if 5-hop can solve the maze"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query_5hop import PureEpisodicGeDIGQuery5Hop
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Test on 15x15 with more steps
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

print("="*60)
print("Extended 5-Hop Test - Can it solve the maze?")
print("="*60)

navigator = PureEpisodicGeDIGQuery5Hop(maze, message_depth=5)
print(f"Start: {navigator.position}, Goal: {navigator.goal}")

# Give it more steps
result = navigator.navigate(max_steps=3000)

print(f"\nFinal Results:")
print(f"Success: {result['success']}")
print(f"Steps taken: {result['steps']}")
print(f"Wall hits: {result['wall_hits']} ({result['wall_hits']/result['steps']*100:.1f}%)")
print(f"Final position: {navigator.position}")

if result['success']:
    print("\n" + "🎉"*20)
    print("HISTORIC ACHIEVEMENT!")
    print("Solved maze with PURE episodic memory!")
    print("No wall filtering, no exploration bonus!")
    print("Just deep multi-hop message passing!")
    print("🎉"*20)
    
    # Show path length
    print(f"\nPath length: {len(result['path'])}")
    print(f"Efficiency: {len(result['path'])/result['steps']*100:.1f}%")
else:
    dist = abs(navigator.position[0]-navigator.goal[0])+abs(navigator.position[1]-navigator.goal[1])
    print(f"\nDistance to goal: {dist}")
    print(f"Got stuck at: {navigator.position}")