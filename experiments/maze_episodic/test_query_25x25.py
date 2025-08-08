#!/usr/bin/env python3
"""Test aspirational query on 25x25 maze"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query import PureEpisodicGeDIGQuery
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Generate 25x25 maze
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(25, 25))

# Test
navigator = PureEpisodicGeDIGQuery(maze, message_depth=3)
print(f'Start: {navigator.position}, Goal: {navigator.goal}')
print('\nTesting 25x25 maze with aspirational query...')

result = navigator.navigate(max_steps=5000)

print(f'\nSuccess: {result["success"]}')
print(f'Steps: {result["steps"]}')
print(f'Wall hits: {result["wall_hits"]} ({result["wall_hits"]/result["steps"]*100:.1f}%)')
print(f'Final position: {navigator.position}')

if result["success"]:
    print("\n🎉 SUCCESS! Aspirational query strategy works!")
else:
    print(f"\nReached position {navigator.position}, goal was {navigator.goal}")
    print(f"Distance to goal: {abs(navigator.position[0] - navigator.goal[0]) + abs(navigator.position[1] - navigator.goal[1])}")