#!/usr/bin/env python3
"""Quick test of query strategy improvement"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query import PureEpisodicGeDIGQuery
from pure_episodic_gedig_purest import PureEpisodicGeDIGPurest
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Generate same maze for both
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

print("="*60)
print("COMPARISON: Purest vs Query Strategy (15x15 maze)")
print("="*60)

# Test purest
print("\n1. PUREST Implementation (no interventions):")
navigator_purest = PureEpisodicGeDIGPurest(maze.copy(), message_depth=3)
result_purest = navigator_purest.navigate(max_steps=500)
print(f"   Wall hits: {result_purest['wall_hits']} ({result_purest['wall_hits']/500*100:.1f}%)")
print(f"   Position: {navigator_purest.position}")

# Test query
print("\n2. QUERY Strategy (aspirational queries):")
navigator_query = PureEpisodicGeDIGQuery(maze.copy(), message_depth=3)
result_query = navigator_query.navigate(max_steps=500)
print(f"   Wall hits: {result_query['wall_hits']} ({result_query['wall_hits']/500*100:.1f}%)")
print(f"   Position: {navigator_query.position}")

# Compare
print("\n" + "="*60)
print("IMPROVEMENT:")
print("="*60)
improvement = (result_purest['wall_hits'] - result_query['wall_hits']) / result_purest['wall_hits'] * 100
print(f"Wall hit reduction: {improvement:.1f}%")
print(f"Purest moved: {abs(navigator_purest.position[0] - 1) + abs(navigator_purest.position[1] - 1)} steps from start")
print(f"Query moved: {abs(navigator_query.position[0] - 1) + abs(navigator_query.position[1] - 1)} steps from start")