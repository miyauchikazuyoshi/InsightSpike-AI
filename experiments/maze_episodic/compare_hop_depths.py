#!/usr/bin/env python3
"""Compare different hop depths quickly"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query import PureEpisodicGeDIGQuery
from pure_episodic_gedig_query_multihop import PureEpisodicGeDIGQueryMultihop
from pure_episodic_gedig_query_5hop import PureEpisodicGeDIGQuery5Hop
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Generate same maze
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

print("="*60)
print("HOP DEPTH COMPARISON (500 steps each)")
print("="*60)

results = {}

# Test original query implementation
print("\n1. Aspirational Query (basic):")
nav1 = PureEpisodicGeDIGQuery(maze.copy(), message_depth=3)
result1 = nav1.navigate(max_steps=500)
results['query'] = {
    'wall_hits': result1['wall_hits'],
    'hit_rate': result1['wall_hits']/500*100,
    'position': nav1.position,
    'distance': abs(nav1.position[0]-nav1.goal[0])+abs(nav1.position[1]-nav1.goal[1])
}
print(f"   Wall hits: {result1['wall_hits']} ({results['query']['hit_rate']:.1f}%)")
print(f"   Distance to goal: {results['query']['distance']}")

# Test multi-hop
print("\n2. Multi-hop (3-hop max):")
nav2 = PureEpisodicGeDIGQueryMultihop(maze.copy(), message_depth=3)
result2 = nav2.navigate(max_steps=500)
results['multihop'] = {
    'wall_hits': result2['wall_hits'],
    'hit_rate': result2['wall_hits']/500*100,
    'position': nav2.position,
    'distance': abs(nav2.position[0]-nav2.goal[0])+abs(nav2.position[1]-nav2.goal[1])
}
print(f"   Wall hits: {result2['wall_hits']} ({results['multihop']['hit_rate']:.1f}%)")
print(f"   Distance to goal: {results['multihop']['distance']}")

# Test 5-hop
print("\n3. Deep 5-hop:")
nav3 = PureEpisodicGeDIGQuery5Hop(maze.copy(), message_depth=5)
result3 = nav3.navigate(max_steps=500)
results['5hop'] = {
    'wall_hits': result3['wall_hits'],
    'hit_rate': result3['wall_hits']/500*100,
    'position': nav3.position,
    'distance': abs(nav3.position[0]-nav3.goal[0])+abs(nav3.position[1]-nav3.goal[1])
}
print(f"   Wall hits: {result3['wall_hits']} ({results['5hop']['hit_rate']:.1f}%)")
print(f"   Distance to goal: {results['5hop']['distance']}")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"{'Method':<15} {'Wall Hit Rate':<15} {'Distance to Goal':<20}")
print("-"*50)
for name, data in results.items():
    print(f"{name:<15} {data['hit_rate']:<15.1f}% {data['distance']:<20}")

# Calculate improvements
baseline = results['query']['hit_rate']
print("\nImprovement over baseline:")
for name, data in results.items():
    if name != 'query':
        improvement = baseline - data['hit_rate']
        print(f"{name}: {improvement:.1f}% reduction in wall hits")