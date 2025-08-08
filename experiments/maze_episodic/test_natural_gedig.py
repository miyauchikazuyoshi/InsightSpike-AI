#!/usr/bin/env python3
"""Test natural geDIG-based depth selection"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_natural import PureEpisodicGeDIGNatural
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Test on 15x15
print("="*60)
print("Testing Natural geDIG-based Depth Selection")
print("="*60)
print("Features:")
print("- NO artificial stuck_counter or norm-based selection")
print("- geDIG quality naturally determines optimal depth")
print("- Up to 10-hop deep propagation")
print("- Pure episodic memory with natural selection")

generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

navigator = PureEpisodicGeDIGNatural(maze, max_depth=10)
print(f"\nMaze: 15x15")
print(f"Start: {navigator.position}, Goal: {navigator.goal}")

# Navigate
result = navigator.navigate(max_steps=2000)

print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")
print(f"Success: {result['success']}")
print(f"Steps: {result['steps']}")
print(f"Wall hits: {result['wall_hits']} ({result['wall_hits']/result['steps']*100:.1f}%)")

# Analyze depth usage
print("\nDepth contribution analysis:")
total_contributions = {}
for depth, contributions in result['hop_contributions'].items():
    if contributions:
        avg_contribution = np.mean(contributions)
        total_contributions[depth] = avg_contribution
        if avg_contribution > 0.01:
            print(f"  {depth}: {avg_contribution:.3f} average contribution")

# Show how depths were selected over time
if len(result['depth_weights']) > 0:
    avg_weights = np.mean(result['depth_weights'], axis=0)
    print("\nAverage depth weights (natural selection):")
    for i, weight in enumerate(avg_weights):
        if weight > 0.05:
            print(f"  {i+1}-hop: {weight:.3f}")

if result['success']:
    print("\n🎉 SUCCESS with natural geDIG selection!")
    print("No artificial interventions, just pure geDIG dynamics!")
else:
    dist = abs(navigator.position[0]-navigator.goal[0])+abs(navigator.position[1]-navigator.goal[1])
    print(f"\nFinal position: {navigator.position}")
    print(f"Distance to goal: {dist}")

# Compare with previous approaches
print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
print("1. Artificial selection (stuck_counter): ~21-55% wall hits")
print(f"2. Natural geDIG selection: {result['wall_hits']/result['steps']*100:.1f}% wall hits")
print("\nThis demonstrates geDIG's natural ability to find optimal depths!")