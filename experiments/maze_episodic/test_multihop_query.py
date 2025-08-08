#!/usr/bin/env python3
"""Test multi-hop query implementation"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pure_episodic_gedig_query_multihop import PureEpisodicGeDIGQueryMultihop
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# Test on 15x15
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(15, 15), seed=42)

print("="*60)
print("Testing Multi-hop Query Strategy (15x15 maze)")
print("="*60)

navigator = PureEpisodicGeDIGQueryMultihop(maze, message_depth=3)
print(f"Start: {navigator.position}, Goal: {navigator.goal}")

result = navigator.navigate(max_steps=1000)

print(f"\nSuccess: {result['success']}")
print(f"Steps: {result['steps']}")
print(f"Wall hits: {result['wall_hits']} ({result['wall_hits']/result['steps']*100:.1f}%)")

print("\nMulti-hop usage:")
for mode, count in result['hop_selections'].items():
    print(f"  {mode}: {count} times")

if result['success']:
    print("\n🎉 SUCCESS with multi-hop query strategy!")