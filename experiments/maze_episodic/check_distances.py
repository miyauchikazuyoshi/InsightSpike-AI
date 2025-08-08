#!/usr/bin/env python3
"""Check distances"""

import numpy as np

# Initial episodes
episodes = [
    np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.0]),  # up
    np.array([0.5, 0.5, 0.25, 0.5, 0.5, 0.0]), # right
    np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0]),  # down
    np.array([0.5, 0.5, 0.75, 0.5, 0.5, 0.0]), # left
    np.array([0.6, 0.6, 0.5, 1.0, 1.0, 1.0])   # goal
]

# Query from (1,1) in 5x5 maze
query = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 1.0])

print("Distances from query:")
for i, ep in enumerate(episodes):
    dist = np.linalg.norm(query - ep)
    print(f"Episode {i}: {dist:.3f}")

print(f"\nDonut search with inner=0.1, outer=0.8:")
for i, ep in enumerate(episodes):
    dist = np.linalg.norm(query - ep)
    if 0.1 < dist <= 0.8:
        print(f"  Episode {i} included (dist={dist:.3f})")
    else:
        print(f"  Episode {i} excluded (dist={dist:.3f})")

# ゴール成分の影響を確認
print("\n\nゴール成分の影響:")
query_no_goal = query.copy()
query_no_goal[5] = 0.0
print(f"Query without goal: {query_no_goal}")

for i, ep in enumerate(episodes):
    dist = np.linalg.norm(query_no_goal - ep)
    print(f"Episode {i}: {dist:.3f}")