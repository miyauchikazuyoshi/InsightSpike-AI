#!/usr/bin/env python3
"""
Verify TopK concept works
"""

import numpy as np
import time

# Simple test data
print("Testing TopK concept...")

# Create dummy episodes
episodes = []
for i in range(1000):
    x = np.random.randint(0, 50)
    y = np.random.randint(0, 50)
    episodes.append({
        'id': i,
        'pos': (x, y),
        'reached_goal': (x == 48 and y == 48)
    })

# Test query
query_pos = (25, 25)
k = 50

# Method 1: Original (all episodes)
start = time.time()
all_episodes = episodes.copy()
time1 = time.time() - start

# Method 2: TopK
start = time.time()
distances = []
for ep in episodes:
    dist = abs(ep['pos'][0] - query_pos[0]) + abs(ep['pos'][1] - query_pos[1])
    distances.append((dist, ep))

distances.sort(key=lambda x: x[0])
topk = [ep for _, ep in distances[:k]]
time2 = time.time() - start

print(f"\nResults:")
print(f"All episodes: {len(all_episodes)}, time: {time1*1000:.2f}ms")
print(f"TopK episodes: {len(topk)}, time: {time2*1000:.2f}ms")
print(f"Speedup: {time1/time2:.2f}x")

# Check if any goal episodes in TopK
goal_in_topk = any(ep['reached_goal'] for ep in topk)
print(f"\nGoal episode in TopK: {goal_in_topk}")

# Show nearest episodes
print(f"\nNearest 5 episodes to {query_pos}:")
for i in range(min(5, len(topk))):
    ep = topk[i]
    dist = abs(ep['pos'][0] - query_pos[0]) + abs(ep['pos'][1] - query_pos[1])
    print(f"  {ep['pos']}, dist={dist}, goal={ep['reached_goal']}")

# Now test on small maze
print("\n" + "="*60)
print("Testing on 15x15 maze...")

from pure_episodic_navigator import create_complex_maze

class MinimalTopKNavigator:
    def __init__(self, maze, k=30):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.episodes = []
        self.k = k
        self.steps = 0
        
    def navigate_simple(self, max_steps=1000):
        """Super simple navigation"""
        while self.position != self.goal and self.steps < max_steps:
            # Random walk with bias toward unvisited
            actions = []
            for action, (dx, dy) in [('up', (0, -1)), ('right', (1, 0)), 
                                    ('down', (0, 1)), ('left', (-1, 0))]:
                nx = self.position[0] + dx
                ny = self.position[1] + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.maze[ny, nx] == 0):
                    # Prefer unvisited
                    weight = 10 if (nx, ny) not in self.visited else 1
                    actions.extend([action] * weight)
            
            if actions:
                action = np.random.choice(actions)
                dx, dy = {'up': (0, -1), 'right': (1, 0), 
                         'down': (0, 1), 'left': (-1, 0)}[action]
                new_pos = (self.position[0] + dx, self.position[1] + dy)
                
                self.position = new_pos
                self.visited.add(new_pos)
                self.episodes.append({
                    'pos': new_pos,
                    'reached_goal': new_pos == self.goal
                })
            
            self.steps += 1
            
            if self.steps % 100 == 0:
                dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
                print(f"Step {self.steps}: pos={self.position}, dist={dist}")
        
        return self.position == self.goal

# Test
maze = create_complex_maze(15, seed=42)
nav = MinimalTopKNavigator(maze)
success = nav.navigate_simple()

print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
print(f"Steps: {nav.steps}")
print(f"Episodes: {len(nav.episodes)}")