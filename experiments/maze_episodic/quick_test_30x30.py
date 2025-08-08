#!/usr/bin/env python3
"""
Quick test on 30x30 to verify implementation
"""

from optimized_true_gedig import OptimizedTrueGeDIG
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import os
import time

# Test 30x30
size = 30
print(f"Testing optimized true geDIG on {size}×{size} maze...")

maze = create_complex_maze(size, seed=42)
nav = OptimizedTrueGeDIG(maze)

# Quick test with fewer steps
max_steps = 3000
start_time = time.time()

steps = 0
while nav.position != nav.goal and steps < max_steps:
    if steps % 500 == 0:
        dist = abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])
        print(f"Step {steps}: pos={nav.position}, dist={dist}, episodes={len(nav.episodes)}")
    
    action = nav.decide_action()
    
    old_pos = nav.position
    dx, dy = {'up': (0, -1), 'right': (1, 0), 
             'down': (0, 1), 'left': (-1, 0)}[action]
    new_pos = (nav.position[0] + dx, nav.position[1] + dy)
    
    result = 'wall'
    reached_goal = False
    
    if (0 <= new_pos[0] < nav.width and 
        0 <= new_pos[1] < nav.height and
        nav.maze[new_pos[1], new_pos[0]] == 0):
        
        if new_pos in nav.visited:
            result = 'visited'
        else:
            result = 'success'
        
        nav.position = new_pos
        nav.visited.add(new_pos)
        nav.path.append(new_pos)
        nav._update_visual_memory(new_pos[0], new_pos[1])
        
        if new_pos == nav.goal:
            reached_goal = True
    
    nav.add_episode(old_pos, action, result, reached_goal)
    steps += 1

elapsed = time.time() - start_time
success = nav.position == nav.goal

print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
print(f"Steps: {steps}")
print(f"Time: {elapsed:.2f}s")
print(f"Episodes: {len(nav.episodes)}")
print(f"Graph edges: {sum(len(neighbors) for neighbors in nav.graph.values()) // 2}")

if success:
    print(f"Efficiency: {steps / (2 * (size - 2)):.2f}x optimal")
    visualize_maze_with_path(maze, nav.path, 'test_30x30_true_gedig.png')