#!/usr/bin/env python3
"""Quick test to show vector search works"""

from donut_gedig_navigator_simple import DonutGeDIGNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import numpy as np

# Test on 10x10 first
size = 10
maze = create_complex_maze(size, seed=42)

# Show maze
print(f"Testing {size}×{size} maze:")
for y in range(size):
    for x in range(size):
        if (x, y) == (1, 1):
            print("S", end="")
        elif (x, y) == (size-2, size-2):
            print("G", end="")
        elif maze[y, x] == 0:
            print(".", end="")
        else:
            print("#", end="")
    print()

nav = DonutGeDIGNavigator(maze, inner_radius=0.05, outer_radius=0.5)

# Navigate with detailed output
steps = 0
max_steps = 1000

print(f"\nStarting navigation...")
print(f"Start: {nav.position}, Goal: {nav.goal}")

while nav.position != nav.goal and steps < max_steps:
    if steps % 50 == 0:
        dist = abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])
        print(f"Step {steps}: pos={nav.position}, dist={dist}, episodes={len(nav.episodes)}")
    
    # Navigate
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
    
    if reached_goal:
        break

if nav.position == nav.goal:
    print(f"\n✓ SUCCESS! Reached goal in {steps} steps")
    print(f"Path length: {len(nav.path)}")
    print(f"Wall hits: {nav.wall_hits}")
    
    # Show final maze with path
    print("\nFinal path:")
    for y in range(size):
        for x in range(size):
            if (x, y) in nav.path:
                print("*", end="")
            elif maze[y, x] == 0:
                print(".", end="")
            else:
                print("#", end="")
        print()
    
    visualize_maze_with_path(maze, nav.path, 'vector_10x10_success.png')
else:
    print(f"\n✗ Failed after {steps} steps")
    print(f"Final position: {nav.position}")
    print(f"Distance to goal: {abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])}")