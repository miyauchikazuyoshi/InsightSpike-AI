#!/usr/bin/env python3
"""Minimal test"""

import numpy as np

# Create simple 5x5 maze
maze = np.ones((5, 5), dtype=int)
# Clear path
maze[1, 1:4] = 0  # Horizontal path
maze[1:4, 3] = 0  # Vertical path

print("Maze:")
for y in range(5):
    for x in range(5):
        if (x, y) == (1, 1):
            print("S", end="")
        elif (x, y) == (3, 3):
            print("G", end="")
        elif maze[y, x] == 0:
            print(".", end="")
        else:
            print("#", end="")
    print()

from donut_gedig_navigator_simple import DonutGeDIGNavigator

nav = DonutGeDIGNavigator(maze, inner_radius=0.1, outer_radius=0.8)
nav.goal = (3, 3)  # Override goal

# Take 10 steps
for step in range(10):
    print(f"\nStep {step}: pos={nav.position}")
    
    # Get action
    action = nav.decide_action()
    print(f"Action: {action}")
    
    # Execute
    old_pos = nav.position
    dx, dy = {'up': (0, -1), 'right': (1, 0), 
             'down': (0, 1), 'left': (-1, 0)}[action]
    new_pos = (nav.position[0] + dx, nav.position[1] + dy)
    
    result = 'wall'
    if (0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5 and maze[new_pos[1], new_pos[0]] == 0):
        nav.position = new_pos
        nav.visited.add(new_pos)
        nav._update_visual_memory(new_pos[0], new_pos[1])
        result = 'success'
        print(f"Moved to {new_pos}")
    else:
        print("Hit wall")
    
    nav.add_episode(old_pos, action, result, new_pos == nav.goal)
    
    if nav.position == nav.goal:
        print("\nReached goal!")
        break