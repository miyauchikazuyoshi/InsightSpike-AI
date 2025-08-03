#!/usr/bin/env python3
"""Check maze layout and pathfinding."""

from insightspike.environments.maze import SimpleMaze
import numpy as np

# Create maze and visualize
maze = SimpleMaze(size=(20, 20))
print('Maze layout:')
print(maze.render('ascii'))
print(f'\nStart: {maze.start_pos}')
print(f'Goal: {maze.goal_pos}')

# Check if there's a clear path
print('\nChecking path from start to goal...')
# Simple check: can we reach goal area?
grid = maze.grid
goal_clear = grid[maze.goal_pos] != 1
print(f'Goal area clear? {goal_clear}')
print(f'Area around goal:')
for i in range(-1, 2):
    for j in range(-1, 2):
        pos = (maze.goal_pos[0] + i, maze.goal_pos[1] + j)
        if 0 <= pos[0] < 20 and 0 <= pos[1] < 20:
            is_wall = grid[pos] == 1
            print(f'  {pos}: {"WALL" if is_wall else "CLEAR"}')

# Try a simple path
print('\n\nTrying to find path manually...')
# Check if we can go right then down
pos = maze.start_pos
print(f'Starting at {pos}')

# Can we go right?
right_pos = (pos[0], pos[1] + 1)
print(f'Right to {right_pos}: {"WALL" if grid[right_pos] == 1 else "CLEAR"}')

# Can we go down?
down_pos = (pos[0] + 1, pos[1])
print(f'Down to {down_pos}: {"WALL" if grid[down_pos] == 1 else "CLEAR"}')