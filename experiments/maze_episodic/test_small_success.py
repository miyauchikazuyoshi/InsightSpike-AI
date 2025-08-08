#!/usr/bin/env python3
"""Test on small maze for success"""

from pure_episodic_donut import PureEpisodicDonutNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path

# 5x5 maze
maze = create_complex_maze(5, seed=42)
nav = PureEpisodicDonutNavigator(maze)

print("5x5 Maze Test")
print(f"Start: {nav.start_pos}, Goal: {nav.goal_pos}")

# Show maze
for y in range(5):
    for x in range(5):
        if (x, y) == nav.start_pos:
            print("S", end="")
        elif (x, y) == nav.goal_pos:
            print("G", end="")
        elif maze[y, x] == 0:
            print(".", end="")
        else:
            print("#", end="")
    print()

result = nav.navigate(max_steps=100)

print(f"\nResults:")
print(f"- Success: {result['success']}")
print(f"- Steps: {result['steps']}")
print(f"- Episodes: {result['episodes']}")

if result['success']:
    print("\n✓ SUCCESS! Pure geDIG with visit count works!")
    visualize_maze_with_path(maze, result['path'], 'pure_gedig_success_5x5.png')
else:
    print(f"\n- Final position: {nav.position}")
    print(f"- Distance to goal: {abs(nav.position[0] - nav.goal_pos[0]) + abs(nav.position[1] - nav.goal_pos[1])}")

# Graph analysis
print(f"\nGraph analysis:")
print(f"- Nodes: {nav.episode_graph.number_of_nodes()}")
print(f"- Edges: {nav.episode_graph.number_of_edges()}")
print(f"- Average degree: {2 * nav.episode_graph.number_of_edges() / nav.episode_graph.number_of_nodes():.2f}")