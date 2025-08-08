#!/usr/bin/env python3
"""Test with longer run time"""

from pure_episodic_donut import PureEpisodicDonutNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path

# 10x10 maze
maze = create_complex_maze(10, seed=42)
nav = PureEpisodicDonutNavigator(maze)

print("10x10 Maze Test - Extended Run")
print(f"Start: {nav.start_pos}, Goal: {nav.goal_pos}")

# Run for more steps
max_steps = 2000
result = nav.navigate(max_steps=max_steps)

print(f"\nResults:")
print(f"- Success: {result['success']}")
print(f"- Steps: {result['steps']}")
print(f"- Episodes: {result['episodes']}")
print(f"- Time: {result['time']:.2f}s")

if result['success']:
    visualize_maze_with_path(maze, result['path'], 'pure_episodic_visit_success.png')
    print("- Saved: pure_episodic_visit_success.png")
    
    # Analyze path efficiency
    optimal_dist = abs(nav.goal_pos[0] - nav.start_pos[0]) + abs(nav.goal_pos[1] - nav.start_pos[1])
    print(f"\n- Path length: {len(result['path'])}")
    print(f"- Optimal distance: {optimal_dist}")
    print(f"- Efficiency: {optimal_dist / len(result['path']) * 100:.1f}%")
else:
    # Show final position
    print(f"\n- Final position: {nav.position}")
    print(f"- Distance to goal: {abs(nav.position[0] - nav.goal_pos[0]) + abs(nav.position[1] - nav.goal_pos[1])}")

# Show exploration statistics
print(f"\nExploration statistics:")
print(f"- Unique positions visited: {len(nav.visit_counts)}")
print(f"- Total maze cells: {maze.shape[0] * maze.shape[1]}")
print(f"- Coverage: {len(nav.visit_counts) / (maze.shape[0] * maze.shape[1]) * 100:.1f}%")