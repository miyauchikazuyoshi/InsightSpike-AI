#!/usr/bin/env python3
"""Test visit count dimension"""

from pure_episodic_donut import PureEpisodicDonutNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path

# Small maze first
maze = create_complex_maze(10, seed=42)
nav = PureEpisodicDonutNavigator(maze)

print("10x10 Maze Test with Visit Count")
print(f"Start: {nav.start_pos}, Goal: {nav.goal_pos}")

# Run for limited steps
max_steps = 500
result = nav.navigate(max_steps=max_steps)

print(f"\nResults:")
print(f"- Success: {result['success']}")
print(f"- Steps: {result['steps']}")
print(f"- Episodes: {result['episodes']}")
print(f"- Time: {result['time']:.2f}s")

if result['success']:
    visualize_maze_with_path(maze, result['path'], 'pure_episodic_visit_10x10.png')
    print("- Saved: pure_episodic_visit_10x10.png")

# Show visit counts
print(f"\nTop 10 most visited positions:")
sorted_visits = sorted(nav.visit_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for pos_str, count in sorted_visits:
    print(f"  {pos_str}: {count} visits")