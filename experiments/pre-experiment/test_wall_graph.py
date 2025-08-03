#!/usr/bin/env python3
"""Test wall graph navigator."""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.wall_graph_navigator import WallGraphNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def visualize_wall_graph(navigator, maze):
    """Visualize the discovered wall graph."""
    print("\nWall Graph Visualization:")
    print("="*50)
    
    # Create a visual representation
    visual = np.zeros_like(maze.grid, dtype=str)
    visual.fill(' ')
    
    # Mark paths
    for i in range(maze.grid.shape[0]):
        for j in range(maze.grid.shape[1]):
            if maze.grid[i, j] == 0:
                visual[i, j] = '.'
    
    # Mark discovered walls
    for wall_pos in navigator.known_walls:
        if 0 <= wall_pos[0] < visual.shape[0] and 0 <= wall_pos[1] < visual.shape[1]:
            visual[wall_pos[0], wall_pos[1]] = '#'
    
    # Mark visited positions
    for pos in navigator.visited_positions:
        if visual[pos[0], pos[1]] == '.':
            visual[pos[0], pos[1]] = '*'
    
    # Mark current position
    if navigator.current_position:
        visual[navigator.current_position[0], navigator.current_position[1]] = 'A'
    
    # Mark goal
    visual[maze.goal_pos[0], maze.goal_pos[1]] = 'G'
    
    # Print the visualization
    for row in visual:
        print(''.join(row))
    
    # Print graph statistics
    metrics = navigator.get_metrics()
    print(f"\nGraph Statistics:")
    print(f"  Walls discovered: {metrics['total_walls']}")
    print(f"  Wall connections: {metrics['total_edges']}")
    print(f"  Graph connectivity: {metrics['graph_connectivity']:.2f}")
    print(f"  Positions visited: {metrics['positions_visited']}")


def test_wall_graph_navigator():
    """Test the wall graph navigator."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create a DFS maze
    np.random.seed(42)
    maze = SimpleMaze(size=(15, 15), maze_type='dfs')
    
    print("Testing Wall Graph Navigator")
    print("="*50)
    print("Rule: Connect only newly discovered walls that are adjacent (distance=1)")
    print("="*50)
    
    # Create navigator
    navigator = WallGraphNavigator(nav_config)
    
    # Run for limited steps to see graph building
    obs = maze.reset()
    max_steps = 300
    
    for step in range(max_steps):
        # Track walls before action
        walls_before = len(navigator.known_walls)
        
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        # Track walls after action
        walls_after = len(navigator.known_walls)
        new_wall_count = walls_after - walls_before
        
        if new_wall_count > 0:
            print(f"\nStep {step}: Discovered {new_wall_count} new walls")
        
        if step % 20 == 0:
            print(f"\nStep {step}:")
            visualize_wall_graph(navigator, maze)
        
        if done:
            if maze.agent_pos == maze.goal_pos:
                print(f"\n🎯 GOAL REACHED in {step + 1} steps!")
                visualize_wall_graph(navigator, maze)
            else:
                print(f"\nEpisode ended (not at goal) in {step + 1} steps")
            break
    
    # Final visualization
    print(f"\nFinal state after {step + 1} steps:")
    visualize_wall_graph(navigator, maze)
    
    # Show some discovered edges
    if navigator.wall_edges:
        print("\nWall connections (showing all):")
        for i, (wall1, wall2) in enumerate(navigator.wall_edges):
            dist = abs(wall1[0] - wall2[0]) + abs(wall1[1] - wall2[1])
            print(f"  {wall1} <-> {wall2} (distance={dist})")


if __name__ == "__main__":
    test_wall_graph_navigator()