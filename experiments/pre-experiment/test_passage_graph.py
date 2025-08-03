#!/usr/bin/env python3
"""Test passage graph navigator."""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.passage_graph_navigator import PassageGraphNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def test_passage_navigator():
    """Test passage-based navigation."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create a small maze for testing
    np.random.seed(42)
    maze = SimpleMaze(size=(11, 11), maze_type='dfs')
    
    print("Testing Passage Graph Navigator")
    print("=" * 50)
    print("Treating passable directions as queries")
    print(f"Maze size: {maze.size}, Goal at: {maze.goal_pos}")
    print("=" * 50)
    
    # Create navigator
    navigator = PassageGraphNavigator(nav_config)
    
    # Run navigation
    obs = maze.reset()
    max_steps = 500
    
    for step in range(max_steps):
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        if done:
            if maze.agent_pos == maze.goal_pos:
                print(f"\n🎯 GOAL REACHED in {step + 1} steps!")
                
                # Show metrics
                metrics = navigator.get_metrics()
                print(f"\nFinal Metrics:")
                print(f"  Total passages discovered: {metrics['total_passages']}")
                print(f"  Passage connections: {metrics['total_connections']}")
                print(f"  Positions visited: {metrics['positions_visited']}")
                print(f"  Graph connectivity: {metrics['graph_connectivity']:.2f}")
            break
    
    if step == max_steps - 1:
        print(f"\nReached max steps ({max_steps})")
        print(f"Final position: {navigator.current_position}")
        print(f"Goal position: {maze.goal_pos}")
        print("\nMaze visualization:")
        print(maze.render())


if __name__ == "__main__":
    test_passage_navigator()