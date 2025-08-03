#!/usr/bin/env python3
"""Test experience memory navigator."""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def test_experience_navigator():
    """Test navigator with visual and physical memory."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=(15, 15), maze_type='dfs')
    
    print("Testing Experience Memory Navigator")
    print("=" * 50)
    print("Combining visual (what we see) and physical (what we experience) memory")
    print(f"Maze size: {maze.size}, Goal at: {maze.goal_pos}")
    print("=" * 50)
    
    # Create navigator
    navigator = ExperienceMemoryNavigator(nav_config)
    
    # Run navigation
    obs = maze.reset()
    max_steps = 300
    
    for step in range(max_steps):
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        # Visualize every 20 steps
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(maze.render())
        
        if done:
            if maze.agent_pos == maze.goal_pos:
                print(f"\n🎯 GOAL REACHED in {step + 1} steps!")
                
                # Show metrics
                metrics = navigator.get_metrics()
                print(f"\nFinal Metrics:")
                print(f"  Total positions visited: {metrics['total_positions']}")
                print(f"  Visual walls detected: {metrics['total_visual_walls']}")
                print(f"  Physical blocks experienced: {metrics['total_physical_blocks']}")
                print(f"  Visual-Physical mismatches: {metrics['visual_physical_mismatches']}")
                
                # Show some interesting experiences
                print("\nInteresting experiences:")
                count = 0
                for pos, node in navigator.memory_nodes.items():
                    for direction, exp in node.experiences.items():
                        if exp.physical != exp.visual and exp.attempts > 0:
                            print(f"  Position {pos}, {['up', 'right', 'down', 'left'][direction]}: "
                                  f"saw {exp.visual.value}, experienced {exp.physical.value}")
                            count += 1
                            if count >= 5:
                                break
                    if count >= 5:
                        break
            break
    
    if step == max_steps - 1:
        print(f"\nReached max steps ({max_steps})")
        metrics = navigator.get_metrics()
        print(f"Final position: {navigator.current_position}")
        print(f"Goal position: {maze.goal_pos}")
        print(f"\nMetrics:")
        print(f"  Total positions visited: {metrics['total_positions']}")
        print(f"  Visual walls detected: {metrics['total_visual_walls']}")
        print(f"  Physical blocks experienced: {metrics['total_physical_blocks']}")


if __name__ == "__main__":
    test_experience_navigator()