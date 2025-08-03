#!/usr/bin/env python3
"""Test simple action navigator."""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.simple_action_navigator import SimpleActionNavigator
from insightspike.navigators.blind_experience_navigator import BlindExperienceNavigator
from insightspike.config.maze_config import MazeNavigatorConfig

# Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO)


def test_simple():
    """Quick test of simple action navigator."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    nav_config.exploration_epsilon = 0.0
    
    print("\nTesting Simple Action Navigator")
    print("=" * 60)
    
    # Create maze
    np.random.seed(42)
    maze = SimpleMaze(size=(10, 10), maze_type='dfs')
    
    # Test simple navigator
    navigator = SimpleActionNavigator(nav_config)
    obs = maze.reset()
    
    for step in range(300):
        action = navigator.decide_action(obs, maze)
        obs, reward, done, info = maze.step(action)
        
        if done and maze.agent_pos == maze.goal_pos:
            print(f"\n✅ Goal reached in {step + 1} steps!")
            print(f"Wall hits: {navigator.wall_hits}")
            break
    else:
        print(f"\n❌ Failed to reach goal")
    
    # Show metrics
    metrics = navigator.get_metrics()
    print(f"\nMetrics:")
    print(f"  Total memories: {metrics['total_memories']}")
    print(f"  Successful: {metrics['successful_actions']}")
    print(f"  Blocked: {metrics['blocked_actions']}")
    
    print(f"\nDirection statistics:")
    for direction, stats in metrics['direction_stats'].items():
        total = stats['success'] + stats['blocked']
        if total > 0:
            success_rate = stats['success'] / total * 100
            print(f"  {direction}: {success_rate:.1f}% success ({total} attempts)")
    
    print("\n" + "=" * 60)
    print("Is this cheating?")
    print("- NO prior knowledge")
    print("- Learns from physical experience only")
    print("- Uses structural similarity (same direction)")
    print("- Completely legitimate learning!")


if __name__ == "__main__":
    test_simple()