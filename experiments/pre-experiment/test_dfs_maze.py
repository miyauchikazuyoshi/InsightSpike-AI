#!/usr/bin/env python3
"""Test Pure geDIG on DFS maze with multiple episodes."""

import sys
from pathlib import Path
import yaml
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.pure_gediq_navigator import PureGeDIGNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def test_multiple_episodes():
    """Test Pure geDIG on the same maze for multiple episodes."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Create DFS maze with fixed seed for reproducibility
    np.random.seed(42)
    maze = SimpleMaze(size=(15, 15), maze_type='dfs')
    
    print("Testing Pure geDIG on 15x15 DFS maze")
    print("=" * 50)
    
    # Run multiple episodes
    num_episodes = 10
    results = []
    
    for episode in range(num_episodes):
        # Create new navigator for each episode (fresh start)
        navigator = PureGeDIGNavigator(nav_config)
        
        # Run episode
        obs = maze.reset()
        done = False
        steps = 0
        max_steps = 2000  # Give more time for complex maze
        
        while not done and steps < max_steps:
            action = navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            steps += 1
        
        success = done and maze.agent_pos == maze.goal_pos
        results.append({
            'episode': episode,
            'success': success,
            'steps': steps,
            'memory_nodes': len(navigator.memory_nodes)
        })
        
        print(f"Episode {episode + 1}: {'SUCCESS' if success else 'FAIL'} "
              f"in {steps} steps, {len(navigator.memory_nodes)} memory nodes")
    
    # Summary
    successes = [r['success'] for r in results]
    success_rate = sum(successes) / len(successes)
    avg_steps = np.mean([r['steps'] for r in results if r['success']]) if any(successes) else 0
    
    print("\n" + "=" * 50)
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average steps (successful runs): {avg_steps:.1f}")
    
    # Test with memory persistence across episodes
    print("\n\nTesting with memory persistence:")
    print("=" * 50)
    
    persistent_navigator = PureGeDIGNavigator(nav_config)
    persistent_results = []
    
    for episode in range(num_episodes):
        # Keep the same navigator (accumulate memory)
        obs = maze.reset()
        done = False
        steps = 0
        max_steps = 2000  # Give more time for complex maze
        
        while not done and steps < max_steps:
            action = persistent_navigator.decide_action(obs, maze)
            obs, reward, done, info = maze.step(action)
            steps += 1
        
        success = done and maze.agent_pos == maze.goal_pos
        persistent_results.append({
            'episode': episode,
            'success': success,
            'steps': steps,
            'memory_nodes': len(persistent_navigator.memory_nodes)
        })
        
        print(f"Episode {episode + 1}: {'SUCCESS' if success else 'FAIL'} "
              f"in {steps} steps, {len(persistent_navigator.memory_nodes)} total memory nodes")
    
    # Summary for persistent memory
    successes = [r['success'] for r in persistent_results]
    success_rate = sum(successes) / len(successes)
    avg_steps = np.mean([r['steps'] for r in persistent_results if r['success']]) if any(successes) else 0
    
    print("\n" + "=" * 50)
    print(f"Success rate (with memory): {success_rate:.1%}")
    print(f"Average steps (successful runs): {avg_steps:.1f}")
    print(f"Final memory size: {len(persistent_navigator.memory_nodes)} nodes")


if __name__ == "__main__":
    test_multiple_episodes()