#!/usr/bin/env python3
"""Test wall-only memory approach."""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.pure_gediq_navigator import PureGeDIGNavigator
from insightspike.navigators.wall_only_gediq_navigator import WallOnlyGeDIGNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def compare_navigators():
    """Compare Pure geDIG vs Wall-only geDIG."""
    # Load config
    config_path = Path(__file__).parent / "maze_experiment_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    nav_config = MazeNavigatorConfig(**config_dict['navigator'])
    
    # Test on different maze sizes
    for maze_size in [(11, 11), (15, 15), (21, 21)]:
        print(f"\n{'='*60}")
        print(f"Testing on {maze_size[0]}x{maze_size[1]} DFS maze")
        print(f"{'='*60}")
        
        # Create maze with fixed seed
        np.random.seed(42)
        maze = SimpleMaze(size=maze_size, maze_type='dfs')
        
        # Test Pure geDIG
        print("\nPure geDIG (memorizes all positions):")
        pure_results = []
        
        for episode in range(5):
            navigator = PureGeDIGNavigator(nav_config)
            obs = maze.reset()
            done = False
            steps = 0
            max_steps = 2000
            
            while not done and steps < max_steps:
                action = navigator.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                steps += 1
            
            success = done and maze.agent_pos == maze.goal_pos
            pure_results.append({
                'success': success,
                'steps': steps,
                'memory_size': len(navigator.memory_nodes)
            })
            
            print(f"  Episode {episode + 1}: {'SUCCESS' if success else 'FAIL'} "
                  f"in {steps} steps, {len(navigator.memory_nodes)} memory nodes")
        
        # Test Wall-only geDIG
        print("\nWall-only geDIG (memorizes only walls):")
        wall_results = []
        
        # Create persistent navigator for wall memory
        wall_navigator = WallOnlyGeDIGNavigator(nav_config)
        
        for episode in range(5):
            wall_navigator.new_episode()  # Reset position tracking but keep wall memory
            obs = maze.reset()
            done = False
            steps = 0
            max_steps = 2000
            
            while not done and steps < max_steps:
                action = wall_navigator.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                steps += 1
            
            success = done and maze.agent_pos == maze.goal_pos
            metrics = wall_navigator.get_metrics()
            wall_results.append({
                'success': success,
                'steps': steps,
                'walls_discovered': metrics['total_walls_discovered'],
                'positions_visited': metrics['total_positions_visited']
            })
            
            print(f"  Episode {episode + 1}: {'SUCCESS' if success else 'FAIL'} "
                  f"in {steps} steps, {metrics['total_walls_discovered']} walls discovered, "
                  f"{metrics['total_positions_visited']} positions visited")
        
        # Summary
        print("\nSummary:")
        pure_success = sum(r['success'] for r in pure_results) / len(pure_results)
        wall_success = sum(r['success'] for r in wall_results) / len(wall_results)
        
        print(f"  Pure geDIG success rate: {pure_success:.0%}")
        print(f"  Wall-only success rate: {wall_success:.0%}")
        
        if any(r['success'] for r in pure_results):
            avg_memory = np.mean([r['memory_size'] for r in pure_results if r['success']])
            print(f"  Pure geDIG avg memory size: {avg_memory:.0f} nodes")
        
        if wall_results:
            print(f"  Wall-only final walls discovered: {wall_results[-1]['walls_discovered']}")
            
        # Calculate theoretical minimum walls
        total_cells = maze_size[0] * maze_size[1]
        wall_cells = np.sum(maze.grid == 1)
        path_cells = total_cells - wall_cells
        print(f"\n  Maze statistics:")
        print(f"    Total cells: {total_cells}")
        print(f"    Wall cells: {wall_cells}")
        print(f"    Path cells: {path_cells}")


if __name__ == "__main__":
    compare_navigators()