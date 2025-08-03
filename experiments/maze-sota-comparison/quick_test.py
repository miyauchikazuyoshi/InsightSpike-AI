#!/usr/bin/env python3
"""Quick test of geDIG vs baselines."""

import sys
from pathlib import Path
import numpy as np
import time

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.navigators.blind_experience_navigator import BlindExperienceNavigator
from insightspike.navigators.experience_memory_navigator import ExperienceMemoryNavigator
from insightspike.config.maze_config import MazeNavigatorConfig


def random_walk(maze, max_steps=1000):
    """Simple random walk baseline."""
    maze.reset()
    steps = 0
    
    while steps < max_steps:
        action = np.random.randint(0, 4)
        obs, reward, done, info = maze.step(action)
        steps += 1
        
        if done and maze.agent_pos == maze.goal_pos:
            return steps
    
    return max_steps


def bfs_solve(maze):
    """BFS solution (oracle - knows full maze)."""
    from collections import deque
    
    start = maze.start_pos
    goal = maze.goal_pos
    grid = maze.grid
    
    # BFS
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        (x, y), path = queue.popleft()
        
        if (x, y) == goal:
            return len(path) - 1
        
        if (x, y) in visited:
            continue
            
        visited.add((x, y))
        
        # Check 4 directions
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and
                grid[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return float('inf')


def test_comparison():
    """Quick comparison test."""
    # Config
    config = {
        'ged_weight': 1.0,
        'ig_weight': 2.0,
        'temperature': 1.0,
        'exploration_epsilon': 0.0
    }
    nav_config = MazeNavigatorConfig(**config)
    
    # Test different maze sizes
    maze_configs = [
        (10, 10, 'dfs'),
        (15, 15, 'rooms'),
        (20, 20, 'spiral'),
    ]
    
    print("QUICK COMPARISON TEST")
    print("=" * 80)
    
    for size_x, size_y, maze_type in maze_configs:
        print(f"\n{size_x}x{size_y} {maze_type} maze:")
        print("-" * 40)
        
        results = {}
        
        # Run multiple trials
        n_trials = 10
        for trial in range(n_trials):
            seed = 42 + trial * 10
            np.random.seed(seed)
            
            # Create maze
            maze = SimpleMaze(size=(size_x, size_y), maze_type=maze_type)
            
            # BFS (optimal)
            bfs_steps = bfs_solve(maze)
            if 'bfs' not in results:
                results['bfs'] = []
            results['bfs'].append(bfs_steps)
            
            # Random walk
            maze_copy = SimpleMaze(size=(size_x, size_y), maze_type=maze_type)
            maze_copy.grid = maze.grid.copy()
            maze_copy.start_pos = maze.start_pos
            maze_copy.goal_pos = maze.goal_pos
            random_steps = random_walk(maze_copy)
            if 'random' not in results:
                results['random'] = []
            results['random'].append(random_steps)
            
            # geDIG blind
            blind_nav = BlindExperienceNavigator(nav_config)
            obs = maze.reset()
            
            for step in range(1000):
                action = blind_nav.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                
                if done and maze.agent_pos == maze.goal_pos:
                    if 'gedig_blind' not in results:
                        results['gedig_blind'] = []
                    results['gedig_blind'].append(step + 1)
                    break
            
            # geDIG visual
            visual_nav = ExperienceMemoryNavigator(nav_config)
            obs = maze.reset()
            
            for step in range(1000):
                action = visual_nav.decide_action(obs, maze)
                obs, reward, done, info = maze.step(action)
                
                if done and maze.agent_pos == maze.goal_pos:
                    if 'gedig_visual' not in results:
                        results['gedig_visual'] = []
                    results['gedig_visual'].append(step + 1)
                    break
        
        # Print results
        for method, steps_list in results.items():
            avg_steps = np.mean(steps_list)
            std_steps = np.std(steps_list)
            print(f"  {method:12}: {avg_steps:6.1f} ± {std_steps:5.1f} steps")
        
        # Calculate speedups
        if 'random' in results and len(results['random']) > 0:
            random_avg = np.mean(results['random'])
            print("\n  Speedup vs random walk:")
            for method in ['bfs', 'gedig_blind', 'gedig_visual']:
                if method in results and len(results[method]) > 0:
                    speedup = random_avg / np.mean(results[method])
                    print(f"    {method:12}: {speedup:5.1f}x")


if __name__ == "__main__":
    test_comparison()