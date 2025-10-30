#!/usr/bin/env python3
"""
Test geDIG success rate on different maze complexities
"""

import sys
import os
import numpy as np
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from navigation.maze_navigator import MazeNavigator

def create_simple_maze_15x15():
    """Simple 15x15 maze with few obstacles"""
    maze = np.ones((15, 15), dtype=int)
    # Create basic corridors
    maze[1, 1:14] = 0  # top corridor
    maze[1:14, 13] = 0  # right corridor
    maze[13, 1:14] = 0  # bottom corridor
    maze[1:14, 1] = 0  # left corridor
    # Add some internal paths
    maze[7, 3:12] = 0  # horizontal middle
    maze[3:12, 7] = 0  # vertical middle
    return maze

def create_complex_maze_25x25():
    """Complex 25x25 maze with many dead ends"""
    maze = np.ones((25, 25), dtype=int)
    
    # Create a more complex pattern
    for i in range(1, 24, 2):
        maze[i, 1:24] = 0  # horizontal corridors
    for j in range(1, 24, 3):
        maze[1:24, j] = 0  # vertical corridors
    
    # Add dead ends
    for _ in range(20):
        x = random.randint(2, 22)
        y = random.randint(2, 22)
        if maze[y, x] == 0:
            # Create dead end branch
            if random.random() > 0.5:
                maze[y, x+1:min(24, x+3)] = 0
            else:
                maze[y+1:min(24, y+3), x] = 0
    
    return maze

def test_success_rate(maze_fn, maze_name, n_trials=10):
    """Test success rate for both strategies"""
    
    print(f"\n{maze_name} Results:")
    print("-" * 40)
    
    results = {'simple': [], 'gedig': []}
    
    for trial in range(n_trials):
        seed = 100 + trial
        random.seed(seed)
        np.random.seed(seed)
        
        maze = maze_fn()
        start = (1, 1)
        goal = (maze.shape[0]-2, maze.shape[1]-2)
        
        # Ensure start and goal are open
        maze[start[1], start[0]] = 0
        maze[goal[1], goal[0]] = 0
        
        for strategy in ['simple', 'gedig']:
            nav = MazeNavigator(
                maze=maze,
                start_pos=start,
                goal_pos=goal,
                wiring_strategy=strategy,
                gedig_threshold=-0.08,
                backtrack_threshold=-0.2,
                simple_mode=True
            )
            
            max_steps = maze.shape[0] * maze.shape[1] * 2
            for step in range(max_steps):
                action = nav.step()
                if nav.current_pos == goal:
                    results[strategy].append({'success': True, 'steps': step+1})
                    break
            else:
                results[strategy].append({'success': False, 'steps': max_steps})
    
    # Calculate success rates
    for strategy in ['simple', 'gedig']:
        successes = sum(1 for r in results[strategy] if r['success'])
        success_rate = successes / n_trials * 100
        
        successful_runs = [r['steps'] for r in results[strategy] if r['success']]
        avg_steps = np.mean(successful_runs) if successful_runs else float('inf')
        
        print(f"{strategy:8}: {success_rate:5.1f}% success rate", end='')
        if success_rate > 0:
            print(f", avg {avg_steps:.0f} steps")
        else:
            print(" (no successful runs)")
    
    return results

def main():
    print("=" * 60)
    print("geDIG Success Rate Test")
    print("=" * 60)
    
    # Test on different maze complexities
    test_success_rate(create_simple_maze_15x15, "Simple 15x15 Maze", n_trials=5)
    test_success_rate(create_complex_maze_25x25, "Complex 25x25 Maze", n_trials=5)
    
    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)
    print("geDIG performance depends on maze complexity and structure.")
    print("For very complex mazes, both strategies may struggle without")
    print("proper threshold tuning and sufficient exploration steps.")

if __name__ == '__main__':
    main()