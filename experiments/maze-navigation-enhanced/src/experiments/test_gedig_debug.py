#!/usr/bin/env python3
"""Debug version of geDIG experiment - with timeout and progress tracking"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator
import random


def generate_simple_maze(size: int) -> np.ndarray:
    """Generate a simple maze."""
    maze = np.ones((size, size), dtype=int)
    
    # Create simple path from start to goal
    # Horizontal corridor
    for x in range(1, size - 1):
        maze[size // 2, x] = 0
    
    # Vertical corridors at both ends
    for y in range(1, size - 1):
        maze[y, 1] = 0
        maze[y, size - 2] = 0
    
    return maze


def test_single_config():
    """Test a single configuration to debug the issue."""
    
    print("Creating 15x15 maze...")
    maze = generate_simple_maze(15)
    start = (1, 1)
    goal = (13, 13)
    
    # Ensure goal is reachable
    maze[goal[0], goal[1]] = 0
    
    print(f"Maze created. Start: {start}, Goal: {goal}")
    print("Testing 'simple' strategy...")
    
    try:
        nav_simple = MazeNavigator(
            maze=maze,
            start_pos=start,
            goal_pos=goal,
            wiring_strategy='simple',
            gedig_threshold=-0.5,
            backtrack_threshold=-0.3,
            simple_mode=True,
            backtrack_debounce=True
        )
        
        steps = 0
        max_steps = 200
        path = []
        
        print("Starting navigation...")
        start_time = time.time()
        
        while steps < max_steps:
            if steps % 20 == 0:
                print(f"  Step {steps}: pos={nav_simple.current_pos}")
            
            action = nav_simple.step()
            path.append(nav_simple.current_pos)
            steps += 1
            
            if nav_simple.current_pos == goal:
                print(f"✓ Goal reached in {steps} steps!")
                break
            
            # Timeout after 10 seconds per maze
            if time.time() - start_time > 10:
                print(f"✗ Timeout after {steps} steps")
                break
        
        if nav_simple.current_pos != goal:
            print(f"✗ Failed to reach goal after {steps} steps")
        
        print(f"Final position: {nav_simple.current_pos}")
        print(f"Unique positions visited: {len(set(path))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nNow testing 'gedig' strategy...")
    
    try:
        nav_gedig = MazeNavigator(
            maze=maze,
            start_pos=start,
            goal_pos=goal,
            wiring_strategy='gedig',  # Use gedig!
            gedig_threshold=-0.5,
            backtrack_threshold=-0.3,
            simple_mode=True,
            backtrack_debounce=True
        )
        
        steps = 0
        path = []
        
        print("Starting navigation...")
        start_time = time.time()
        
        while steps < max_steps:
            if steps % 20 == 0:
                print(f"  Step {steps}: pos={nav_gedig.current_pos}")
            
            action = nav_gedig.step()
            path.append(nav_gedig.current_pos)
            steps += 1
            
            if nav_gedig.current_pos == goal:
                print(f"✓ Goal reached in {steps} steps!")
                break
            
            # Timeout after 10 seconds per maze
            if time.time() - start_time > 10:
                print(f"✗ Timeout after {steps} steps")
                break
        
        if nav_gedig.current_pos != goal:
            print(f"✗ Failed to reach goal after {steps} steps")
        
        print(f"Final position: {nav_gedig.current_pos}")
        print(f"Unique positions visited: {len(set(path))}")
        
        # Check if geDIG was actually computed
        gedig_history = getattr(nav_gedig, 'gedig_history', [])
        print(f"geDIG values computed: {len(gedig_history)}")
        if gedig_history:
            print(f"  Mean: {np.mean(gedig_history):.3f}")
            print(f"  Min: {np.min(gedig_history):.3f}")
            print(f"  Max: {np.max(gedig_history):.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_single_config()