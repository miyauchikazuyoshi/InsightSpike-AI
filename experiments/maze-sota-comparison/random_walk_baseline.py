#!/usr/bin/env python3
"""
Random Walk Baseline for Maze Navigation
========================================

Tests how many steps random walk takes to solve the same mazes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from datetime import datetime
import json

class RandomWalkNavigator:
    """Pure random walk navigator for baseline comparison"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
    def get_valid_actions(self, pos: Tuple[int, int]) -> List[str]:
        """Get valid actions from current position"""
        x, y = pos
        valid_actions = []
        
        for action, (dx, dy) in [('up', (0, -1)), ('right', (1, 0)), 
                                 ('down', (0, 1)), ('left', (-1, 0))]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.maze[ny, nx] == 0):
                valid_actions.append(action)
        
        return valid_actions
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 50000) -> Dict:
        """Navigate using pure random walk"""
        
        print(f"\nRandom Walk Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        start_time = time.time()
        
        # Track walkable cells for coverage
        walkable_cells = sum(1 for y in range(self.height) 
                           for x in range(self.width) 
                           if self.maze[y, x] == 0)
        
        while self.position != goal and steps < max_steps:
            # Progress report
            if steps % 5000 == 0 and steps > 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                coverage = len(self.visited) / walkable_cells * 100
                print(f"Step {steps}: pos={self.position}, dist={dist}, coverage={coverage:.1f}%")
            
            # Get valid actions
            valid_actions = self.get_valid_actions(self.position)
            
            if not valid_actions:
                # Dead end - shouldn't happen in a proper maze
                print("Dead end reached!")
                break
            
            # Random choice
            action = np.random.choice(valid_actions)
            
            # Execute action
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            self.position = (self.position[0] + dx, self.position[1] + dy)
            self.visited.add(self.position)
            self.path.append(self.position)
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        coverage = len(self.visited) / walkable_cells * 100
        
        print(f"\nNavigation complete: success={success}, steps={steps}")
        print(f"Coverage: {coverage:.1f}%, Time: {elapsed_time:.2f}s")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'coverage': coverage,
            'elapsed_time': elapsed_time
        }


def test_random_walk_baseline(num_runs=10):
    """Test random walk on multiple maze instances"""
    
    print("="*70)
    print("RANDOM WALK BASELINE TEST")
    print("="*70)
    
    all_results = []
    
    for run in range(num_runs):
        print(f"\n{'='*30} RUN {run + 1}/{num_runs} {'='*30}")
        
        # Generate maze with same algorithm as multi-hop tests
        np.random.seed(42 + run)
        size = 50
        maze = np.ones((size, size), dtype=int)
        
        # Recursive backtracker
        def carve_passages(cx, cy):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            np.random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size and maze[ny, nx] == 1:
                    maze[cy + dy // 2, cx + dx // 2] = 0
                    maze[ny, nx] = 0
                    carve_passages(nx, ny)
        
        maze[1, 1] = 0
        carve_passages(1, 1)
        maze[size-2, size-2] = 0
        
        # Add loops
        for _ in range(size):
            x = np.random.randint(2, size-2)
            y = np.random.randint(2, size-2)
            if maze[y, x] == 1:
                neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                              if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
                if neighbors >= 2:
                    maze[y, x] = 0
        
        goal = (size-2, size-2)
        
        # Run random walk
        navigator = RandomWalkNavigator(maze)
        result = navigator.navigate(goal, max_steps=50000)
        all_results.append(result)
    
    # Calculate statistics
    successful_runs = [r for r in all_results if r['success']]
    
    print("\n" + "="*70)
    print("RANDOM WALK RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nSuccess rate: {len(successful_runs)}/{num_runs} = {len(successful_runs)/num_runs*100:.1f}%")
    
    if successful_runs:
        avg_steps = np.mean([r['steps'] for r in successful_runs])
        std_steps = np.std([r['steps'] for r in successful_runs])
        min_steps = min(r['steps'] for r in successful_runs)
        max_steps = max(r['steps'] for r in successful_runs)
        
        print(f"\nFor successful runs:")
        print(f"  Average steps: {avg_steps:.1f} ± {std_steps:.1f}")
        print(f"  Min steps: {min_steps}")
        print(f"  Max steps: {max_steps}")
        
        avg_coverage = np.mean([r['coverage'] for r in successful_runs])
        print(f"  Average coverage: {avg_coverage:.1f}%")
    
    # Display all results
    print("\n| Run | Success | Steps  | Coverage |")
    print("|-----|---------|--------|----------|")
    for i, result in enumerate(all_results):
        print(f"| {i+1:3} | {str(result['success']):7} | {result['steps']:6} | {result['coverage']:7.1f}% |")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'random_walk_baseline_{timestamp}.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_runs': num_runs,
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to random_walk_baseline_{timestamp}.json")
    
    return all_results


if __name__ == "__main__":
    results = test_random_walk_baseline(num_runs=10)