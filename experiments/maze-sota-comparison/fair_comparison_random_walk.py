#!/usr/bin/env python3
"""
Fair Comparison: Random Walk with Wall Avoidance
================================================

Compares our multi-hop navigator with a random walk that also avoids walls.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from datetime import datetime
import json

class SmartRandomWalkNavigator:
    """Random walk that avoids walls (like our current implementation)"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
    def get_valid_actions(self, pos: Tuple[int, int]) -> List[str]:
        """Get valid actions that don't hit walls"""
        x, y = pos
        valid_actions = []
        
        for action, (dx, dy) in [('up', (0, -1)), ('right', (1, 0)), 
                                 ('down', (0, 1)), ('left', (-1, 0))]:
            nx, ny = x + dx, y + dy
            # 壁を避ける（現在の実装と同じ）
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.maze[ny, nx] == 0):
                valid_actions.append(action)
        
        return valid_actions
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 5000) -> Dict:
        """Navigate using smart random walk (wall-avoiding)"""
        
        steps = 0
        backtrack_count = 0
        start_time = time.time()
        
        # Track walkable cells
        walkable_cells = sum(1 for y in range(self.height) 
                           for x in range(self.width) 
                           if self.maze[y, x] == 0)
        
        while self.position != goal and steps < max_steps:
            # Get valid actions
            valid_actions = self.get_valid_actions(self.position)
            
            if not valid_actions:
                # Dead end - backtrack
                backtrack_count += 1
                if len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
            else:
                # Random choice among valid actions
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
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'coverage': coverage,
            'backtrack_count': backtrack_count,
            'elapsed_time': elapsed_time
        }


def compare_with_smart_random_walk(num_runs=10):
    """Compare multi-hop with smart random walk"""
    
    print("="*70)
    print("FAIR COMPARISON: Multi-hop vs Smart Random Walk")
    print("="*70)
    
    # Import our multi-hop navigator
    from multihop_gedig_comparison import MultiHopGeDIGNavigator
    
    results = {
        'random_walk': [],
        '1-hop': [],
        '2-hop': [],
        '3-hop': []
    }
    
    for run in range(num_runs):
        print(f"\n{'='*30} RUN {run + 1}/{num_runs} {'='*30}")
        
        # Generate maze
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
        
        # Test smart random walk
        print("Smart Random Walk...", end='', flush=True)
        rw_nav = SmartRandomWalkNavigator(maze)
        rw_result = rw_nav.navigate(goal)
        results['random_walk'].append(rw_result)
        print(f" {'SUCCESS' if rw_result['success'] else 'FAIL'} in {rw_result['steps']} steps")
        
        # Test multi-hop
        for hop_count in [1, 2, 3]:
            print(f"{hop_count}-hop...", end='', flush=True)
            mh_nav = MultiHopGeDIGNavigator(maze, hop_count=hop_count)
            mh_result = mh_nav.navigate(goal, max_steps=5000)
            results[f'{hop_count}-hop'].append(mh_result)
            print(f" {'SUCCESS' if mh_result['success'] else 'FAIL'} in {mh_result['steps']} steps")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Calculate statistics
    stats = {}
    for method in results:
        method_results = results[method]
        successful = [r for r in method_results if r['success']]
        
        stats[method] = {
            'success_rate': len(successful) / num_runs * 100,
            'avg_steps_all': np.mean([r['steps'] for r in method_results]),
            'avg_steps_success': np.mean([r['steps'] for r in successful]) if successful else None,
            'avg_coverage': np.mean([r['coverage'] for r in method_results])
        }
    
    # Display comparison
    print("\n| Method      | Success% | Avg Steps (all) | Avg Steps (success) | Avg Coverage% |")
    print("|-------------|----------|-----------------|---------------------|---------------|")
    
    for method, stat in stats.items():
        success_steps = f"{stat['avg_steps_success']:.0f}" if stat['avg_steps_success'] else "N/A"
        print(f"| {method:11} | {stat['success_rate']:7.0f}% | {stat['avg_steps_all']:15.0f} | "
              f"{success_steps:19} | {stat['avg_coverage']:13.1f} |")
    
    # Statistical test: are multi-hop methods significantly different from random walk?
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    rw_steps = [r['steps'] for r in results['random_walk']]
    
    for method in ['1-hop', '2-hop', '3-hop']:
        method_steps = [r['steps'] for r in results[method]]
        
        # Simple comparison
        avg_improvement = (1 - np.mean(method_steps) / np.mean(rw_steps)) * 100
        print(f"\n{method} vs Random Walk:")
        print(f"  Average improvement: {avg_improvement:+.1f}%")
        
        # Success rate comparison
        rw_success = stats['random_walk']['success_rate']
        method_success = stats[method]['success_rate']
        print(f"  Success rate: {method_success:.0f}% vs {rw_success:.0f}% (random walk)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'fair_comparison_results_{timestamp}.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_runs': num_runs,
            'results': results,
            'statistics': stats
        }, f, indent=2)
    
    print(f"\nResults saved to fair_comparison_results_{timestamp}.json")
    
    return results, stats


if __name__ == "__main__":
    results, stats = compare_with_smart_random_walk(num_runs=10)