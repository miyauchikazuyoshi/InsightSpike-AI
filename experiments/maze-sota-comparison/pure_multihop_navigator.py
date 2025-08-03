#!/usr/bin/env python3
"""
Pure Multi-hop geDIG Navigator
==============================

A pure implementation without wall-avoidance "cheat".
All actions are evaluated equally, including moves into walls.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import json

@dataclass
class PureEpisode:
    """Episode with action result"""
    position: Tuple[int, int]
    action: str
    next_position: Tuple[int, int]
    success: bool  # True if moved, False if hit wall
    visit_count: int
    timestamp: int
    value: float  # The value assigned by n-hop evaluation

class PureMultiHopNavigator:
    """Pure navigator without wall-avoidance preprocessing"""
    
    def __init__(self, maze: np.ndarray, hop_count: int = 1):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        self.hop_count = hop_count
        
        # Episode memory
        self.episodes: List[PureEpisode] = []
        self.position_visits = {(1, 1): 1}
        self.episode_counter = 0
        
        # Statistics
        self.wall_hits = 0
        self.successful_moves = 0
        self.coverage_over_time = []
        
    def _normalize_position(self, pos: Tuple[int, int]) -> np.ndarray:
        """Normalize position to [-1, 1]"""
        x, y = pos
        norm_x = (x / self.width) * 2 - 1
        norm_y = (y / self.height) * 2 - 1
        return np.array([norm_x, norm_y])
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall"""
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.maze[y, x] == 0
    
    def _calculate_n_hop_value(self, action: str, current_pos: Tuple[int, int]) -> Tuple[float, Tuple[int, int]]:
        """
        Calculate n-hop evaluation value for an action.
        Returns (value, next_position).
        """
        # Get next position
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                  'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # Base value calculation
        if not self._is_valid_position(next_pos):
            # Wall or out of bounds - strong negative value
            base_value = -5.0
        else:
            # Valid position
            visit_count = self.position_visits.get(next_pos, 0)
            if next_pos in self.visited:
                # Already visited - mild negative
                base_value = -0.5 / (1 + visit_count)
            else:
                # Unvisited - positive value
                base_value = 2.0 / (1 + visit_count)
        
        # Add n-hop lookahead bonus (only for valid positions)
        if base_value > -5.0 and self.hop_count > 1:
            lookahead_bonus = self._calculate_lookahead_bonus(next_pos, self.hop_count - 1)
            value = base_value + lookahead_bonus * 0.5
        else:
            value = base_value
        
        return value, next_pos
    
    def _calculate_lookahead_bonus(self, pos: Tuple[int, int], remaining_hops: int) -> float:
        """Calculate bonus based on what's reachable from this position"""
        if remaining_hops <= 0 or not self._is_valid_position(pos):
            return 0.0
        
        total_bonus = 0.0
        
        # Check all directions
        for action in ['up', 'right', 'down', 'left']:
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            next_pos = (pos[0] + dx, pos[1] + dy)
            
            if self._is_valid_position(next_pos):
                # Bonus for reachable unvisited positions
                if next_pos not in self.visited:
                    total_bonus += 1.0
                
                # Recursive lookahead
                if remaining_hops > 1:
                    total_bonus += self._calculate_lookahead_bonus(next_pos, remaining_hops - 1) * 0.3
        
        return total_bonus
    
    def decide_action(self) -> Tuple[str, float]:
        """
        Decide action by evaluating ALL possible actions.
        Returns (action, value).
        """
        current_pos = self.position
        action_values = {}
        
        # Evaluate ALL four actions
        for action in ['up', 'right', 'down', 'left']:
            value, next_pos = self._calculate_n_hop_value(action, current_pos)
            action_values[action] = (value, next_pos)
        
        # Softmax selection
        actions = list(action_values.keys())
        values = np.array([v[0] for v in action_values.values()])
        
        # Temperature parameter
        temperature = 0.5
        
        # Softmax probabilities
        exp_values = np.exp(values / temperature)
        probs = exp_values / exp_values.sum()
        
        # Choose action
        chosen_action = np.random.choice(actions, p=probs)
        chosen_value = action_values[chosen_action][0]
        
        return chosen_action, chosen_value
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 10000) -> Dict:
        """Navigate maze using pure n-hop evaluation"""
        
        print(f"\nPure {self.hop_count}-hop Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        
        # Track walkable cells
        walkable_cells = sum(1 for y in range(self.height) 
                           for x in range(self.width) 
                           if self.maze[y, x] == 0)
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Track coverage
            if steps % 100 == 0:
                coverage = len(self.visited) / walkable_cells * 100
                self.coverage_over_time.append((steps, coverage))
            
            # Progress report
            if steps % 1000 == 0 and steps > 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                coverage = len(self.visited) / walkable_cells * 100
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%, wall_hits={self.wall_hits}")
            
            # Decide action
            action, value = self.decide_action()
            
            # Get next position
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            
            # Try to execute action
            if self._is_valid_position(next_pos):
                # Successful move
                self.successful_moves += 1
                old_pos = self.position
                self.position = next_pos
                
                if next_pos not in self.visited:
                    self.visited.add(next_pos)
                
                self.path.append(next_pos)
                self.position_visits[next_pos] = self.position_visits.get(next_pos, 0) + 1
                
                # Record episode
                episode = PureEpisode(
                    position=old_pos,
                    action=action,
                    next_position=next_pos,
                    success=True,
                    visit_count=self.position_visits[next_pos],
                    timestamp=steps,
                    value=value
                )
                self.episodes.append(episode)
                
            else:
                # Hit wall or boundary
                self.wall_hits += 1
                
                # Record failed episode
                episode = PureEpisode(
                    position=self.position,
                    action=action,
                    next_position=next_pos,
                    success=False,
                    visit_count=0,
                    timestamp=steps,
                    value=value
                )
                self.episodes.append(episode)
                
                # Check if we need to backtrack
                # (all actions lead to walls or visited positions)
                all_blocked = True
                for test_action in ['up', 'right', 'down', 'left']:
                    test_value, test_next = self._calculate_n_hop_value(test_action, self.position)
                    if test_value > -0.5:  # At least one non-negative option
                        all_blocked = False
                        break
                
                if all_blocked and len(self.path) > 1:
                    # Backtrack
                    backtrack_count += 1
                    self.path.pop()
                    self.position = self.path[-1]
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        final_coverage = len(self.visited) / walkable_cells * 100
        
        print(f"\nNavigation complete: success={success}, steps={steps}")
        print(f"Coverage: {final_coverage:.1f}%, Wall hits: {self.wall_hits}")
        print(f"Success rate: {self.successful_moves/(self.successful_moves+self.wall_hits)*100:.1f}%")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'coverage': final_coverage,
            'wall_hits': self.wall_hits,
            'successful_moves': self.successful_moves,
            'move_success_rate': self.successful_moves / (self.successful_moves + self.wall_hits) * 100,
            'backtrack_count': backtrack_count,
            'elapsed_time': elapsed_time,
            'coverage_over_time': self.coverage_over_time
        }


def test_pure_implementation(num_runs=5):
    """Test pure implementation with different hop counts"""
    
    print("="*70)
    print("PURE IMPLEMENTATION TEST (No Wall-Avoidance)")
    print("="*70)
    
    all_results = {
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
        
        # Test different hop counts
        for hop_count in [1, 2, 3]:
            navigator = PureMultiHopNavigator(maze, hop_count=hop_count)
            result = navigator.navigate(goal, max_steps=10000)
            all_results[f'{hop_count}-hop'].append(result)
    
    # Analysis
    print("\n" + "="*70)
    print("PURE IMPLEMENTATION RESULTS SUMMARY")
    print("="*70)
    
    # Calculate statistics
    stats = {}
    for method in all_results:
        method_results = all_results[method]
        successful = [r for r in method_results if r['success']]
        
        stats[method] = {
            'success_rate': len(successful) / num_runs * 100,
            'avg_steps_all': np.mean([r['steps'] for r in method_results]),
            'avg_steps_success': np.mean([r['steps'] for r in successful]) if successful else None,
            'avg_coverage': np.mean([r['coverage'] for r in method_results]),
            'avg_wall_hits': np.mean([r['wall_hits'] for r in method_results]),
            'avg_move_success_rate': np.mean([r['move_success_rate'] for r in method_results])
        }
    
    # Display results
    print("\n| Method | Success% | Avg Steps | Wall Hits | Move Success% | Coverage% |")
    print("|--------|----------|-----------|-----------|---------------|-----------|")
    
    for method, stat in stats.items():
        print(f"| {method:6} | {stat['success_rate']:7.0f}% | {stat['avg_steps_all']:9.0f} | "
              f"{stat['avg_wall_hits']:9.0f} | {stat['avg_move_success_rate']:12.1f}% | "
              f"{stat['avg_coverage']:9.1f} |")
    
    # Detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    print("\n| Run | Method | Success | Steps  | Wall Hits | Move Success% |")
    print("|-----|--------|---------|--------|-----------|---------------|")
    
    for run in range(num_runs):
        for method in ['1-hop', '2-hop', '3-hop']:
            result = all_results[method][run]
            print(f"| {run+1:3} | {method:6} | {str(result['success']):7} | "
                  f"{result['steps']:6} | {result['wall_hits']:9} | "
                  f"{result['move_success_rate']:12.1f}% |")
        if run < num_runs - 1:
            print("|-----|--------|---------|--------|-----------|---------------|")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'pure_implementation_results_{timestamp}.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_runs': num_runs,
            'results': {k: [{kk: vv for kk, vv in r.items() if kk != 'coverage_over_time'} 
                           for r in v] for k, v in all_results.items()},
            'statistics': stats
        }, f, indent=2)
    
    print(f"\nResults saved to pure_implementation_results_{timestamp}.json")
    
    return all_results, stats


if __name__ == "__main__":
    results, stats = test_pure_implementation(num_runs=5)