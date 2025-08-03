#!/usr/bin/env python3
"""
Adaptive 2-hop Navigator with Improved Strategy
==============================================

Implements adaptive thresholds and hybrid strategies
to avoid the over-conservative trap.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class Adaptive2HopNavigator:
    """Adaptive 2-hop navigator that balances exploration and caution"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.visited = set()
        self.path = []
        self.decision_history = []
        self.dead_end_cache = set()
        self.stuck_counter = 0
        self.last_positions = deque(maxlen=10)
        
    def navigate(self, start: Tuple[int, int], goal: Tuple[int, int], 
                max_steps: int = 3000) -> Dict:
        """Navigate with adaptive 2-hop strategy"""
        
        self.visited = {start}
        self.path = [start]
        position = start
        
        steps = 0
        backtrack_count = 0
        strategy_switches = 0
        current_strategy = "adaptive"
        
        while position != goal and steps < max_steps:
            # Adaptive threshold based on progress
            progress_ratio = len(self.visited) / max(steps + 1, 1)
            exploration_phase = steps < 200 or progress_ratio > 0.5
            
            # Detect if stuck
            self.last_positions.append(position)
            if len(self.last_positions) == 10 and len(set(self.last_positions)) < 3:
                self.stuck_counter += 1
                if self.stuck_counter > 3:
                    # Force exploration mode
                    current_strategy = "1hop"
                    strategy_switches += 1
                    self.stuck_counter = 0
            
            # Choose strategy
            if current_strategy == "1hop" or exploration_phase:
                decision = self._evaluate_1hop(position, goal)
            else:
                decision = self._evaluate_adaptive_2hop(position, goal, progress_ratio)
            
            # Execute decision
            if decision.get('action') == 'backtrack':
                backtrack_count += 1
                target = decision.get('target')
                if target and target in self.path:
                    idx = self.path.index(target)
                    self.path = self.path[:idx + 1]
                    position = target
                else:
                    # Simple backtrack
                    if len(self.path) > 1:
                        self.path.pop()
                        position = self.path[-1]
            else:
                # Move forward
                next_pos = decision.get('action')
                if next_pos and next_pos not in self.visited:
                    position = next_pos
                    self.visited.add(position)
                    self.path.append(position)
                else:
                    # Need to backtrack
                    backtrack_count += 1
                    if len(self.path) > 1:
                        self.path.pop()
                        position = self.path[-1]
            
            steps += 1
            
            # Reset strategy if making progress
            if steps % 50 == 0 and len(self.visited) > steps * 0.5:
                current_strategy = "adaptive"
        
        success = position == goal
        efficiency = len(self.visited) / steps * 100 if steps > 0 else 0
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'efficiency': efficiency,
            'backtrack_count': backtrack_count,
            'strategy_switches': strategy_switches,
            'final_position': position
        }
    
    def _evaluate_1hop(self, position: Tuple[int, int], 
                      goal: Tuple[int, int]) -> Dict:
        """Simple greedy 1-hop evaluation"""
        neighbors = self._get_neighbors(position)
        
        best_action = None
        best_distance = float('inf')
        
        for neighbor in neighbors:
            if neighbor not in self.visited:
                dist = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                if dist < best_distance:
                    best_distance = dist
                    best_action = neighbor
        
        return {'action': best_action, 'type': '1hop'}
    
    def _evaluate_adaptive_2hop(self, position: Tuple[int, int], 
                               goal: Tuple[int, int], 
                               progress_ratio: float) -> Dict:
        """Adaptive 2-hop evaluation with dynamic thresholds"""
        
        # Adaptive threshold: more conservative as we explore more
        threshold = 0.5 + (0.3 * (1 - progress_ratio))
        
        # Quick 2-hop scan
        dead_end_prob = self._quick_2hop_scan(position)
        
        if dead_end_prob > threshold:
            # Only backtrack if really necessary
            unvisited_neighbors = sum(1 for n in self._get_neighbors(position) 
                                    if n not in self.visited)
            if unvisited_neighbors == 0:
                backtrack_target = self._find_junction()
                if backtrack_target:
                    return {
                        'action': 'backtrack',
                        'target': backtrack_target,
                        'type': '2hop_adaptive'
                    }
        
        # Otherwise use 1-hop
        return self._evaluate_1hop(position, goal)
    
    def _quick_2hop_scan(self, position: Tuple[int, int]) -> float:
        """Quick dead-end probability estimation"""
        
        # Simple 2-hop connectivity check
        reachable = 0
        visited_scan = {position}
        queue = deque([(position, 0)])
        
        while queue and len(visited_scan) < 20:
            pos, depth = queue.popleft()
            if depth >= 2:
                continue
                
            for neighbor in self._get_neighbors(pos):
                if neighbor not in visited_scan and neighbor not in self.visited:
                    visited_scan.add(neighbor)
                    reachable += 1
                    queue.append((neighbor, depth + 1))
        
        # Return probability (0-1)
        if reachable < 3:
            return 0.9
        elif reachable < 6:
            return 0.5
        else:
            return 0.1
    
    def _find_junction(self) -> Optional[Tuple[int, int]]:
        """Find junction with unexplored paths"""
        for i in range(len(self.path) - 1, max(0, len(self.path) - 30), -1):
            pos = self.path[i]
            unvisited = sum(1 for n in self._get_neighbors(pos) 
                          if n not in self.visited)
            if unvisited >= 2:
                return pos
        return None
    
    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors"""
        neighbors = []
        i, j = position
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < self.height and 
                0 <= nj < self.width and 
                self.maze[ni, nj] == 0):
                neighbors.append((ni, nj))
        
        return neighbors


def test_adaptive_navigator():
    """Quick test of adaptive navigator"""
    
    # Generate simple test maze
    np.random.seed(42)
    size = 50
    maze = np.ones((size, size), dtype=int)
    
    # Create paths
    for i in range(1, size-1, 2):
        for j in range(1, size-1, 2):
            maze[i, j] = 0
            if np.random.random() > 0.3:
                if i+1 < size-1:
                    maze[i+1, j] = 0
            if np.random.random() > 0.3:
                if j+1 < size-1:
                    maze[i, j+1] = 0
    
    maze[1, 1] = 0
    maze[size-2, size-2] = 0
    
    # Test navigation
    navigator = Adaptive2HopNavigator(maze)
    result = navigator.navigate((1, 1), (48, 48), max_steps=2000)
    
    print(f"Adaptive 2-hop Results:")
    print(f"  Success: {result['success']}")
    print(f"  Steps: {result['steps']}")
    print(f"  Efficiency: {result['efficiency']:.1f}%")
    print(f"  Backtracks: {result['backtrack_count']}")
    print(f"  Strategy switches: {result['strategy_switches']}")
    
    return result


if __name__ == "__main__":
    test_adaptive_navigator()