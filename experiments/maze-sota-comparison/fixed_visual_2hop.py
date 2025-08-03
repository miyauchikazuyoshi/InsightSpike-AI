#!/usr/bin/env python3
"""
Fixed Visual Episodic 2-hop Navigator
=====================================

Fixes the visual information collection bug.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import time
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class Episode7D:
    """7次元エピソード"""
    x: Optional[int]
    y: Optional[int]
    direction: Optional[str]
    result: Optional[str]
    visit_count: Optional[int]
    goal_or_not: bool
    wall_or_path: Optional[str]


class FixedVisual2HopNavigator:
    """Fixed visual episodic navigator"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode memory
        self.episodes: List[Episode7D] = []
        self.position_visits: Dict[Tuple[int, int], int] = {(1, 1): 1}
        
        # Visual memory - track what we've seen from each position
        self.visual_memory: Dict[Tuple[int, int], Dict[str, str]] = {}
        
        # Initialize
        self._add_goal_episode()
        self._update_visual_memory(1, 1)
        
    def _add_goal_episode(self):
        """Add abstract goal episode"""
        self.episodes.append(Episode7D(
            x=None, y=None,
            direction=None,
            result=None,
            visit_count=0,
            goal_or_not=True,
            wall_or_path='path'
        ))
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory from current position"""
        if (x, y) not in self.visual_memory:
            self.visual_memory[(x, y)] = {}
        
        directions = {
            'right': (1, 0),
            'left': (-1, 0),
            'up': (0, -1),
            'down': (0, 1)
        }
        
        for direction, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                wall_or_path = 'path' if self.maze[ny, nx] == 0 else 'wall'
                self.visual_memory[(x, y)][direction] = wall_or_path
                
                # Record as episode
                self.episodes.append(Episode7D(
                    x=nx, y=ny,
                    direction=direction,
                    result=None,
                    visit_count=self.position_visits.get((nx, ny), 0),
                    goal_or_not=False,
                    wall_or_path=wall_or_path
                ))
    
    def evaluate_1hop(self) -> Dict:
        """1-hop evaluation using visual memory"""
        x, y = self.position
        visual_info = self.visual_memory.get((x, y), {})
        
        # Score each direction
        direction_scores = {}
        
        for direction in ['up', 'down', 'left', 'right']:
            next_pos = self._get_next_position(self.position, direction)
            
            # Check visual info
            if direction in visual_info:
                if visual_info[direction] == 'wall':
                    direction_scores[direction] = -1000  # Avoid walls
                elif next_pos not in self.visited:
                    # Prefer unvisited paths
                    visit_count = self.position_visits.get(next_pos, 0)
                    direction_scores[direction] = 10.0 / (1.0 + visit_count)
                else:
                    # Already visited
                    direction_scores[direction] = -1
            else:
                # Unknown - shouldn't happen after visual update
                direction_scores[direction] = -100
        
        # Find best direction
        valid_actions = [(d, s) for d, s in direction_scores.items() if s > 0]
        
        if valid_actions:
            valid_actions.sort(key=lambda x: x[1], reverse=True)
            return {
                'action': valid_actions[0][0],
                'type': '1hop',
                'score': valid_actions[0][1]
            }
        
        # Need to backtrack
        return {'action': 'backtrack', 'type': '1hop'}
    
    def evaluate_2hop(self) -> Dict:
        """2-hop evaluation"""
        
        # For now, just add simple 2-hop logic
        if len(self.episodes) < 100:
            return self.evaluate_1hop()
        
        # Check if we're in a dead-end pattern
        # (all neighbors visited or walls)
        x, y = self.position
        visual_info = self.visual_memory.get((x, y), {})
        
        dead_end = True
        for direction, info in visual_info.items():
            if info == 'path':
                next_pos = self._get_next_position((x, y), direction)
                if next_pos not in self.visited:
                    dead_end = False
                    break
        
        if dead_end:
            # Find backtrack point
            for i in range(len(self.path) - 2, max(0, len(self.path) - 20), -1):
                old_pos = self.path[i]
                old_visual = self.visual_memory.get(old_pos, {})
                
                for direction, info in old_visual.items():
                    if info == 'path':
                        next_pos = self._get_next_position(old_pos, direction)
                        if next_pos not in self.visited:
                            return {
                                'action': 'backtrack',
                                'target': old_pos,
                                'type': '2hop',
                                'reason': 'dead_end_escape'
                            }
        
        # Otherwise use 1-hop
        return self.evaluate_1hop()
    
    def _get_next_position(self, pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Get next position given action"""
        x, y = pos
        if action == 'up':
            return (x, y - 1)
        elif action == 'down':
            return (x, y + 1)
        elif action == 'left':
            return (x - 1, y)
        elif action == 'right':
            return (x + 1, y)
        return pos
    
    def navigate(self, goal: Tuple[int, int], use_2hop: bool = True,
                max_steps: int = 3000) -> Dict:
        """Navigate maze"""
        
        print(f"Fixed Visual Navigation: 2-hop={use_2hop}")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        two_hop_decisions = 0
        wall_hits = 0
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Evaluate
            if use_2hop:
                decision = self.evaluate_2hop()
            else:
                decision = self.evaluate_1hop()
            
            if decision['type'] == '2hop':
                two_hop_decisions += 1
            
            # Progress
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"visited={len(self.visited)}, backtracks={backtrack_count}")
            
            # Execute
            action = decision['action']
            
            if action == 'backtrack':
                backtrack_count += 1
                target = decision.get('target')
                if target and target in self.path:
                    idx = self.path.index(target)
                    self.path = self.path[:idx + 1]
                    self.position = target
                elif len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
            else:
                next_pos = self._get_next_position(self.position, action)
                
                # Move (we should already know it's valid from visual info)
                self.position = next_pos
                self.visited.add(next_pos)
                self.path.append(next_pos)
                self.position_visits[next_pos] = self.position_visits.get(next_pos, 0) + 1
                
                # Update visual memory from new position
                self._update_visual_memory(next_pos[0], next_pos[1])
                
                # Record movement episode
                self.episodes.append(Episode7D(
                    x=self.position[0], y=self.position[1],
                    direction=action,
                    result='moved',
                    visit_count=self.position_visits.get(self.position, 0),
                    goal_or_not=False,
                    wall_or_path='path'
                ))
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        
        print(f"\nNavigation complete: success={success}, steps={steps}, "
              f"time={elapsed_time:.2f}s")
        print(f"Visited: {len(self.visited)}, Episodes: {len(self.episodes)}")
        print(f"Backtracks: {backtrack_count}, 2-hop decisions: {two_hop_decisions}")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'episode_count': len(self.episodes),
            'backtrack_count': backtrack_count,
            'two_hop_decisions': two_hop_decisions,
            'elapsed_time': elapsed_time,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0
        }


def test_fixed_visual():
    """Test fixed visual navigation"""
    
    # Generate maze
    np.random.seed(42)
    size = 30
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
    
    # Add extra paths
    for _ in range(size // 2):
        x = np.random.randint(1, size-1)
        y = np.random.randint(1, size-1)
        if maze[y, x] == 1:
            neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                          if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
            if neighbors >= 2:
                maze[y, x] = 0
    
    print("="*70)
    print("FIXED VISUAL 2-HOP NAVIGATION TEST")
    print("="*70)
    
    goal = (size-2, size-2)
    
    # Test 1-hop
    print("\n--- 1-hop Fixed Visual ---")
    nav_1hop = FixedVisual2HopNavigator(maze)
    result_1hop = nav_1hop.navigate(goal, use_2hop=False, max_steps=2000)
    
    # Test 2-hop
    print("\n--- 2-hop Fixed Visual ---")
    nav_2hop = FixedVisual2HopNavigator(maze)
    result_2hop = nav_2hop.navigate(goal, use_2hop=True, max_steps=2000)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for name, result in [("1-hop", result_1hop), ("2-hop", result_2hop)]:
        print(f"\n{name}:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Efficiency: {result['efficiency']:.1f}%")
        print(f"  Backtracks: {result['backtrack_count']}")
        if '2hop' in name:
            print(f"  2-hop decisions: {result['two_hop_decisions']}")


if __name__ == "__main__":
    test_fixed_visual()