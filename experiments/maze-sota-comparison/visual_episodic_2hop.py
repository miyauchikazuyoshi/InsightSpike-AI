#!/usr/bin/env python3
"""
Visual Episodic 2-hop Navigator
================================

Properly uses visual information from episodes
for better navigation decisions.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from collections import deque, defaultdict
import time
from datetime import datetime
import json

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


class VisualEpisodic2HopNavigator:
    """Visual information aware episodic navigator"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode memory
        self.episodes: List[Episode7D] = []
        self.position_visits: Dict[Tuple[int, int], int] = {(1, 1): 1}
        
        # Initialize with goal episode
        self._add_goal_episode()
        self._record_visual_information(1, 1)
        
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
        
    def _record_visual_information(self, x: int, y: int):
        """Record what we can see from current position"""
        directions = [
            ((x+1, y), 'right'),
            ((x-1, y), 'left'),
            ((x, y-1), 'up'),
            ((x, y+1), 'down')
        ]
        
        for (nx, ny), direction in directions:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                wall_or_path = 'path' if self.maze[ny, nx] == 0 else 'wall'
                visit_count = self.position_visits.get((nx, ny), 0)
                
                self.episodes.append(Episode7D(
                    x=nx, y=ny,
                    direction=direction,
                    result=None,
                    visit_count=visit_count,
                    goal_or_not=False,
                    wall_or_path=wall_or_path
                ))
    
    def _create_query(self) -> Episode7D:
        """Create query from current position"""
        x, y = self.position
        return Episode7D(
            x=x, y=y,
            direction=None,
            result=None,
            visit_count=None,
            goal_or_not=None,
            wall_or_path=None
        )
    
    def _calculate_similarity(self, query: Episode7D, episode: Episode7D) -> float:
        """Calculate similarity between query and episode"""
        # Goal episode with null coordinates
        if episode.goal_or_not and episode.x is None:
            return 0.3  # Moderate constant attraction
        
        if episode.x is None or episode.y is None:
            return 0.0
            
        distance = abs(query.x - episode.x) + abs(query.y - episode.y)
        base_score = 1.0 / (1.0 + distance * 0.1)
        
        # Bonus for adjacent visual information
        if distance == 1 and episode.direction is not None:
            base_score *= 2.0  # Higher weight for visual info
            
        # Penalty for walls
        if episode.wall_or_path == 'wall':
            base_score *= 0.1
            
        # Penalty for highly visited
        if episode.visit_count is not None and episode.visit_count > 3:
            base_score *= 0.5
            
        return base_score
    
    def _search_episodes(self) -> List[Tuple[Episode7D, float]]:
        """Search episodes using current position as query"""
        query = self._create_query()
        results = []
        
        for episode in self.episodes:
            similarity = self._calculate_similarity(query, episode)
            if similarity > 0.05:
                results.append((episode, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:20]  # Top 20
    
    def evaluate_1hop_visual(self) -> Dict:
        """1-hop evaluation using visual episodes"""
        search_results = self._search_episodes()
        
        # Score each direction based on episodes
        direction_scores = {
            'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0
        }
        
        x, y = self.position
        
        for episode, score in search_results:
            # Goal influence
            if episode.goal_or_not:
                # Slight preference for all directions
                for d in direction_scores:
                    direction_scores[d] += score * 0.25
            
            # Visual information - high priority
            elif episode.direction and episode.wall_or_path == 'path':
                direction_scores[episode.direction] += score * 3.0
            
            # Wall avoidance
            elif episode.wall_or_path == 'wall' and episode.direction:
                # This is a wall we can see
                direction_scores[episode.direction] -= score * 5.0
        
        # Choose best valid action
        valid_actions = []
        for direction, score in direction_scores.items():
            next_pos = self._get_next_position(self.position, direction)
            if next_pos not in self.visited:  # Don't need to check validity - visual info tells us
                valid_actions.append((direction, score))
        
        if valid_actions:
            valid_actions.sort(key=lambda x: x[1], reverse=True)
            return {
                'action': valid_actions[0][0],
                'type': '1hop_visual',
                'score': valid_actions[0][1],
                'episode_count': len(search_results)
            }
        
        return {'action': 'backtrack', 'type': '1hop_visual'}
    
    def evaluate_2hop_visual(self) -> Dict:
        """2-hop evaluation with visual information consideration"""
        
        # Need enough episodes
        if len(self.episodes) < 50:
            return self.evaluate_1hop_visual()
        
        # Build position-episode graph
        position_graph = nx.Graph()
        
        # Add nodes for positions we know about
        for ep in self.episodes:
            if ep.x is not None and ep.y is not None:
                pos = (ep.x, ep.y)
                if pos not in position_graph:
                    position_graph.add_node(pos)
        
        # Add edges based on visual connections
        for ep in self.episodes:
            if ep.direction and ep.wall_or_path == 'path' and ep.x is not None:
                # This episode shows a visual connection
                from_pos = self.position  # Current position when we saw this
                to_pos = (ep.x, ep.y)
                if from_pos in position_graph and to_pos in position_graph:
                    position_graph.add_edge(from_pos, to_pos)
        
        # 2-hop structural analysis
        current_pos = self.position
        if current_pos not in position_graph:
            return self.evaluate_1hop_visual()
        
        # Check 2-hop connectivity
        two_hop_positions = set()
        for neighbor in position_graph.neighbors(current_pos):
            for n2 in position_graph.neighbors(neighbor):
                if n2 != current_pos and n2 not in self.visited:
                    two_hop_positions.add(n2)
        
        # If we have good 2-hop connectivity, use 1-hop
        if len(two_hop_positions) > 3:
            return self.evaluate_1hop_visual()
        
        # Otherwise, check for better paths
        # Look for positions with better connectivity
        best_direction = None
        best_connectivity = len(two_hop_positions)
        
        for direction in ['up', 'down', 'left', 'right']:
            next_pos = self._get_next_position(self.position, direction)
            if next_pos in position_graph and next_pos not in self.visited:
                # Check its 2-hop connectivity
                next_2hop = set()
                for n in position_graph.neighbors(next_pos):
                    for n2 in position_graph.neighbors(n):
                        if n2 != next_pos and n2 not in self.visited:
                            next_2hop.add(n2)
                
                if len(next_2hop) > best_connectivity:
                    best_connectivity = len(next_2hop)
                    best_direction = direction
        
        if best_direction:
            return {
                'action': best_direction,
                'type': '2hop_visual',
                'reason': 'better_connectivity',
                'connectivity': best_connectivity
            }
        
        # Fallback to 1-hop
        return self.evaluate_1hop_visual()
    
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
        """Navigate using visual episodic evaluation"""
        
        print(f"Visual Episodic Navigation: 2-hop={use_2hop}")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        two_hop_decisions = 0
        wall_hits = 0
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Evaluate
            if use_2hop:
                decision = self.evaluate_2hop_visual()
            else:
                decision = self.evaluate_1hop_visual()
            
            if decision['type'].startswith('2hop'):
                two_hop_decisions += 1
            
            # Progress report
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"episodes={len(self.episodes)}, 2hop={two_hop_decisions}")
            
            # Execute action
            action = decision['action']
            
            if action == 'backtrack':
                backtrack_count += 1
                if len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
            else:
                next_pos = self._get_next_position(self.position, action)
                
                # Check if valid (we should know from visual info)
                if (0 <= next_pos[0] < self.width and 
                    0 <= next_pos[1] < self.height and
                    self.maze[next_pos[1], next_pos[0]] == 0):
                    
                    # Record movement episode
                    self.episodes.append(Episode7D(
                        x=self.position[0], y=self.position[1],
                        direction=action,
                        result='moved',
                        visit_count=self.position_visits.get(self.position, 0),
                        goal_or_not=False,
                        wall_or_path='path'
                    ))
                    
                    # Move
                    self.position = next_pos
                    self.visited.add(next_pos)
                    self.path.append(next_pos)
                    self.position_visits[next_pos] = self.position_visits.get(next_pos, 0) + 1
                    
                    # Record new visual information
                    self._record_visual_information(next_pos[0], next_pos[1])
                else:
                    # Hit a wall (shouldn't happen with good visual info)
                    wall_hits += 1
                    self.episodes.append(Episode7D(
                        x=self.position[0], y=self.position[1],
                        direction=action,
                        result='wall',
                        visit_count=self.position_visits.get(self.position, 0),
                        goal_or_not=False,
                        wall_or_path='path'
                    ))
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        
        print(f"\nNavigation complete: success={success}, steps={steps}, "
              f"time={elapsed_time:.2f}s")
        print(f"Episodes: {len(self.episodes)}, 2-hop decisions: {two_hop_decisions}")
        print(f"Wall hits: {wall_hits} (should be near 0)")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'episode_count': len(self.episodes),
            'backtrack_count': backtrack_count,
            'two_hop_decisions': two_hop_decisions,
            'wall_hits': wall_hits,
            'elapsed_time': elapsed_time,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0
        }


def test_visual_episodic():
    """Test visual episodic navigation"""
    
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
    print("VISUAL EPISODIC 2-HOP NAVIGATION TEST")
    print("="*70)
    
    goal = (size-2, size-2)
    
    # Test 1-hop visual
    print("\n--- 1-hop Visual Episodic ---")
    nav_1hop = VisualEpisodic2HopNavigator(maze)
    result_1hop = nav_1hop.navigate(goal, use_2hop=False, max_steps=2000)
    
    # Test 2-hop visual
    print("\n--- 2-hop Visual Episodic ---")
    nav_2hop = VisualEpisodic2HopNavigator(maze)
    result_2hop = nav_2hop.navigate(goal, use_2hop=True, max_steps=2000)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n1-hop Visual Episodic:")
    print(f"  Success: {result_1hop['success']}")
    print(f"  Steps: {result_1hop['steps']}")
    print(f"  Episodes: {result_1hop['episode_count']}")
    print(f"  Wall hits: {result_1hop['wall_hits']}")
    print(f"  Efficiency: {result_1hop['efficiency']:.1f}%")
    
    print(f"\n2-hop Visual Episodic:")
    print(f"  Success: {result_2hop['success']}")
    print(f"  Steps: {result_2hop['steps']}")
    print(f"  Episodes: {result_2hop['episode_count']}")
    print(f"  2-hop decisions: {result_2hop['two_hop_decisions']}")
    print(f"  Wall hits: {result_2hop['wall_hits']}")
    print(f"  Efficiency: {result_2hop['efficiency']:.1f}%")
    
    if result_1hop['steps'] > 0 and result_2hop['steps'] > 0:
        print(f"\nImprovement:")
        print(f"  Steps: {(1 - result_2hop['steps']/result_1hop['steps'])*100:+.1f}%")
        print(f"  Efficiency: {result_2hop['efficiency'] - result_1hop['efficiency']:+.1f}%")
        
    print(f"\nKey insight: Wall hits should be near 0 with proper visual information use")


if __name__ == "__main__":
    test_visual_episodic()