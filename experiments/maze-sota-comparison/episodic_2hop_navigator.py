#!/usr/bin/env python3
"""
Episodic 2-hop Navigator
========================

Episode-based navigation with 2-hop structural evaluation.
Uses accumulated episodes to make better decisions.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from collections import deque
import time
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class Episode7D:
    """7次元エピソード"""
    x: int
    y: int
    direction: Optional[str]
    result: Optional[str]
    visit_count: Optional[int]
    goal_or_not: bool
    wall_or_path: str


class Episodic2HopNavigator:
    """Episode-based navigator with 2-hop evaluation"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)  # Start position
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode memory
        self.episodes: List[Episode7D] = []
        self.position_visits: Dict[Tuple[int, int], int] = {(1, 1): 1}
        
        # Goal episode (null coordinates version)
        self._add_goal_episode()
        
        # Record initial visual information
        self._record_visual_information(1, 1)
        
    def _add_goal_episode(self):
        """Add goal episode with null coordinates"""
        self.episodes.append(Episode7D(
            x=None, y=None,  # Null coordinates
            direction=None,
            result=None,
            visit_count=0,
            goal_or_not=True,
            wall_or_path='path'
        ))
        
    def _record_visual_information(self, x: int, y: int):
        """Record visual information from current position"""
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
        # Goal episode special handling
        if episode.goal_or_not and episode.x is None:
            # Abstract goal - use special logic
            return 0.5  # Moderate constant attraction
        
        # Position-based similarity
        if episode.x is None or episode.y is None:
            return 0.0
            
        distance = abs(query.x - episode.x) + abs(query.y - episode.y)
        base_score = 1.0 / (1.0 + distance * 0.1)
        
        # Bonus for adjacent visual information
        if distance == 1 and episode.direction is not None:
            base_score *= 1.5
            
        # Penalty for highly visited positions
        if episode.visit_count is not None and episode.visit_count > 3:
            base_score *= 0.5
            
        return base_score
    
    def evaluate_1hop(self) -> Dict:
        """1-hop evaluation using episode similarity"""
        query = self._create_query()
        
        # Search similar episodes
        results = []
        for episode in self.episodes:
            similarity = self._calculate_similarity(query, episode)
            if similarity > 0.1:
                results.append((episode, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Decide action based on top episodes
        action = self._decide_action_from_episodes(results[:10])
        
        return {
            'action': action,
            'type': '1hop',
            'episode_count': len(results)
        }
    
    def evaluate_2hop(self) -> Dict:
        """2-hop structural evaluation on episode graph"""
        
        # Need at least 50 episodes for meaningful 2-hop
        if len(self.episodes) < 50:
            return self.evaluate_1hop()
        
        # Build episode graph for 2-hop analysis
        episode_graph = self._build_episode_graph()
        
        # Check for structural patterns
        structural_insight = self._analyze_2hop_structure(episode_graph)
        
        if structural_insight['detected']:
            return {
                'action': structural_insight['action'],
                'type': '2hop',
                'reason': structural_insight['reason'],
                'confidence': structural_insight['confidence']
            }
        
        # Fallback to 1-hop
        return self.evaluate_1hop()
    
    def _build_episode_graph(self) -> nx.Graph:
        """Build graph from episodes for structural analysis"""
        G = nx.Graph()
        
        # Add nodes for unique positions
        position_episodes = {}
        for i, ep in enumerate(self.episodes):
            if ep.x is not None and ep.y is not None:
                pos = (ep.x, ep.y)
                if pos not in position_episodes:
                    position_episodes[pos] = []
                position_episodes[pos].append(i)
                G.add_node(pos)
        
        # Add edges based on adjacency and movement episodes
        for ep in self.episodes:
            if ep.direction and ep.result == 'moved' and ep.x is not None:
                # Movement episode creates edge
                from_pos = (ep.x, ep.y)
                to_pos = self._get_next_position(from_pos, ep.direction)
                if to_pos in G:
                    G.add_edge(from_pos, to_pos)
        
        return G
    
    def _analyze_2hop_structure(self, G: nx.Graph) -> Dict:
        """Analyze 2-hop structure for insights"""
        current_pos = self.position
        
        if current_pos not in G:
            return {'detected': False}
        
        # Get 2-hop neighborhood
        neighbors_1hop = set(G.neighbors(current_pos))
        neighbors_2hop = set()
        
        for n1 in neighbors_1hop:
            for n2 in G.neighbors(n1):
                if n2 != current_pos:
                    neighbors_2hop.add(n2)
        
        # Detect dead-end pattern
        if len(neighbors_2hop) < len(neighbors_1hop):
            # Contracting structure - likely dead end
            # Find better path
            for neighbor in neighbors_1hop:
                if neighbor not in self.visited:
                    neighbor_2hop = set()
                    for n2 in G.neighbors(neighbor):
                        if n2 != neighbor:
                            neighbor_2hop.add(n2)
                    
                    if len(neighbor_2hop) > len(neighbors_2hop):
                        # This neighbor has better connectivity
                        return {
                            'detected': True,
                            'action': self._get_direction_to(neighbor),
                            'reason': 'better_connectivity',
                            'confidence': 0.8
                        }
        
        # Detect loop pattern
        revisit_count = sum(1 for pos in neighbors_2hop if pos in self.visited)
        if revisit_count > len(neighbors_2hop) * 0.7:
            # Most 2-hop neighbors already visited - likely in loop
            # Find least visited direction
            best_direction = None
            min_visits = float('inf')
            
            for neighbor in neighbors_1hop:
                visits = self.position_visits.get(neighbor, 0)
                if visits < min_visits and neighbor not in self.visited:
                    min_visits = visits
                    best_direction = self._get_direction_to(neighbor)
            
            if best_direction:
                return {
                    'detected': True,
                    'action': best_direction,
                    'reason': 'escape_loop',
                    'confidence': 0.7
                }
        
        return {'detected': False}
    
    def _decide_action_from_episodes(self, top_episodes: List[Tuple[Episode7D, float]]) -> str:
        """Decide action based on top episodes"""
        x, y = self.position
        
        # Score each direction
        direction_scores = {
            'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0
        }
        
        for episode, score in top_episodes:
            # Goal episode influence (even with null coordinates)
            if episode.goal_or_not:
                # Random exploration with slight bias
                for d in direction_scores:
                    direction_scores[d] += score * 0.25
            
            # Visual information
            elif episode.direction and episode.wall_or_path == 'path':
                direction_scores[episode.direction] += score * 2.0
            
            # Wall avoidance
            elif episode.wall_or_path == 'wall':
                for direction in ['up', 'down', 'left', 'right']:
                    next_pos = self._get_next_position((x, y), direction)
                    if next_pos == (episode.x, episode.y):
                        direction_scores[direction] -= score * 3.0
        
        # Choose valid action with highest score
        valid_actions = []
        for direction, score in direction_scores.items():
            next_pos = self._get_next_position((x, y), direction)
            if self._is_valid_position(next_pos) and next_pos not in self.visited:
                valid_actions.append((direction, score))
        
        if valid_actions:
            valid_actions.sort(key=lambda x: x[1], reverse=True)
            return valid_actions[0][0]
        
        # No unvisited positions - backtrack
        return 'backtrack'
    
    def _get_next_position(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Get next position given direction"""
        x, y = pos
        if direction == 'up':
            return (x, y - 1)
        elif direction == 'down':
            return (x, y + 1)
        elif direction == 'left':
            return (x - 1, y)
        elif direction == 'right':
            return (x + 1, y)
        return pos
    
    def _get_direction_to(self, target: Tuple[int, int]) -> str:
        """Get direction from current position to target"""
        x, y = self.position
        tx, ty = target
        
        if tx > x:
            return 'right'
        elif tx < x:
            return 'left'
        elif ty > y:
            return 'down'
        elif ty < y:
            return 'up'
        return 'wait'
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid and walkable"""
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.maze[y, x] == 0)
    
    def navigate(self, goal: Tuple[int, int], use_2hop: bool = True,
                max_steps: int = 3000) -> Dict:
        """Navigate maze using episodic memory"""
        
        print(f"Episodic Navigation: 2-hop={use_2hop}")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        two_hop_decisions = 0
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Evaluate
            if use_2hop:
                decision = self.evaluate_2hop()
            else:
                decision = self.evaluate_1hop()
            
            if decision['type'] == '2hop':
                two_hop_decisions += 1
            
            # Progress report
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"episodes={len(self.episodes)}, 2hop_decisions={two_hop_decisions}")
            
            # Execute action
            action = decision['action']
            
            if action == 'backtrack':
                backtrack_count += 1
                if len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
                else:
                    break
            else:
                next_pos = self._get_next_position(self.position, action)
                if self._is_valid_position(next_pos):
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
                    
                    # Check if reached goal
                    if self.position == goal:
                        # Mark goal discovery
                        self.episodes.append(Episode7D(
                            x=goal[0], y=goal[1],
                            direction=None,
                            result='goal_reached',
                            visit_count=self.position_visits.get(goal, 0),
                            goal_or_not=True,
                            wall_or_path='path'
                        ))
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        
        print(f"\nNavigation complete: success={success}, steps={steps}, "
              f"time={elapsed_time:.2f}s")
        print(f"Episodes: {len(self.episodes)}, 2-hop decisions: {two_hop_decisions}")
        
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


def test_episodic_navigation():
    """Test episodic navigation on 50x50 maze"""
    
    # Generate maze
    np.random.seed(42)
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
    
    # Add some random paths
    for _ in range(size):
        x = np.random.randint(1, size-1)
        y = np.random.randint(1, size-1)
        if maze[y, x] == 1:
            neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                          if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
            if neighbors >= 2:
                maze[y, x] = 0
    
    print("="*70)
    print("EPISODIC 2-HOP NAVIGATION TEST")
    print("="*70)
    
    goal = (48, 48)
    
    # Test 1-hop episodic
    print("\n--- 1-hop Episodic ---")
    nav_1hop = Episodic2HopNavigator(maze)
    result_1hop = nav_1hop.navigate(goal, use_2hop=False, max_steps=3000)
    
    # Test 2-hop episodic
    print("\n--- 2-hop Episodic ---")
    nav_2hop = Episodic2HopNavigator(maze)
    result_2hop = nav_2hop.navigate(goal, use_2hop=True, max_steps=3000)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n1-hop Episodic:")
    print(f"  Success: {result_1hop['success']}")
    print(f"  Steps: {result_1hop['steps']}")
    print(f"  Episodes: {result_1hop['episode_count']}")
    print(f"  Efficiency: {result_1hop['efficiency']:.1f}%")
    
    print(f"\n2-hop Episodic:")
    print(f"  Success: {result_2hop['success']}")
    print(f"  Steps: {result_2hop['steps']}")
    print(f"  Episodes: {result_2hop['episode_count']}")
    print(f"  2-hop decisions: {result_2hop['two_hop_decisions']}")
    print(f"  Efficiency: {result_2hop['efficiency']:.1f}%")
    
    if result_1hop['steps'] > 0 and result_2hop['steps'] > 0:
        print(f"\nImprovement:")
        print(f"  Steps: {(1 - result_2hop['steps']/result_1hop['steps'])*100:+.1f}%")
        print(f"  Efficiency: {result_2hop['efficiency'] - result_1hop['efficiency']:+.1f}%")


if __name__ == "__main__":
    test_episodic_navigation()