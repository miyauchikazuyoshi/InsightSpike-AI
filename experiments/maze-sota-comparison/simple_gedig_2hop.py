#!/usr/bin/env python3
"""
Simple geDIG 2-hop Implementation
=================================

Simplified version focusing on the core concept of 2-hop GED evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import time
from datetime import datetime
import json


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


class SimpleGeDIG2HopNavigator:
    """Simple implementation of geDIG 2-hop evaluation for maze navigation"""
    
    def __init__(self, maze: np.ndarray, use_2hop: bool = True):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        self.use_2hop = use_2hop
        
        # Episode memory
        self.episodes: List[Episode7D] = []
        self.position_visits: Dict[Tuple[int, int], int] = {(1, 1): 1}
        
        # Episode graph
        self.episode_graph = nx.Graph()
        self.episode_graph.add_node("pos_1_1", pos=(1, 1))
        
        # Visual memory
        self.visual_memory: Dict[Tuple[int, int], Dict[str, str]] = {}
        
        # Statistics
        self.ged_decreases = []
        
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
        self.episode_graph.add_node("GOAL", type='goal')
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory and episode graph"""
        current_node = f"pos_{x}_{y}"
        
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
                
                # Add episode
                self.episodes.append(Episode7D(
                    x=nx, y=ny,
                    direction=direction,
                    result=None,
                    visit_count=self.position_visits.get((nx, ny), 0),
                    goal_or_not=False,
                    wall_or_path=wall_or_path
                ))
                
                # Update graph
                if wall_or_path == 'path':
                    neighbor_node = f"pos_{nx}_{ny}"
                    if neighbor_node not in self.episode_graph:
                        self.episode_graph.add_node(neighbor_node, pos=(nx, ny))
                    self.episode_graph.add_edge(current_node, neighbor_node, type='visual')
    
    def _calculate_simple_ged(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Simple GED calculation: node diff + edge diff"""
        node_diff = abs(g2.number_of_nodes() - g1.number_of_nodes())
        edge_diff = abs(g2.number_of_edges() - g1.number_of_edges())
        return node_diff + 0.5 * edge_diff
    
    def _evaluate_action_ged(self, action: str) -> Dict:
        """Evaluate action using GED change"""
        next_pos = self._get_next_position(self.position, action)
        
        if not self._is_valid_position(next_pos) or next_pos in self.visited:
            return {'ged_change': float('inf'), 'is_decrease': False}
        
        # Current graph state
        g1 = self.episode_graph.copy()
        
        # Hypothetical graph after action
        g2 = g1.copy()
        current_node = f"pos_{self.position[0]}_{self.position[1]}"
        next_node = f"pos_{next_pos[0]}_{next_pos[1]}"
        
        if next_node not in g2:
            g2.add_node(next_node, pos=next_pos)
        g2.add_edge(current_node, next_node, type='movement')
        
        # Add potential new connections
        for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
            nnx, nny = next_pos[0] + dx, next_pos[1] + dy
            neighbor_node = f"pos_{nnx}_{nny}"
            if neighbor_node in g2 and neighbor_node != current_node:
                g2.add_edge(next_node, neighbor_node, type='potential')
        
        # Calculate GED change
        ged_before = self._calculate_simple_ged(nx.Graph(), g1)
        ged_after = self._calculate_simple_ged(nx.Graph(), g2)
        ged_change = ged_after - ged_before
        
        # For 2-hop, check if this creates shortcuts
        if self.use_2hop and len(self.episodes) > 50:
            # Check 2-hop connectivity improvement
            shortcut_bonus = self._calculate_2hop_bonus(g1, g2, next_node)
            ged_change -= shortcut_bonus
        
        return {
            'ged_change': ged_change,
            'is_decrease': ged_change < 0,
            'nodes_before': g1.number_of_nodes(),
            'nodes_after': g2.number_of_nodes(),
            'edges_before': g1.number_of_edges(),
            'edges_after': g2.number_of_edges()
        }
    
    def _calculate_2hop_bonus(self, g1: nx.Graph, g2: nx.Graph, 
                             new_node: str) -> float:
        """Calculate bonus for creating 2-hop shortcuts"""
        if new_node not in g2:
            return 0.0
        
        # Count new 2-hop connections created
        new_2hop_connections = 0
        
        for neighbor in g2.neighbors(new_node):
            for n2 in g2.neighbors(neighbor):
                if n2 != new_node:
                    # Check if this 2-hop connection existed before
                    try:
                        path_before = nx.shortest_path_length(g1, new_node, n2)
                        if path_before > 2:
                            new_2hop_connections += 1
                    except:
                        new_2hop_connections += 1
        
        # Bonus for creating shortcuts
        return new_2hop_connections * 0.3
    
    def evaluate_actions(self) -> Dict:
        """Evaluate all possible actions"""
        x, y = self.position
        visual_info = self.visual_memory.get((x, y), {})
        
        action_evaluations = {}
        
        for action in ['up', 'down', 'left', 'right']:
            # Skip walls
            if action in visual_info and visual_info[action] == 'wall':
                continue
                
            eval_result = self._evaluate_action_ged(action)
            action_evaluations[action] = eval_result
        
        # Choose action with lowest GED change
        if action_evaluations:
            sorted_actions = sorted(action_evaluations.items(), 
                                  key=lambda x: x[1]['ged_change'])
            
            best_action, best_eval = sorted_actions[0]
            
            # Log GED decrease
            if best_eval['is_decrease']:
                self.ged_decreases.append({
                    'step': len(self.path),
                    'position': self.position,
                    'action': best_action,
                    'ged_change': best_eval['ged_change']
                })
                print(f"GED DECREASE! Action: {best_action}, "
                      f"Change: {best_eval['ged_change']:.2f}")
            
            return {
                'action': best_action,
                'evaluation': best_eval
            }
        
        return {'action': 'backtrack'}
    
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
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid and walkable"""
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.maze[y, x] == 0)
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 3000) -> Dict:
        """Navigate maze"""
        
        print(f"Simple geDIG {'2-hop' if self.use_2hop else '1-hop'} Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Evaluate actions
            decision = self.evaluate_actions()
            
            # Progress
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"visited={len(self.visited)}, GED_decreases={len(self.ged_decreases)}")
            
            # Execute
            action = decision['action']
            
            if action == 'backtrack':
                backtrack_count += 1
                if len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
            else:
                next_pos = self._get_next_position(self.position, action)
                
                # Update graph
                current_node = f"pos_{self.position[0]}_{self.position[1]}"
                next_node = f"pos_{next_pos[0]}_{next_pos[1]}"
                
                if next_node not in self.episode_graph:
                    self.episode_graph.add_node(next_node, pos=next_pos)
                self.episode_graph.add_edge(current_node, next_node, type='movement')
                
                # Move
                self.position = next_pos
                self.visited.add(next_pos)
                self.path.append(next_pos)
                self.position_visits[next_pos] = self.position_visits.get(next_pos, 0) + 1
                
                # Update visual memory
                self._update_visual_memory(next_pos[0], next_pos[1])
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        
        print(f"\nNavigation complete: success={success}, steps={steps}, "
              f"time={elapsed_time:.2f}s")
        print(f"GED decreases: {len(self.ged_decreases)}")
        print(f"Graph: {self.episode_graph.number_of_nodes()} nodes, "
              f"{self.episode_graph.number_of_edges()} edges")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'backtrack_count': backtrack_count,
            'ged_decrease_count': len(self.ged_decreases),
            'elapsed_time': elapsed_time,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0,
            'graph_nodes': self.episode_graph.number_of_nodes(),
            'graph_edges': self.episode_graph.number_of_edges()
        }


def test_simple_gedig():
    """Test simple geDIG implementation"""
    
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
    print("SIMPLE geDIG 2-HOP NAVIGATION TEST")
    print("="*70)
    
    goal = (size-2, size-2)
    
    # Test 1-hop
    print("\n--- 1-hop GED ---")
    nav_1hop = SimpleGeDIG2HopNavigator(maze, use_2hop=False)
    result_1hop = nav_1hop.navigate(goal, max_steps=2000)
    
    # Test 2-hop
    print("\n--- 2-hop GED ---")
    nav_2hop = SimpleGeDIG2HopNavigator(maze, use_2hop=True)
    result_2hop = nav_2hop.navigate(goal, max_steps=2000)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for name, result in [("1-hop", result_1hop), ("2-hop", result_2hop)]:
        print(f"\n{name}:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Efficiency: {result['efficiency']:.1f}%")
        print(f"  GED decreases: {result['ged_decrease_count']}")
        print(f"  Graph: {result['graph_nodes']} nodes, {result['graph_edges']} edges")
    
    if result_1hop['steps'] > 0 and result_2hop['steps'] > 0:
        print(f"\nImprovement with 2-hop:")
        print(f"  Steps: {(1 - result_2hop['steps']/result_1hop['steps'])*100:+.1f}%")
        print(f"  Efficiency: {result_2hop['efficiency'] - result_1hop['efficiency']:+.1f}%")


if __name__ == "__main__":
    test_simple_gedig()