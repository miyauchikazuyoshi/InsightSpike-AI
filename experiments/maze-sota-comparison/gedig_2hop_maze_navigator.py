#!/usr/bin/env python3
"""
geDIG 2-hop Maze Navigator
==========================

Uses actual geDIG normalized calculator with 2-hop evaluation
for maze navigation based on episode graphs.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import time
from datetime import datetime
import json

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from insightspike.algorithms.gedig_core_normalize import GeDIGNormalizedCalculator

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


class GeDIG2HopMazeNavigator:
    """Maze navigator using actual geDIG 2-hop evaluation"""
    
    def __init__(self, maze: np.ndarray, use_2hop: bool = True):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode memory
        self.episodes: List[Episode7D] = []
        self.position_visits: Dict[Tuple[int, int], int] = {(1, 1): 1}
        
        # Episode graph for geDIG calculation
        self.episode_graph = nx.Graph()
        self.episode_graph.add_node("pos_1_1", pos=(1, 1))
        
        # Visual memory
        self.visual_memory: Dict[Tuple[int, int], Dict[str, str]] = {}
        
        # geDIG calculator configuration
        config = {
            "node_cost": 1.0,
            "edge_cost": 0.5,
            "graph": {
                "metrics": {
                    "use_multihop_gedig": use_2hop,
                    "max_hops": 2,
                    "decay_factor": 0.7
                }
            },
            "normalization": {
                "enabled": True,
                "mode": "conservation",
                "size_normalization": {"beta": 0.5},
                "reward": {"lambda_ig": 1.0, "mu_ged": 0.5},
                "spike_detection": {"mode": "threshold", "threshold": -0.5},
                "z_transform": {"use_running_stats": True, "window_size": 50}
            }
        }
        self.gedig_calculator = GeDIGNormalizedCalculator(config)
        
        # Statistics
        self.gedig_history = []
        self.spike_count = 0
        
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
        # Add goal node to graph
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
                
                # Add to episode list
                self.episodes.append(Episode7D(
                    x=nx, y=ny,
                    direction=direction,
                    result=None,
                    visit_count=self.position_visits.get((nx, ny), 0),
                    goal_or_not=False,
                    wall_or_path=wall_or_path
                ))
                
                # Update graph if it's a path
                if wall_or_path == 'path':
                    neighbor_node = f"pos_{nx}_{ny}"
                    if neighbor_node not in self.episode_graph:
                        self.episode_graph.add_node(neighbor_node, pos=(nx, ny))
                    self.episode_graph.add_edge(current_node, neighbor_node, type='visual')
    
    def _evaluate_action_with_gedig(self, action: str) -> Dict:
        """Evaluate an action using geDIG calculation"""
        next_pos = self._get_next_position(self.position, action)
        
        # Check if valid
        if not self._is_valid_position(next_pos) or next_pos in self.visited:
            return {'gedig': float('inf'), 'spike': False}
        
        # Create hypothetical graph after taking this action
        graph_after = self.episode_graph.copy()
        current_node = f"pos_{self.position[0]}_{self.position[1]}"
        next_node = f"pos_{next_pos[0]}_{next_pos[1]}"
        
        # Add node if new
        if next_node not in graph_after:
            graph_after.add_node(next_node, pos=next_pos)
        
        # Add movement edge
        graph_after.add_edge(current_node, next_node, type='movement')
        
        # Add potential visual connections
        for direction, (dx, dy) in [('right', (1, 0)), ('left', (-1, 0)), 
                                    ('up', (0, -1)), ('down', (0, 1))]:
            nnx, nny = next_pos[0] + dx, next_pos[1] + dy
            if 0 <= nnx < self.width and 0 <= nny < self.height:
                if self.maze[nny, nnx] == 0:
                    potential_node = f"pos_{nnx}_{nny}"
                    if potential_node in graph_after:
                        graph_after.add_edge(next_node, potential_node, type='potential')
        
        # Calculate geDIG
        try:
            result = self.gedig_calculator.calculate(
                self.episode_graph, 
                graph_after,
                focal_nodes={current_node, next_node}
            )
            
            return {
                'gedig': result['gedig'],
                'spike': result['has_spike'],
                'ged': result['ged'],
                'ig': result['ig'],
                'normalized_ged': result['normalized_metrics']['ged_normalized'],
                'ig_z_score': result['normalized_metrics']['ig_z_score']
            }
        except Exception as e:
            print(f"geDIG calculation error: {e}")
            return {'gedig': 0.0, 'spike': False}
    
    def evaluate_actions(self) -> Dict:
        """Evaluate all possible actions using geDIG"""
        x, y = self.position
        visual_info = self.visual_memory.get((x, y), {})
        
        action_evaluations = {}
        
        for action in ['up', 'down', 'left', 'right']:
            # Check visual info first
            if action in visual_info and visual_info[action] == 'wall':
                continue  # Skip walls
                
            eval_result = self._evaluate_action_with_gedig(action)
            action_evaluations[action] = eval_result
        
        # Choose action with lowest (most negative) geDIG
        if action_evaluations:
            # Sort by geDIG value (ascending)
            sorted_actions = sorted(action_evaluations.items(), 
                                  key=lambda x: x[1]['gedig'])
            
            best_action, best_eval = sorted_actions[0]
            
            # Log if spike detected
            if best_eval['spike']:
                self.spike_count += 1
                print(f"SPIKE detected! Action: {best_action}, geDIG: {best_eval['gedig']:.3f}")
            
            return {
                'action': best_action,
                'evaluation': best_eval,
                'all_evaluations': action_evaluations
            }
        
        # No valid actions - backtrack
        return {'action': 'backtrack', 'evaluation': {'gedig': 0.0, 'spike': False}}
    
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
        """Navigate maze using geDIG 2-hop evaluation"""
        
        print(f"geDIG 2-hop Maze Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        print(f"Multi-hop enabled: {self.gedig_calculator._base_calculator.enable_multihop}")
        
        steps = 0
        backtrack_count = 0
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Evaluate actions
            decision = self.evaluate_actions()
            
            # Progress report
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"visited={len(self.visited)}, spikes={self.spike_count}")
            
            # Log geDIG values
            if decision.get('evaluation'):
                self.gedig_history.append({
                    'step': steps,
                    'position': self.position,
                    'action': decision['action'],
                    'gedig': decision['evaluation'].get('gedig', 0),
                    'spike': decision['evaluation'].get('spike', False)
                })
            
            # Execute action
            action = decision['action']
            
            if action == 'backtrack':
                backtrack_count += 1
                if len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
            else:
                next_pos = self._get_next_position(self.position, action)
                
                # Update graph with actual movement
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
        print(f"Total spikes detected: {self.spike_count}")
        print(f"Graph size: {self.episode_graph.number_of_nodes()} nodes, "
              f"{self.episode_graph.number_of_edges()} edges")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'episode_count': len(self.episodes),
            'backtrack_count': backtrack_count,
            'spike_count': self.spike_count,
            'elapsed_time': elapsed_time,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0,
            'gedig_history': self.gedig_history[-100:],  # Last 100 entries
            'graph_nodes': self.episode_graph.number_of_nodes(),
            'graph_edges': self.episode_graph.number_of_edges()
        }


def test_gedig_navigation():
    """Test geDIG-based navigation"""
    
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
    print("geDIG 2-HOP MAZE NAVIGATION TEST")
    print("="*70)
    
    goal = (size-2, size-2)
    
    # Test 1-hop geDIG
    print("\n--- 1-hop geDIG ---")
    nav_1hop = GeDIG2HopMazeNavigator(maze, use_2hop=False)
    result_1hop = nav_1hop.navigate(goal, max_steps=2000)
    
    # Test 2-hop geDIG
    print("\n--- 2-hop geDIG ---")
    nav_2hop = GeDIG2HopMazeNavigator(maze, use_2hop=True)
    result_2hop = nav_2hop.navigate(goal, max_steps=2000)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    for name, result in [("1-hop geDIG", result_1hop), ("2-hop geDIG", result_2hop)]:
        print(f"\n{name}:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Efficiency: {result['efficiency']:.1f}%")
        print(f"  Spikes: {result['spike_count']}")
        print(f"  Graph: {result['graph_nodes']} nodes, {result['graph_edges']} edges")
    
    # Analyze spike patterns
    if result_2hop['spike_count'] > 0:
        print(f"\nSpike Analysis (2-hop):")
        spike_events = [h for h in result_2hop['gedig_history'] if h['spike']]
        for i, event in enumerate(spike_events[:5]):  # First 5 spikes
            print(f"  Spike {i+1}: step={event['step']}, "
                  f"pos={event['position']}, action={event['action']}, "
                  f"geDIG={event['gedig']:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'maze_size': size,
        '1hop_gedig': result_1hop,
        '2hop_gedig': result_2hop
    }
    
    with open(f'gedig_2hop_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to gedig_2hop_results_{timestamp}.json")


if __name__ == "__main__":
    test_gedig_navigation()