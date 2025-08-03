#!/usr/bin/env python3
"""
True Episodic 2-hop Navigator with geDIG
========================================

Implements proper 2-hop evaluation on episode graph
using GED (Graph Edit Distance) decrease detection.
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


class TrueEpisodic2HopNavigator:
    """True implementation of episodic 2-hop navigation with GED analysis"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode memory
        self.episodes: List[Episode7D] = []
        self.position_visits: Dict[Tuple[int, int], int] = {(1, 1): 1}
        
        # Episode graph for GED analysis
        self.episode_graph = nx.Graph()
        self.position_to_episodes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # Statistics
        self.ged_decreases = []
        self.structural_insights = []
        
        # Initialize with goal episode
        self._add_goal_episode()
        self._record_visual_information(1, 1)
        
    def _add_goal_episode(self):
        """Add abstract goal episode"""
        goal_ep = Episode7D(
            x=None, y=None,
            direction=None,
            result=None,
            visit_count=0,
            goal_or_not=True,
            wall_or_path='path'
        )
        self.episodes.append(goal_ep)
        
        # Add to graph as special node
        self.episode_graph.add_node('GOAL', type='goal')
        
    def _record_visual_information(self, x: int, y: int):
        """Record visual information and update episode graph"""
        current_node = f"pos_{x}_{y}"
        
        if current_node not in self.episode_graph:
            self.episode_graph.add_node(current_node, pos=(x, y), type='position')
        
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
                
                ep_idx = len(self.episodes)
                self.episodes.append(Episode7D(
                    x=nx, y=ny,
                    direction=direction,
                    result=None,
                    visit_count=visit_count,
                    goal_or_not=False,
                    wall_or_path=wall_or_path
                ))
                
                # Update episode graph
                neighbor_node = f"pos_{nx}_{ny}"
                if wall_or_path == 'path':
                    if neighbor_node not in self.episode_graph:
                        self.episode_graph.add_node(neighbor_node, pos=(nx, ny), type='position')
                    self.episode_graph.add_edge(current_node, neighbor_node, type='visual')
                
                self.position_to_episodes[(nx, ny)].append(ep_idx)
    
    def _calculate_ged_change(self, action: str) -> float:
        """Calculate GED change if we take this action"""
        next_pos = self._get_next_position(self.position, action)
        
        if not self._is_valid_position(next_pos):
            return float('inf')  # Invalid action
        
        # Current graph metrics
        current_nodes = self.episode_graph.number_of_nodes()
        current_edges = self.episode_graph.number_of_edges()
        
        # Simulate adding new position
        next_node = f"pos_{next_pos[0]}_{next_pos[1]}"
        new_edges = 0
        
        if next_node not in self.episode_graph:
            # New node would be added
            new_edges += 1  # Edge from current position
            
            # Check potential connections to existing nodes
            for node in self.episode_graph.nodes():
                if node.startswith('pos_'):
                    node_data = self.episode_graph.nodes[node]
                    if 'pos' in node_data:
                        pos = node_data['pos']
                        dist = abs(pos[0] - next_pos[0]) + abs(pos[1] - next_pos[1])
                        if dist == 1:
                            new_edges += 1
        
        # Simple GED calculation
        ged_change = 1 + new_edges  # 1 for node, new_edges for edges
        
        # Check for 2-hop structural improvement
        two_hop_benefit = self._evaluate_2hop_benefit(next_pos)
        
        return ged_change - two_hop_benefit
    
    def _evaluate_2hop_benefit(self, next_pos: Tuple[int, int]) -> float:
        """Evaluate 2-hop structural benefit of moving to next_pos"""
        
        # Check if this position would create shortcuts
        shortcut_value = 0.0
        
        # Find nodes that would become 2-hop connected
        current_node = f"pos_{self.position[0]}_{self.position[1]}"
        next_node = f"pos_{next_pos[0]}_{next_pos[1]}"
        
        if current_node in self.episode_graph:
            current_neighbors = set(self.episode_graph.neighbors(current_node))
            
            # Simulate what would be 2-hop reachable from next_pos
            potential_2hop = set()
            for neighbor in current_neighbors:
                if neighbor.startswith('pos_'):
                    for n2 in self.episode_graph.neighbors(neighbor):
                        if n2 != current_node:
                            potential_2hop.add(n2)
            
            # High value if many new 2-hop connections
            shortcut_value = len(potential_2hop) * 0.5
            
            # Extra bonus if approaching unexplored areas
            if next_pos not in self.visited:
                shortcut_value += 2.0
        
        return shortcut_value
    
    def evaluate_1hop(self) -> Dict:
        """1-hop evaluation using episode similarity"""
        # Simple greedy approach
        best_action = None
        best_score = float('-inf')
        
        for action in ['up', 'down', 'left', 'right']:
            next_pos = self._get_next_position(self.position, action)
            
            if self._is_valid_position(next_pos) and next_pos not in self.visited:
                # Simple heuristic: prefer unexplored
                score = 1.0
                
                # Bonus for low visit count
                visit_count = self.position_visits.get(next_pos, 0)
                score += 1.0 / (1.0 + visit_count)
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        return {
            'action': best_action or 'backtrack',
            'type': '1hop',
            'score': best_score
        }
    
    def evaluate_2hop(self) -> Dict:
        """2-hop evaluation using GED analysis"""
        
        # Need sufficient episodes
        if len(self.episodes) < 30:
            return self.evaluate_1hop()
        
        # Evaluate GED change for each action
        action_ged = {}
        for action in ['up', 'down', 'left', 'right']:
            ged_change = self._calculate_ged_change(action)
            action_ged[action] = ged_change
        
        # Find action with best (lowest) GED change
        best_action = min(action_ged.items(), key=lambda x: x[1])
        
        # Detect structural insight (GED decrease)
        if best_action[1] < 0:
            self.ged_decreases.append({
                'position': self.position,
                'action': best_action[0],
                'ged_change': best_action[1],
                'episode_count': len(self.episodes)
            })
            
            return {
                'action': best_action[0],
                'type': '2hop',
                'reason': 'ged_decrease',
                'ged_change': best_action[1]
            }
        
        # Check for dead-end pattern
        if all(ged >= 3 for ged in action_ged.values()):
            # All actions increase complexity significantly
            # Look for backtrack opportunity
            for i in range(len(self.path) - 2, max(0, len(self.path) - 20), -1):
                old_pos = self.path[i]
                unvisited = sum(1 for a in ['up', 'down', 'left', 'right']
                              if self._is_valid_position(self._get_next_position(old_pos, a))
                              and self._get_next_position(old_pos, a) not in self.visited)
                if unvisited >= 2:
                    return {
                        'action': 'backtrack',
                        'target': old_pos,
                        'type': '2hop',
                        'reason': 'high_ged_all_directions'
                    }
        
        # Default to best GED action
        return {
            'action': best_action[0],
            'type': '2hop',
            'reason': 'lowest_ged',
            'ged_change': best_action[1]
        }
    
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
    
    def navigate(self, goal: Tuple[int, int], use_2hop: bool = True,
                max_steps: int = 3000) -> Dict:
        """Navigate using true 2-hop episodic evaluation"""
        
        print(f"True Episodic 2-hop Navigation: 2-hop={use_2hop}")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        two_hop_decisions = 0
        ged_decrease_count = 0
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Evaluate
            if use_2hop:
                decision = self.evaluate_2hop()
            else:
                decision = self.evaluate_1hop()
            
            if decision['type'] == '2hop':
                two_hop_decisions += 1
                if decision.get('reason') == 'ged_decrease':
                    ged_decrease_count += 1
            
            # Progress report
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"episodes={len(self.episodes)}, 2hop={two_hop_decisions}, "
                      f"GED_decreases={ged_decrease_count}")
            
            # Execute action
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
                    
                    # Update graph
                    current_node = f"pos_{self.position[0]}_{self.position[1]}"
                    next_node = f"pos_{next_pos[0]}_{next_pos[1]}"
                    
                    if next_node not in self.episode_graph:
                        self.episode_graph.add_node(next_node, pos=next_pos, type='position')
                    self.episode_graph.add_edge(current_node, next_node, type='movement')
                    
                    # Move
                    self.position = next_pos
                    self.visited.add(next_pos)
                    self.path.append(next_pos)
                    self.position_visits[next_pos] = self.position_visits.get(next_pos, 0) + 1
                    
                    # Record new visual information
                    self._record_visual_information(next_pos[0], next_pos[1])
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        
        print(f"\nNavigation complete: success={success}, steps={steps}, "
              f"time={elapsed_time:.2f}s")
        print(f"Episodes: {len(self.episodes)}, 2-hop decisions: {two_hop_decisions}")
        print(f"GED decreases detected: {ged_decrease_count}")
        print(f"Graph nodes: {self.episode_graph.number_of_nodes()}, "
              f"edges: {self.episode_graph.number_of_edges()}")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'episode_count': len(self.episodes),
            'backtrack_count': backtrack_count,
            'two_hop_decisions': two_hop_decisions,
            'ged_decrease_count': ged_decrease_count,
            'graph_nodes': self.episode_graph.number_of_nodes(),
            'graph_edges': self.episode_graph.number_of_edges(),
            'elapsed_time': elapsed_time,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0,
            'ged_decreases': self.ged_decreases
        }


def test_true_episodic():
    """Test true episodic 2-hop navigation"""
    
    # Generate maze
    np.random.seed(42)
    size = 30  # Smaller for faster testing
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
    
    print("="*70)
    print("TRUE EPISODIC 2-HOP NAVIGATION TEST")
    print("="*70)
    
    goal = (size-2, size-2)
    
    # Test 1-hop
    print("\n--- 1-hop True Episodic ---")
    nav_1hop = TrueEpisodic2HopNavigator(maze)
    result_1hop = nav_1hop.navigate(goal, use_2hop=False, max_steps=2000)
    
    # Test 2-hop
    print("\n--- 2-hop True Episodic ---")
    nav_2hop = TrueEpisodic2HopNavigator(maze)
    result_2hop = nav_2hop.navigate(goal, use_2hop=True, max_steps=2000)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n1-hop True Episodic:")
    print(f"  Success: {result_1hop['success']}")
    print(f"  Steps: {result_1hop['steps']}")
    print(f"  Episodes: {result_1hop['episode_count']}")
    print(f"  Efficiency: {result_1hop['efficiency']:.1f}%")
    print(f"  Graph: {result_1hop['graph_nodes']} nodes, {result_1hop['graph_edges']} edges")
    
    print(f"\n2-hop True Episodic:")
    print(f"  Success: {result_2hop['success']}")
    print(f"  Steps: {result_2hop['steps']}")
    print(f"  Episodes: {result_2hop['episode_count']}")
    print(f"  2-hop decisions: {result_2hop['two_hop_decisions']}")
    print(f"  GED decreases: {result_2hop['ged_decrease_count']}")
    print(f"  Efficiency: {result_2hop['efficiency']:.1f}%")
    print(f"  Graph: {result_2hop['graph_nodes']} nodes, {result_2hop['graph_edges']} edges")
    
    if result_1hop['steps'] > 0 and result_2hop['steps'] > 0:
        print(f"\nImprovement:")
        print(f"  Steps: {(1 - result_2hop['steps']/result_1hop['steps'])*100:+.1f}%")
        print(f"  Efficiency: {result_2hop['efficiency'] - result_1hop['efficiency']:+.1f}%")
    
    # Analyze GED decreases
    if result_2hop['ged_decreases']:
        print(f"\nGED Decrease Analysis:")
        print(f"  Total occurrences: {len(result_2hop['ged_decreases'])}")
        for i, decrease in enumerate(result_2hop['ged_decreases'][:5]):  # First 5
            print(f"  {i+1}. At pos {decrease['position']}, "
                  f"action={decrease['action']}, "
                  f"GED change={decrease['ged_change']:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'maze_size': size,
        '1hop': result_1hop,
        '2hop': result_2hop
    }
    
    with open(f'true_episodic_2hop_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to true_episodic_2hop_results_{timestamp}.json")


if __name__ == "__main__":
    test_true_episodic()