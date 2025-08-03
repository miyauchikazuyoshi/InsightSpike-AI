"""Passage graph navigator that builds a graph of passable directions."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging

from ..environments.maze import MazeObservation
from ..config.maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class PassageNode:
    """A passable direction as a graph node."""
    position: Tuple[int, int]  # Position where passage exists
    direction: int  # Direction of passage (0-3)
    to_position: Tuple[int, int]  # Where this passage leads to
    vector: np.ndarray  # Feature vector
    
    def __hash__(self):
        return hash((self.position, self.direction))
    
    def __eq__(self, other):
        return self.position == other.position and self.direction == other.direction


class PassageGraphNavigator:
    """Navigator that treats passable directions as queries."""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        # Passage nodes: (position, direction) -> PassageNode
        self.passage_nodes: Dict[Tuple[Tuple[int, int], int], PassageNode] = {}
        # Edges between passages (continuity)
        self.passage_edges: Set[Tuple[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]]] = set()
        
        # Track state
        self.current_position = None
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.goal_position = None
        
        # Feature embedding
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0
        self.k_ig = 2.5  # Higher to encourage exploration
        
    def _init_embedder(self):
        """Initialize passage feature embedder."""
        np.random.seed(42)
        
        # Base passage vector (openness, possibility)
        self.passage_base = np.random.randn(self.feature_dim)
        self.passage_base /= np.linalg.norm(self.passage_base)
        
        # Direction vectors
        self.direction_vectors = {
            0: np.array([0.0, -1.0] + [0.0] * (self.feature_dim - 2)),  # Up
            1: np.array([1.0, 0.0] + [0.0] * (self.feature_dim - 2)),   # Right
            2: np.array([0.0, 1.0] + [0.0] * (self.feature_dim - 2)),   # Down
            3: np.array([-1.0, 0.0] + [0.0] * (self.feature_dim - 2))   # Left
        }
        
        for d in self.direction_vectors:
            self.direction_vectors[d] = self.direction_vectors[d][:self.feature_dim]
            self.direction_vectors[d] /= (np.linalg.norm(self.direction_vectors[d]) + 1e-8)
    
    def _get_passage_vector(self, position: Tuple[int, int], direction: int) -> np.ndarray:
        """Create feature vector for a passage."""
        # Base passage vector
        vector = 0.6 * self.passage_base
        
        # Add direction
        vector += 0.2 * self.direction_vectors[direction]
        
        # Add position encoding
        pos_encoding = np.array([
            np.sin(position[0] / 10),
            np.cos(position[0] / 10),
            np.sin(position[1] / 10),
            np.cos(position[1] / 10)
        ])
        
        if len(pos_encoding) < self.feature_dim:
            pos_encoding = np.pad(pos_encoding, (0, self.feature_dim - len(pos_encoding)))
        else:
            pos_encoding = pos_encoding[:self.feature_dim]
        
        vector += 0.2 * pos_encoding
        
        return vector / np.linalg.norm(vector)
    
    def _scan_passages(self, position: Tuple[int, int], observation: MazeObservation, maze) -> List[Tuple[int, Tuple[int, int]]]:
        """Scan for passable directions from current position."""
        passages = []
        
        for direction in observation.possible_moves:
            delta = maze.ACTIONS[direction]
            to_pos = (position[0] + delta[0], position[1] + delta[1])
            passages.append((direction, to_pos))
            
        return passages
    
    def _get_passage_weight(self, to_position: Tuple[int, int]) -> float:
        """Get weight for a passage based on whether destination is visited."""
        if to_position in self.visited_positions:
            return 0.1  # Much lower weight for passages to visited positions
        return 1.0  # Full weight for passages to unvisited positions
    
    def _update_passage_graph(self, position: Tuple[int, int], new_passages: List[Tuple[int, Tuple[int, int]]]):
        """Update passage graph with newly discovered passages."""
        new_nodes = []
        
        for direction, to_pos in new_passages:
            key = (position, direction)
            if key not in self.passage_nodes:
                # Create new passage node
                node = PassageNode(
                    position=position,
                    direction=direction,
                    to_position=to_pos,
                    vector=self._get_passage_vector(position, direction)
                )
                self.passage_nodes[key] = node
                new_nodes.append(node)
                print(f"  New passage: {position} -> {['up', 'right', 'down', 'left'][direction]} -> {to_pos}")
        
        # Connect passages that form continuous paths
        for node in new_nodes:
            # Check if there's a passage from the destination back
            for other_key, other_node in self.passage_nodes.items():
                # Continuous path: this passage leads to where another starts
                if node.to_position == other_node.position:
                    edge = ((node.position, node.direction), other_key)
                    self.passage_edges.add(edge)
                    print(f"    Connected: {node.position}->{node.to_position} continues to {other_node.to_position}")
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action based on passage discovery."""
        current_pos = observation.position
        self.current_position = current_pos
        self.visited_positions.add(current_pos)
        
        # Check if we found the goal
        if observation.is_goal and self.goal_position is None:
            self.goal_position = current_pos
            print(f"🎯 Goal discovered at {current_pos}!")
        
        # Scan passable directions as queries
        passages = self._scan_passages(current_pos, observation, maze)
        new_passages = [(d, to) for d, to in passages 
                        if (current_pos, d) not in self.passage_nodes]
        
        # Update passage graph
        if new_passages:
            print(f"Position {current_pos}: Discovered {len(new_passages)} new passages")
            self._update_passage_graph(current_pos, new_passages)
        
        # Evaluate each possible action
        action_scores = {}
        
        for action in observation.possible_moves:
            delta = maze.ACTIONS[action]
            next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # GED (movement cost)
            ged = 1.0
            
            # IG for discovering new passages
            ig = 0.0
            
            # Base IG based on whether we've been there
            if next_pos not in self.visited_positions:
                # Unvisited positions likely have undiscovered passages
                ig = 2.0
            else:
                ig = 0.1
            
            # Weight existing passages based on their destinations
            # Passages to unvisited areas are more valuable
            passage_key = (current_pos, action)
            if passage_key in self.passage_nodes:
                passage = self.passage_nodes[passage_key]
                weight = self._get_passage_weight(passage.to_position)
                ig *= weight  # Scale IG by passage weight
                
            
            # Bonus for positions that might reveal new passage continuity
            unvisited_connections = 0
            for _, node in self.passage_nodes.items():
                if next_pos == node.to_position and node.position not in self.visited_positions:
                    unvisited_connections += 1
            
            ig += 0.5 * unvisited_connections
            
            # Goal bonus
            if self.goal_position and next_pos == self.goal_position:
                ig += 10.0
            
            # geDIG objective
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = f
        
        if not action_scores:
            return 0
        
        # Choose best action
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a])
        
        # Exploration
        if np.random.random() < self.config.exploration_epsilon:
            return np.random.choice(observation.possible_moves)
        
        return best_action
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        return {
            'total_passages': len(self.passage_nodes),
            'total_connections': len(self.passage_edges),
            'positions_visited': len(self.visited_positions),
            'passage_positions': list(set(node.position for node in self.passage_nodes.values())),
            'graph_connectivity': len(self.passage_edges) / max(1, len(self.passage_nodes))
        }