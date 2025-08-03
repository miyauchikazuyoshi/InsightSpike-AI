"""Wall graph navigator that builds a graph of wall connections."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging

from ..environments.maze import MazeObservation
from ..config.maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class WallNode:
    """A wall as a graph node."""
    position: Tuple[int, int]  # Wall position in maze
    vector: np.ndarray  # Feature vector
    
    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        return self.position == other.position


class WallGraphNavigator:
    """Navigator that builds a graph where nodes are walls and edges are wall connections."""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.wall_nodes: Dict[Tuple[int, int], WallNode] = {}  # Wall position -> node
        self.wall_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()  # Wall connections
        
        # Track agent state
        self.current_position = None
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.known_walls: Set[Tuple[int, int]] = set()  # All discovered wall positions
        self.goal_position = None  # Will be set when discovered
        
        # Feature embedding
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0
        self.k_ig = 2.0
        
    def _init_embedder(self):
        """Initialize wall feature embedder."""
        np.random.seed(42)
        
        # Base wall vector
        self.wall_base = np.random.randn(self.feature_dim)
        self.wall_base /= np.linalg.norm(self.wall_base)
        
    def _get_wall_vector(self, wall_pos: Tuple[int, int]) -> np.ndarray:
        """Create feature vector for a wall based on its position."""
        # Position encoding
        pos_encoding = np.array([
            np.sin(wall_pos[0] / 10),
            np.cos(wall_pos[0] / 10),
            np.sin(wall_pos[1] / 10),
            np.cos(wall_pos[1] / 10)
        ])
        
        if len(pos_encoding) < self.feature_dim:
            pos_encoding = np.pad(pos_encoding, (0, self.feature_dim - len(pos_encoding)))
        else:
            pos_encoding = pos_encoding[:self.feature_dim]
        
        # Combine base vector with position encoding
        vector = 0.7 * self.wall_base + 0.3 * pos_encoding
        return vector / np.linalg.norm(vector)
    
    def _scan_for_walls(self, position: Tuple[int, int], observation: MazeObservation) -> List[Tuple[int, int]]:
        """Scan all 4 directions from current position to find walls."""
        walls = []
        
        # Check each direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        
        for i, (dx, dy) in enumerate(directions):
            if i not in observation.possible_moves:
                # There's a wall in this direction
                wall_pos = (position[0] + dx, position[1] + dy)
                walls.append(wall_pos)
                
        return walls
    
    def _find_wall_connections(self, wall_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find other walls connected to this wall (adjacent walls)."""
        connected = []
        
        # Check 4-connected neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (wall_pos[0] + dx, wall_pos[1] + dy)
            if neighbor in self.known_walls:
                connected.append(neighbor)
                
        return connected
    
    def _update_wall_graph(self, new_walls: List[Tuple[int, int]]):
        """Update the wall graph with newly discovered walls."""
        # First, add all new wall nodes
        for wall_pos in new_walls:
            if wall_pos not in self.wall_nodes:
                # Create new wall node
                node = WallNode(
                    position=wall_pos,
                    vector=self._get_wall_vector(wall_pos)
                )
                self.wall_nodes[wall_pos] = node
                self.known_walls.add(wall_pos)
                logger.debug(f"Added wall node at {wall_pos}")
        
        # Connect newly discovered walls to each other if adjacent
        for i, wall1 in enumerate(new_walls):
            for wall2 in new_walls[i+1:]:
                # Check if walls are adjacent (Manhattan distance = 1)
                manhattan_dist = abs(wall1[0] - wall2[0]) + abs(wall1[1] - wall2[1])
                if manhattan_dist == 1:
                    # Add edge (undirected)
                    edge = tuple(sorted([wall1, wall2]))
                    self.wall_edges.add(edge)
                    print(f"  Connected new walls: {wall1} <-> {wall2}")
        
        # Also connect new walls to existing adjacent walls
        for new_wall in new_walls:
            for existing_wall in self.known_walls:
                if existing_wall not in new_walls:  # Don't double-count
                    manhattan_dist = abs(new_wall[0] - existing_wall[0]) + abs(new_wall[1] - existing_wall[1])
                    if manhattan_dist == 1:
                        edge = tuple(sorted([new_wall, existing_wall]))
                        self.wall_edges.add(edge)
                        print(f"  Connected new wall {new_wall} to existing wall {existing_wall}")
    
    def _calculate_wall_discovery_potential(self, position: Tuple[int, int]) -> float:
        """Estimate potential for discovering new walls from a position."""
        # Walls represent high entropy (uncertainty) regions
        # The potential is based on proximity to unknown high-entropy areas
        
        if not self.known_walls:
            return 1.0  # Maximum entropy when no walls known
            
        # Calculate entropy field around known walls
        # Near walls but not on them = high discovery potential
        wall_distances = [
            abs(position[0] - wall[0]) + abs(position[1] - wall[1])
            for wall in self.known_walls
        ]
        min_dist = min(wall_distances)
        
        # Entropy decreases with distance from walls
        # But very close (distance 1) might be another wall
        if min_dist == 1:
            return 2.0  # Very high - likely to find adjacent walls
        elif min_dist == 2:
            return 1.5  # High - wall edges
        else:
            return 1.0 / (min_dist + 1)  # Decreases with distance
        
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action based on wall graph exploration."""
        current_pos = observation.position
        self.current_position = current_pos
        self.visited_positions.add(current_pos)
        
        # Check if we found the goal
        if observation.is_goal and self.goal_position is None:
            self.goal_position = current_pos
            print(f"🎯 Goal discovered at {current_pos}!")
        
        # Scan for walls from current position
        visible_walls = self._scan_for_walls(current_pos, observation)
        new_walls = [w for w in visible_walls if w not in self.known_walls]
        
        # Update wall graph with only new (unknown) walls
        if new_walls:
            self._update_wall_graph(new_walls)
            print(f"Position {current_pos}: Discovered {len(new_walls)} new walls: {new_walls}")
        
        # Evaluate each possible action
        action_scores = {}
        
        for action in observation.possible_moves:
            delta = maze.ACTIONS[action]
            next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # GED (movement cost in low-entropy space)
            ged = 1.0
            
            # IG (information gain from high-entropy regions)
            # Walls = high entropy, paths = low entropy
            wall_potential = self._calculate_wall_discovery_potential(next_pos)
            
            if next_pos not in self.visited_positions:
                # Unvisited position - base IG + entropy potential
                ig = 1.0 + wall_potential
            else:
                # Visited position - only entropy potential matters
                ig = 0.2 * wall_potential
                
            # Special bonus: if we can see new walls from next position
            # (simulate looking ahead for high-entropy regions)
            expected_new_walls = 0
            for d in range(4):
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][d]
                wall_check_pos = (next_pos[0] + dx, next_pos[1] + dy)
                if wall_check_pos not in self.known_walls:
                    expected_new_walls += 0.25
            
            ig += expected_new_walls
            
            # Goal bonus if we know where it is
            if self.goal_position and next_pos == self.goal_position:
                ig += 10.0  # Huge bonus for reaching goal
            
            # geDIG objective
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = f
            
        if not action_scores:
            return 0
            
        # Find actions with minimum score
        min_score = min(action_scores.values())
        best_actions = [a for a, score in action_scores.items() if abs(score - min_score) < 1e-6]
        
        if len(best_actions) > 1:
            # Multiple equally good actions - use secondary criteria
            # Option 1: Random selection among equals
            if not hasattr(self, 'start_position') or self.start_position is None:
                self.start_position = (1, 1)  # Default start
                
            # Option 2: Prefer moving away from start (exploration bias)
            def distance_from_start(action):
                delta = maze.ACTIONS[action]
                next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                return abs(next_pos[0] - self.start_position[0]) + abs(next_pos[1] - self.start_position[1])
            
            # Sort by distance from start (descending)
            best_actions.sort(key=distance_from_start, reverse=True)
            best_action = best_actions[0]
            
            print(f"  Multiple equal actions, chose {maze.ACTION_NAMES[best_action]} (furthest from start)")
        else:
            best_action = best_actions[0]
        
        # Exploration
        if np.random.random() < self.config.exploration_epsilon:
            return np.random.choice(observation.possible_moves)
            
        return best_action
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        return {
            'total_walls': len(self.wall_nodes),
            'total_edges': len(self.wall_edges),
            'positions_visited': len(self.visited_positions),
            'wall_positions': list(self.known_walls),
            'graph_connectivity': len(self.wall_edges) / max(1, len(self.wall_nodes))
        }