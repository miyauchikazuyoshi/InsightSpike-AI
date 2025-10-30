"""Wall-only geDIG navigator that memorizes only walls as queries."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging

from ...environments.maze import MazeObservation
from ..maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class WallNode:
    """Memory node for a wall."""
    position: Tuple[int, int]  # Wall position (estimated)
    from_position: Tuple[int, int]  # Position where we detected this wall
    direction: int  # Direction from from_position to wall (0-3)
    vector: np.ndarray  # Feature vector for this wall
    information_gain: float  # IG when discovered
    
    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        return self.position == other.position


class WallOnlyGeDIGNavigator:
    """Navigator that only memorizes walls as new information."""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.wall_nodes: Dict[Tuple[int, int], WallNode] = {}
        self.current_episode = 0
        
        # Feature embeddings
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0
        self.k_ig = 2.0  # Higher IG weight to encourage wall discovery
        
        # Track current position
        self.current_position = None
        self.visited_positions: Set[Tuple[int, int]] = set()
        
    def _init_embedder(self):
        """Initialize feature embedder for walls."""
        np.random.seed(42)
        
        # Wall direction vectors - each wall has a directional feature
        self.wall_base_vector = np.random.randn(self.feature_dim)
        self.wall_base_vector /= np.linalg.norm(self.wall_base_vector)
        
        # Direction modifiers
        self.direction_vectors = {
            0: np.array([0.0, -1.0] + [0.0] * (self.feature_dim - 2)),  # Wall above
            1: np.array([1.0, 0.0] + [0.0] * (self.feature_dim - 2)),   # Wall right
            2: np.array([0.0, 1.0] + [0.0] * (self.feature_dim - 2)),   # Wall below
            3: np.array([-1.0, 0.0] + [0.0] * (self.feature_dim - 2))   # Wall left
        }
        
        for d in self.direction_vectors:
            self.direction_vectors[d] = self.direction_vectors[d][:self.feature_dim]
            self.direction_vectors[d] /= (np.linalg.norm(self.direction_vectors[d]) + 1e-8)
    
    def _get_wall_vector(self, wall_pos: Tuple[int, int], direction: int) -> np.ndarray:
        """Create feature vector for a wall."""
        # Base wall vector
        vector = 0.6 * self.wall_base_vector
        
        # Add direction information
        vector += 0.2 * self.direction_vectors[direction]
        
        # Add position encoding
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
        
        vector += 0.2 * pos_encoding
        
        return vector / np.linalg.norm(vector)
    
    def _calculate_wall_position(self, from_pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Calculate wall position from current position and direction."""
        if direction == 0:  # Up
            return (from_pos[0] - 1, from_pos[1])
        elif direction == 1:  # Right
            return (from_pos[0], from_pos[1] + 1)
        elif direction == 2:  # Down
            return (from_pos[0] + 1, from_pos[1])
        else:  # Left
            return (from_pos[0], from_pos[1] - 1)
    
    def _detect_new_walls(self, observation: MazeObservation, maze) -> List[Tuple[int, Tuple[int, int]]]:
        """Detect walls that we haven't memorized yet."""
        new_walls = []
        current_pos = observation.position
        
        # Check all 4 directions
        for direction in range(4):
            # Check if this direction is blocked
            if direction not in observation.possible_moves:
                wall_pos = self._calculate_wall_position(current_pos, direction)
                
                # Check if we already know about this wall
                if wall_pos not in self.wall_nodes:
                    new_walls.append((direction, wall_pos))
                    
        return new_walls
    
    def _calculate_information_gain(self, wall_pos: Tuple[int, int], 
                                   from_pos: Tuple[int, int]) -> float:
        """Calculate IG for discovering a new wall."""
        # Base IG for any wall discovery
        ig = 1.0
        
        # Bonus for walls discovered from unvisited positions
        if from_pos not in self.visited_positions:
            ig += 0.5
        
        # Bonus for walls in unexplored areas (far from other known walls)
        if self.wall_nodes:
            min_dist_to_known = min(
                abs(wall_pos[0] - known.position[0]) + abs(wall_pos[1] - known.position[1])
                for known in self.wall_nodes.values()
            )
            if min_dist_to_known > 3:
                ig += 0.5
        
        return ig
    
    def _donut_search_walls(self, current_pos: Tuple[int, int]) -> List[WallNode]:
        """Find relevant walls using donut search."""
        relevant_walls = []
        
        for wall_node in self.wall_nodes.values():
            # Distance from current position to wall
            dist = np.sqrt(
                (wall_node.position[0] - current_pos[0])**2 + 
                (wall_node.position[1] - current_pos[1])**2
            )
            
            # Donut search
            if self.config.donut_inner_radius < dist <= self.config.donut_outer_radius:
                relevant_walls.append(wall_node)
        
        return relevant_walls
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action based on wall discovery potential."""
        current_pos = observation.position
        self.current_position = current_pos
        self.visited_positions.add(current_pos)
        
        # Detect and memorize new walls
        new_walls = self._detect_new_walls(observation, maze)
        for direction, wall_pos in new_walls:
            ig = self._calculate_information_gain(wall_pos, current_pos)
            
            wall_node = WallNode(
                position=wall_pos,
                from_position=current_pos,
                direction=direction,
                vector=self._get_wall_vector(wall_pos, direction),
                information_gain=ig
            )
            
            self.wall_nodes[wall_pos] = wall_node
            logger.debug(f"Discovered wall at {wall_pos} from {current_pos}, direction {direction}, IG={ig:.2f}")
        
        # Get relevant walls for decision making
        nearby_walls = self._donut_search_walls(current_pos)
        
        # Evaluate each possible action
        action_scores = {}
        
        for action in observation.possible_moves:
            # Calculate next position
            delta = maze.ACTIONS[action]
            next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Estimate potential wall discoveries from next position
            potential_new_walls = 0
            for check_dir in range(4):
                potential_wall_pos = self._calculate_wall_position(next_pos, check_dir)
                if potential_wall_pos not in self.wall_nodes:
                    potential_new_walls += 1
            
            # Calculate GED (distance to move)
            ged = 1.0  # Simple unit distance
            
            # Calculate IG (potential for wall discovery)
            ig = 0.0
            
            # High IG for positions with potential new wall discoveries
            if potential_new_walls > 0:
                ig = potential_new_walls * 0.5
            
            # Bonus IG for unvisited positions
            if next_pos not in self.visited_positions:
                ig += 1.0
            
            # Reduce IG if we're moving toward known walls (we already know them)
            for wall in nearby_walls:
                wall_dist = abs(next_pos[0] - wall.position[0]) + abs(next_pos[1] - wall.position[1])
                if wall_dist < 2:
                    ig *= 0.5
            
            # geDIG objective: f = w*GED - k*IG
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = f
            
            logger.debug(f"  Action {maze.ACTION_NAMES[action]}: to {next_pos}, "
                        f"potential walls={potential_new_walls}, GED={ged:.2f}, IG={ig:.2f}, f={f:.2f}")
        
        if not action_scores:
            logger.warning(f"No valid actions at {current_pos}")
            return 0
        
        # Choose best action (lowest f)
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a])
        
        # Exploration
        if np.random.random() < self.config.exploration_epsilon:
            return np.random.choice(observation.possible_moves)
        
        return best_action
    
    def new_episode(self):
        """New episode - keep wall memory but reset position tracking."""
        self.current_episode += 1
        self.current_position = None
        # Keep wall memory across episodes!
        # Only reset visited positions for exploration calculation
        self.visited_positions.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        return {
            'total_walls_discovered': len(self.wall_nodes),
            'total_positions_visited': len(self.visited_positions),
            'episodes': self.current_episode,
            'wall_positions': list(self.wall_nodes.keys()),
            'total_information_gain': sum(w.information_gain for w in self.wall_nodes.values())
        }