"""Wall-aware geDIG navigator that learns wall locations."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging

from ...environments.maze import MazeObservation
from ..maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class WallMemory:
    """Memory of a wall location."""
    from_pos: Tuple[int, int]  # Position where we hit the wall from
    direction: int  # Direction we tried to move (0-3)
    wall_pos: Tuple[int, int]  # Calculated wall position
    
    def __hash__(self):
        return hash((self.from_pos, self.direction))


@dataclass
class LocationMemory:
    """Enhanced memory node with wall information."""
    position: Tuple[int, int]
    node_type: str
    vector: np.ndarray
    information_gain: float
    visits: int = 1
    known_walls: Set[int] = None  # Directions where walls exist (0-3)
    
    def __post_init__(self):
        if self.known_walls is None:
            self.known_walls = set()


class WallAwareGeDIGNavigator:
    """Navigator that explicitly learns and remembers wall locations."""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.location_memory: Dict[Tuple[int, int], LocationMemory] = {}
        self.wall_memory: Set[WallMemory] = set()
        self.current_episode = 0
        
        # Feature embeddings
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0
        self.k_ig = 1.5
        
        # Track last position to detect wall hits
        self.last_position = None
        self.last_action = None
        
    def _init_embedder(self):
        """Initialize feature embedder."""
        np.random.seed(42)
        
        self.type_vectors = {
            'wall': np.random.randn(self.feature_dim),
            'junction': np.random.randn(self.feature_dim),
            'corridor': np.random.randn(self.feature_dim),
            'dead_end': np.random.randn(self.feature_dim),
            'goal': np.random.randn(self.feature_dim),
            'unknown': np.zeros(self.feature_dim)
        }
        
        # Add directional wall vectors
        self.wall_direction_vectors = {
            0: np.array([0, -1] + [0] * (self.feature_dim - 2)),  # Up
            1: np.array([1, 0] + [0] * (self.feature_dim - 2)),   # Right
            2: np.array([0, 1] + [0] * (self.feature_dim - 2)),   # Down
            3: np.array([-1, 0] + [0] * (self.feature_dim - 2))   # Left
        }
        
        # Normalize
        for key in self.type_vectors:
            if key != 'unknown':
                self.type_vectors[key] /= np.linalg.norm(self.type_vectors[key])
        
        for key in self.wall_direction_vectors:
            self.wall_direction_vectors[key] /= np.linalg.norm(self.wall_direction_vectors[key])
    
    def _get_observation_vector(self, obs: MazeObservation, 
                               location_memory: Optional[LocationMemory] = None) -> np.ndarray:
        """Convert observation to vector, including wall information."""
        obs_type = obs.get_location_type()
        base_vector = self.type_vectors.get(obs_type, self.type_vectors['unknown'])
        
        # Add position encoding
        pos_encoding = np.array([
            np.sin(obs.position[0] / 10),
            np.cos(obs.position[0] / 10),
            np.sin(obs.position[1] / 10),
            np.cos(obs.position[1] / 10)
        ])
        
        if len(pos_encoding) < self.feature_dim:
            pos_encoding = np.pad(pos_encoding, (0, self.feature_dim - len(pos_encoding)))
        else:
            pos_encoding = pos_encoding[:self.feature_dim]
        
        # Base vector
        vector = 0.7 * base_vector + 0.1 * pos_encoding
        
        # Add wall information if available
        if location_memory and location_memory.known_walls:
            wall_vector = np.zeros(self.feature_dim)
            for wall_dir in location_memory.known_walls:
                wall_vector += 0.1 * self.wall_direction_vectors[wall_dir]
            vector += 0.2 * wall_vector
        
        return vector / np.linalg.norm(vector)
    
    def _detect_wall_hit(self, current_pos: Tuple[int, int]) -> bool:
        """Detect if we hit a wall (position didn't change)."""
        if self.last_position is None:
            return False
        return current_pos == self.last_position and self.last_action is not None
    
    def _record_wall(self, from_pos: Tuple[int, int], direction: int):
        """Record a wall location."""
        # Calculate wall position
        if direction == 0:  # Up
            wall_pos = (from_pos[0] - 1, from_pos[1])
        elif direction == 1:  # Right
            wall_pos = (from_pos[0], from_pos[1] + 1)
        elif direction == 2:  # Down
            wall_pos = (from_pos[0] + 1, from_pos[1])
        else:  # Left
            wall_pos = (from_pos[0], from_pos[1] - 1)
        
        wall_mem = WallMemory(from_pos, direction, wall_pos)
        self.wall_memory.add(wall_mem)
        
        # Update location memory with wall info
        if from_pos in self.location_memory:
            self.location_memory[from_pos].known_walls.add(direction)
        
        logger.debug(f"Recorded wall: from {from_pos} direction {direction} (wall at {wall_pos})")
    
    def _is_wall_known(self, from_pos: Tuple[int, int], direction: int) -> bool:
        """Check if a wall is already known."""
        if from_pos in self.location_memory:
            return direction in self.location_memory[from_pos].known_walls
        return False
    
    def _calculate_information_gain(self, obs: MazeObservation, 
                                   action: Optional[int] = None) -> float:
        """Calculate IG, considering wall discoveries."""
        ig = 1.0
        
        # Higher IG for important locations
        if obs.is_goal:
            ig = 10.0
        elif obs.is_junction:
            ig = 2.0
        elif obs.is_dead_end:
            ig = 1.5
        
        # Bonus IG for discovering new walls
        if action is not None and not self._is_wall_known(obs.position, action):
            ig += 0.5  # Bonus for potentially discovering a wall
        
        # Reduce IG for known locations
        if obs.position in self.location_memory:
            ig *= 0.1
            
        return ig
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action using wall-aware geDIG."""
        current_pos = observation.position
        
        # Detect wall hit from last action
        if self._detect_wall_hit(current_pos) and self.last_action is not None:
            self._record_wall(current_pos, self.last_action)
            observation.hit_wall = True  # Mark as wall hit
        
        # Create or update location memory
        if current_pos not in self.location_memory:
            ig = self._calculate_information_gain(observation)
            
            location_mem = LocationMemory(
                position=current_pos,
                node_type=observation.get_location_type(),
                vector=self._get_observation_vector(observation),
                information_gain=ig
            )
            self.location_memory[current_pos] = location_mem
            logger.debug(f"Memorized {current_pos} as {location_mem.node_type}, IG={ig:.2f}")
        else:
            # Update visit count
            self.location_memory[current_pos].visits += 1
        
        # Get current location memory
        current_memory = self.location_memory[current_pos]
        current_vector = self._get_observation_vector(observation, current_memory)
        
        # Donut search for relevant memories
        relevant_memories = self._donut_search(current_pos, current_vector)
        
        # Evaluate each possible action
        action_scores = {}
        
        for action in range(4):  # Check all 4 directions
            # Skip if we know there's a wall
            if self._is_wall_known(current_pos, action):
                continue
                
            # Skip if not in possible moves (physical constraint)
            if action not in observation.possible_moves:
                # This means there's a wall we haven't recorded yet
                self._record_wall(current_pos, action)
                continue
            
            # Calculate next position
            delta = maze.ACTIONS[action]
            next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Calculate GED
            if next_pos in self.location_memory:
                next_memory = self.location_memory[next_pos]
                ged = self._calculate_ged(current_pos, next_pos, 
                                        current_vector, next_memory.vector)
                ig = 0.1  # Low IG for known locations
            else:
                ged = self._calculate_ged(current_pos, next_pos, current_vector, None)
                ig = self._calculate_information_gain(observation, action)
            
            # geDIG objective
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = f
            
            logger.debug(f"  Action {maze.ACTION_NAMES[action]}: GED={ged:.2f}, IG={ig:.2f}, f={f:.2f}")
        
        if not action_scores:
            logger.warning(f"No valid actions at {current_pos}")
            return 0
        
        # Choose best action
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a])
        
        # Store current state for wall detection
        self.last_position = current_pos
        self.last_action = best_action
        
        # Exploration
        if np.random.random() < self.config.exploration_epsilon:
            valid_actions = list(action_scores.keys())
            if valid_actions:
                best_action = np.random.choice(valid_actions)
        
        return best_action
    
    def _donut_search(self, current_pos: Tuple[int, int], 
                     query_vector: np.ndarray) -> List[LocationMemory]:
        """Donut search with wall awareness."""
        relevant_memories = []
        
        for memory in self.location_memory.values():
            spatial_dist = np.sqrt(
                (memory.position[0] - current_pos[0])**2 + 
                (memory.position[1] - current_pos[1])**2
            )
            
            if self.config.donut_inner_radius < spatial_dist <= self.config.donut_outer_radius:
                relevant_memories.append(memory)
        
        return relevant_memories
    
    def _calculate_ged(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                      from_vector: np.ndarray, to_vector: Optional[np.ndarray] = None) -> float:
        """Calculate GED."""
        spatial_dist = np.sqrt(
            (to_pos[0] - from_pos[0])**2 + 
            (to_pos[1] - from_pos[1])**2
        )
        
        if to_vector is not None:
            semantic_dist = np.linalg.norm(from_vector - to_vector)
        else:
            semantic_dist = 0.5
        
        return 0.5 * spatial_dist + 0.5 * semantic_dist
    
    def new_episode(self):
        """New episode - reset position tracking but keep wall memory."""
        self.current_episode += 1
        self.last_position = None
        self.last_action = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics including wall knowledge."""
        total_walls_known = sum(len(mem.known_walls) for mem in self.location_memory.values())
        
        return {
            'total_locations': len(self.location_memory),
            'total_walls_recorded': len(self.wall_memory),
            'total_wall_directions_known': total_walls_known,
            'episodes': self.current_episode,
            'location_positions': list(self.location_memory.keys()),
            'wall_positions': [w.wall_pos for w in self.wall_memory]
        }