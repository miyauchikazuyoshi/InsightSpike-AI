"""Pure geDIG-based navigator focusing on GED and IG balance."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ...environments.maze import MazeObservation
from ..maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """Memory node with position and type."""
    position: Tuple[int, int]
    node_type: str  # 'wall', 'junction', 'corridor', 'goal'
    vector: np.ndarray
    information_gain: float  # IG value for this discovery
    visits: int = 1
    

class PureGeDIGNavigator:
    """Navigator using pure geDIG principles: GED minimization with IG maximization."""
    
    def __init__(self, config: MazeNavigatorConfig):
        """Initialize the navigator.
        
        Args:
            config: Navigation configuration
        """
        self.config = config
        self.memory_nodes: Dict[Tuple[int, int], MemoryNode] = {}
        self.current_episode = 0
        
        # Simple feature embeddings
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0  # Weight for GED term
        self.k_ig = 1.5   # Temperature for IG term (increased for more exploration)
        
    def _init_embedder(self):
        """Initialize simple feature embedder."""
        np.random.seed(42)
        
        # Semantic vectors for different location types
        self.type_vectors = {
            'wall': np.random.randn(self.feature_dim),
            'junction': np.random.randn(self.feature_dim),
            'corridor': np.random.randn(self.feature_dim),
            'dead_end': np.random.randn(self.feature_dim),
            'goal': np.random.randn(self.feature_dim),
            'unknown': np.zeros(self.feature_dim)  # Unknown is origin
        }
        
        # Normalize all vectors
        for key in self.type_vectors:
            if key != 'unknown':
                self.type_vectors[key] /= np.linalg.norm(self.type_vectors[key])
    
    def _get_observation_vector(self, obs: MazeObservation) -> np.ndarray:
        """Convert observation to vector."""
        obs_type = obs.get_location_type()
        base_vector = self.type_vectors.get(obs_type, self.type_vectors['unknown'])
        
        # Add slight position encoding
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
        
        # 90% type, 10% position
        vector = 0.9 * base_vector + 0.1 * pos_encoding
        return vector / np.linalg.norm(vector)
    
    def _calculate_information_gain(self, obs: MazeObservation) -> float:
        """Calculate information gain for discovering this location.
        
        IG is higher for:
        - New discoveries (not in memory)
        - Important locations (junctions, dead ends, goals)
        - Walls (important for navigation)
        """
        # Base IG for discovery
        ig = 1.0
        
        # Higher IG for important location types
        if obs.is_goal:
            ig = 10.0  # Huge reward for finding goal
        elif obs.is_junction:
            ig = 2.0   # Junctions are valuable
        elif obs.is_dead_end:
            ig = 1.5   # Dead ends are worth remembering
        elif obs.hit_wall:
            ig = 1.5   # Walls are important boundaries
        
        # Reduce IG if we've been here before
        if obs.position in self.memory_nodes:
            ig *= 0.1  # Much less value in revisiting
            
        return ig
    
    def _donut_search(self, current_pos: Tuple[int, int], 
                     query_vector: np.ndarray) -> List[MemoryNode]:
        """Donut search: find relevant memories, excluding very close ones.
        
        This implements the "ignore known" principle.
        """
        relevant_memories = []
        
        for node in self.memory_nodes.values():
            # Calculate spatial distance
            spatial_dist = np.sqrt(
                (node.position[0] - current_pos[0])**2 + 
                (node.position[1] - current_pos[1])**2
            )
            
            # Skip if too close (inner radius) - these are "known"
            if spatial_dist <= self.config.donut_inner_radius:
                continue
                
            # Skip if too far (outer radius)
            if spatial_dist > self.config.donut_outer_radius:
                continue
            
            # Calculate semantic distance
            semantic_dist = np.linalg.norm(query_vector - node.vector)
            
            # Include in results (can add semantic filtering if needed)
            relevant_memories.append(node)
        
        return relevant_memories
    
    def _calculate_ged(self, from_pos: Tuple[int, int], 
                      to_pos: Tuple[int, int],
                      from_vector: np.ndarray,
                      to_vector: Optional[np.ndarray] = None) -> float:
        """Calculate Generalized Euclidean Distance.
        
        Combines spatial and semantic distance.
        """
        # Spatial distance component
        spatial_dist = np.sqrt(
            (to_pos[0] - from_pos[0])**2 + 
            (to_pos[1] - from_pos[1])**2
        )
        
        # Semantic distance component
        if to_vector is not None:
            semantic_dist = np.linalg.norm(from_vector - to_vector)
        else:
            # Unknown location - use moderate distance
            semantic_dist = 0.5
        
        # Combine distances (equal weighting for now)
        ged = 0.5 * spatial_dist + 0.5 * semantic_dist
        
        return ged
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide next action using geDIG objective: f = w*GED - k*IG.
        
        Lower f is better (minimize cost, maximize information gain).
        """
        current_pos = observation.position
        current_vector = self._get_observation_vector(observation)
        
        # Check if we should remember this location
        if observation.position not in self.memory_nodes:
            # Calculate IG for this discovery
            ig = self._calculate_information_gain(observation)
            
            # Create memory node
            node = MemoryNode(
                position=observation.position,
                node_type=observation.get_location_type(),
                vector=current_vector,
                information_gain=ig
            )
            self.memory_nodes[observation.position] = node
            logger.debug(f"Memorized {observation.position} as {node.node_type}, IG={ig:.2f}")
        
        # Get relevant memories using donut search
        relevant_memories = self._donut_search(current_pos, current_vector)
        
        # Evaluate each possible action using geDIG objective
        action_scores = {}
        
        for action in observation.possible_moves:
            # Get next position
            delta = maze.ACTIONS[action]
            next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Check if we have memory of next position
            if next_pos in self.memory_nodes:
                next_node = self.memory_nodes[next_pos]
                ged = self._calculate_ged(current_pos, next_pos, 
                                        current_vector, next_node.vector)
                ig = 0.1  # Low IG for known locations
            else:
                # Unknown location
                ged = self._calculate_ged(current_pos, next_pos, 
                                        current_vector, None)
                ig = 1.0  # Base IG for exploration
                
                # Boost IG if nearby memories suggest interesting area
                for memory in relevant_memories:
                    if memory.node_type in ['junction', 'goal']:
                        ig *= 1.2  # Slightly higher IG near interesting areas
            
            # Calculate geDIG objective: f = w*GED - k*IG
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = f
            
            logger.debug(f"  Action {maze.ACTION_NAMES[action]}: GED={ged:.2f}, IG={ig:.2f}, f={f:.2f}")
        
        # Choose action with lowest f (best trade-off)
        if not action_scores:
            # No possible moves - shouldn't happen but handle it
            logger.warning(f"No possible moves at {observation.position}")
            return 0  # Default action
        
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a])
        
        # Add exploration noise
        if np.random.random() < self.config.exploration_epsilon:
            return np.random.choice(observation.possible_moves)
        
        return best_action
    
    def new_episode(self):
        """Called at the start of a new episode."""
        self.current_episode += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        node_types_count = {}
        total_ig = 0
        
        for node in self.memory_nodes.values():
            node_types_count[node.node_type] = node_types_count.get(node.node_type, 0) + 1
            total_ig += node.information_gain
        
        return {
            'total_nodes': len(self.memory_nodes),
            'node_types_count': node_types_count,
            'total_information_gain': total_ig,
            'episodes': self.current_episode,
            'node_positions': list(self.memory_nodes.keys()),
            'node_types': [n.node_type for n in self.memory_nodes.values()]
        }