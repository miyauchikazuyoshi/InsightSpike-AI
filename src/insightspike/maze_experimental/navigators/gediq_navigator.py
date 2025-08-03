"""geDIG-based navigator for maze exploration."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ...environments.maze import MazeObservation
from ..query.sphere_search import SphereSearch
from ..algorithms.gedig_core import GEDIGCalculator
from ..maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class NavigationMemoryNode:
    """Memory node for navigation."""
    position: Tuple[int, int]
    features: Dict[str, Any]
    vector: np.ndarray
    creation_energy: float
    visits: int = 1
    last_visited: int = 0
    
    def should_merge(self, other: 'NavigationMemoryNode', threshold: float = 0.9) -> bool:
        """Check if this node should be merged with another."""
        if np.linalg.norm(self.vector - other.vector) < (1 - threshold):
            return True
        return False


class GeDIGNavigator:
    """Navigator using geDIG principles for maze exploration."""
    
    def __init__(self, config: MazeNavigatorConfig):
        """Initialize the navigator.
        
        Args:
            config: Navigation configuration
        """
        self.config = config
        self.memory_nodes: Dict[Tuple[int, int], NavigationMemoryNode] = {}
        self.sphere_search = SphereSearch()
        self.gediq_calc = GEDIGCalculator()
        
        # Episode tracking
        self.current_episode = 0
        self.total_energy_spent = 0.0
        
        # Simple feature embedder (can be replaced with sentence transformer)
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
    def _init_embedder(self):
        """Initialize feature embedder."""
        if self.config.use_pretrained_embedder:
            # TODO: Load sentence transformer
            logger.info("Using pretrained embedder not implemented, falling back to simple embedder")
        
        # Simple embedder: map feature types to random vectors
        self.feature_vectors = {
            'wall': np.random.randn(self.feature_dim),
            'junction': np.random.randn(self.feature_dim),
            'dead_end': np.random.randn(self.feature_dim),
            'corridor': np.random.randn(self.feature_dim),
            'goal': np.random.randn(self.feature_dim),
            'unknown': np.random.randn(self.feature_dim)
        }
        
        # Normalize
        for key in self.feature_vectors:
            self.feature_vectors[key] /= np.linalg.norm(self.feature_vectors[key])
    
    def embed_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to vector embedding."""
        feature_type = features.get('type', 'unknown')
        base_vector = self.feature_vectors.get(feature_type, self.feature_vectors['unknown'])
        
        # Add some variation based on position
        position = features.get('position', (0, 0))
        position_encoding = np.array([
            np.sin(position[0] / 10),
            np.cos(position[0] / 10),
            np.sin(position[1] / 10),
            np.cos(position[1] / 10)
        ])
        
        # Combine base vector with position encoding
        if len(position_encoding) < self.feature_dim:
            position_encoding = np.pad(position_encoding, (0, self.feature_dim - len(position_encoding)))
        else:
            position_encoding = position_encoding[:self.feature_dim]
        
        vector = 0.8 * base_vector + 0.2 * position_encoding
        return vector / np.linalg.norm(vector)
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide next action based on geDIG principles.
        
        Args:
            observation: Current observation from maze
            maze: Maze environment (for looking ahead)
            
        Returns:
            Action to take (0-3)
        """
        current_pos = observation.position
        
        # Check if we should create a memory node
        if self._should_create_node(observation):
            self._create_memory_node(observation)
        
        # Sphere search for nearby memories
        query_vector = self.embed_features(observation.to_features())
        nearby_memories = self._search_nearby_memories(current_pos, query_vector)
        
        # Evaluate each possible action
        action_energies = {}
        for action in observation.possible_moves:
            energy = self._evaluate_action_energy(
                current_pos, action, nearby_memories, maze
            )
            action_energies[action] = energy
        
        # Epsilon-greedy exploration
        if np.random.random() < self.config.exploration_epsilon:
            return np.random.choice(observation.possible_moves)
        else:
            # Choose action with lowest energy
            return min(action_energies.keys(), key=lambda a: action_energies[a])
    
    def _should_create_node(self, observation: MazeObservation) -> bool:
        """Determine if current location should be memorized."""
        # Always create node for important locations
        if observation.is_goal or observation.is_junction or observation.is_dead_end:
            return True
        
        # Create node if hit wall (to remember obstacles)
        if observation.hit_wall:
            return True
        
        # Check if we already have a node here
        if observation.position in self.memory_nodes:
            # Update visit count instead
            self.memory_nodes[observation.position].visits += 1
            self.memory_nodes[observation.position].last_visited = self.current_episode
            return False
        
        # Don't create nodes for simple corridors (unless first time)
        if observation.num_paths == 2 and len(self.memory_nodes) > 10:
            return False
        
        return True
    
    def _create_memory_node(self, observation: MazeObservation):
        """Create a new memory node."""
        features = observation.to_features()
        vector = self.embed_features(features)
        
        node = NavigationMemoryNode(
            position=observation.position,
            features=features,
            vector=vector,
            creation_energy=self.config.node_creation_cost,
            last_visited=self.current_episode
        )
        
        self.memory_nodes[observation.position] = node
        self.total_energy_spent += self.config.node_creation_cost
        
        logger.debug(f"Created memory node at {observation.position}, type: {features['type']}")
    
    def _search_nearby_memories(self, position: Tuple[int, int], 
                               query_vector: np.ndarray) -> List[NavigationMemoryNode]:
        """Search for nearby memory nodes."""
        nearby = []
        
        for node in self.memory_nodes.values():
            # Spatial distance
            spatial_dist = np.sqrt(
                (node.position[0] - position[0])**2 + 
                (node.position[1] - position[1])**2
            )
            
            # Vector distance
            vector_dist = np.linalg.norm(query_vector - node.vector)
            
            # Combined distance (weighted)
            combined_dist = 0.7 * spatial_dist + 0.3 * vector_dist * 10
            
            if combined_dist <= self.config.search_radius:
                nearby.append(node)
        
        return nearby
    
    def _evaluate_action_energy(self, current_pos: Tuple[int, int], 
                               action: int,
                               nearby_memories: List[NavigationMemoryNode],
                               maze) -> float:
        """Evaluate energy cost of taking an action."""
        # Get next position
        delta = maze.ACTIONS[action]
        next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
        
        energy = 0.0
        
        # Check for walls ahead (donut search)
        walls_ahead = self._count_walls_ahead(next_pos, nearby_memories)
        energy += walls_ahead * self.config.wall_penalty
        
        # Node creation cost (if new area)
        if next_pos not in self.memory_nodes:
            energy += self.config.node_creation_cost
        else:
            # Visiting known area is cheaper
            energy += 0.1
        
        # Bonus for unexplored areas (if not too many walls)
        if next_pos not in self.memory_nodes and walls_ahead < 2:
            energy -= self.config.unknown_bonus
        
        # Consider dead ends (high cost)
        dead_end_memories = [m for m in nearby_memories if m.features['type'] == 'dead_end']
        for dead_end in dead_end_memories:
            dist = np.sqrt((dead_end.position[0] - next_pos[0])**2 + 
                          (dead_end.position[1] - next_pos[1])**2)
            if dist < 3:
                energy += 2.0 / (dist + 1)
        
        return energy
    
    def _count_walls_ahead(self, position: Tuple[int, int], 
                          nearby_memories: List[NavigationMemoryNode]) -> int:
        """Count walls in the direction of movement using donut search."""
        wall_count = 0
        
        for memory in nearby_memories:
            if memory.features['type'] == 'wall':
                # Distance from position
                dist = np.sqrt(
                    (memory.position[0] - position[0])**2 + 
                    (memory.position[1] - position[1])**2
                )
                
                # Donut search: only count if within outer radius but not too close
                if self.config.donut_inner_radius < dist <= self.config.donut_outer_radius:
                    wall_count += 1
        
        return wall_count
    
    def sleep_phase(self):
        """Optimize memory during sleep phase."""
        logger.info(f"Entering sleep phase after episode {self.current_episode}")
        
        # 1. Merge similar nodes
        self._merge_similar_nodes()
        
        # 2. Forget rarely visited nodes
        self._forget_unused_nodes()
        
        # 3. Discover shortcuts
        self._discover_shortcuts()
        
        # 4. Extract patterns
        patterns = self._extract_patterns()
        logger.info(f"Extracted patterns: {patterns}")
        
    def _merge_similar_nodes(self):
        """Merge nodes that are very similar."""
        merged = set()
        nodes = list(self.memory_nodes.values())
        
        for i, node1 in enumerate(nodes):
            if id(node1) in merged:
                continue
                
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if id(node2) in merged:
                    continue
                    
                if node1.should_merge(node2):
                    # Keep the more visited node
                    if node1.visits >= node2.visits:
                        del self.memory_nodes[node2.position]
                        merged.add(id(node2))
                    else:
                        del self.memory_nodes[node1.position]
                        merged.add(id(node1))
                        break
    
    def _forget_unused_nodes(self):
        """Remove nodes that haven't been visited recently."""
        threshold_episode = self.current_episode - 50
        to_remove = []
        
        for pos, node in self.memory_nodes.items():
            if node.last_visited < threshold_episode and node.visits < 3:
                to_remove.append(pos)
        
        for pos in to_remove:
            del self.memory_nodes[pos]
            
        if to_remove:
            logger.info(f"Forgot {len(to_remove)} unused nodes")
    
    def _discover_shortcuts(self):
        """Find potential shortcuts between nodes."""
        # TODO: Implement shortcut discovery
        pass
    
    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract navigation patterns from memory."""
        patterns = {
            'junction_count': sum(1 for n in self.memory_nodes.values() 
                                 if n.features['type'] == 'junction'),
            'dead_end_count': sum(1 for n in self.memory_nodes.values() 
                                 if n.features['type'] == 'dead_end'),
            'wall_count': sum(1 for n in self.memory_nodes.values() 
                             if n.features['type'] == 'wall'),
            'total_nodes': len(self.memory_nodes),
            'avg_visits': np.mean([n.visits for n in self.memory_nodes.values()])
                         if self.memory_nodes else 0
        }
        return patterns
    
    def new_episode(self):
        """Called at the start of a new episode."""
        self.current_episode += 1
        
        # Sleep phase at intervals
        if self.current_episode % self.config.sleep_interval == 0:
            self.sleep_phase()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        return {
            'total_nodes': len(self.memory_nodes),
            'total_energy': self.total_energy_spent,
            'episodes': self.current_episode,
            'node_positions': list(self.memory_nodes.keys()),
            'node_types': [n.features['type'] for n in self.memory_nodes.values()]
        }