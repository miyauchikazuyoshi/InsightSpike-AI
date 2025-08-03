"""Experience-based navigator that memorizes both visual and physical experiences."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from enum import Enum

from ..environments.maze import MazeObservation
from ..config.maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


class ExperienceType(Enum):
    """Types of experiences at a position."""
    VISUAL_WALL = "visual_wall"  # Saw a wall
    VISUAL_PATH = "visual_path"  # Saw a path
    PHYSICAL_BLOCKED = "physical_blocked"  # Tried to move but blocked
    PHYSICAL_PASSED = "physical_passed"  # Successfully moved through
    UNKNOWN = "unknown"  # Not yet experienced


@dataclass
class DirectionalExperience:
    """Experience in a specific direction from a position."""
    direction: int  # 0-3 (up, right, down, left)
    visual: ExperienceType = ExperienceType.UNKNOWN
    physical: ExperienceType = ExperienceType.UNKNOWN
    attempts: int = 0  # Number of times tried
    last_update: int = 0  # When last updated
    
    @property
    def is_passable(self) -> Optional[bool]:
        """Determine if passable based on experience."""
        if self.physical == ExperienceType.PHYSICAL_PASSED:
            return True
        elif self.physical == ExperienceType.PHYSICAL_BLOCKED:
            return False
        elif self.visual == ExperienceType.VISUAL_WALL:
            return False
        elif self.visual == ExperienceType.VISUAL_PATH:
            return True  # Might be passable
        return None  # Unknown
    
    @property
    def confidence(self) -> float:
        """Confidence in our knowledge about this direction."""
        if self.physical != ExperienceType.UNKNOWN:
            return 1.0  # Physical experience is most reliable
        elif self.visual != ExperienceType.UNKNOWN:
            return 0.7  # Visual experience is less certain
        return 0.0


@dataclass
class MemoryNode:
    """Memory node with visual and physical experiences."""
    position: Tuple[int, int]
    experiences: Dict[int, DirectionalExperience] = field(default_factory=dict)
    visits: int = 0
    vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # Initialize experiences for all 4 directions
        for d in range(4):
            if d not in self.experiences:
                self.experiences[d] = DirectionalExperience(direction=d)
    
    def update_visual(self, direction: int, is_wall: bool, timestep: int):
        """Update visual experience."""
        exp = self.experiences[direction]
        exp.visual = ExperienceType.VISUAL_WALL if is_wall else ExperienceType.VISUAL_PATH
        exp.last_update = timestep
    
    def update_physical(self, direction: int, blocked: bool, timestep: int):
        """Update physical experience."""
        exp = self.experiences[direction]
        exp.physical = ExperienceType.PHYSICAL_BLOCKED if blocked else ExperienceType.PHYSICAL_PASSED
        exp.attempts += 1
        exp.last_update = timestep
    
    def get_information_gain(self, direction: int) -> float:
        """Calculate IG for exploring in a direction."""
        exp = self.experiences[direction]
        
        # High IG for unknown experiences
        if exp.physical == ExperienceType.UNKNOWN:
            ig = 1.0
            # Extra bonus if visual suggests it's passable
            if exp.visual == ExperienceType.VISUAL_PATH:
                ig += 0.5
        else:
            # Low IG for already experienced
            ig = 0.1 / (exp.attempts + 1)
        
        # Uncertainty bonus for visual-physical mismatch
        if (exp.visual == ExperienceType.VISUAL_PATH and 
            exp.physical == ExperienceType.PHYSICAL_BLOCKED):
            ig += 0.3  # Interesting contradiction
        
        return ig


class ExperienceMemoryNavigator:
    """Navigator using both visual and physical memory."""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.memory_nodes: Dict[Tuple[int, int], MemoryNode] = {}
        self.current_position = None
        self.last_position = None
        self.last_action = None
        self.timestep = 0
        self.goal_position = None
        
        # Feature embedding
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0
        self.k_ig = 2.0
        
        # Track geDIG values
        self.gediq_history = []  # List of (step, action, ged, ig, f) tuples
        
    def _init_embedder(self):
        """Initialize feature embedder."""
        np.random.seed(42)
        
        # Base vectors for different states
        self.state_vectors = {
            'visited': np.random.randn(self.feature_dim),
            'wall_visual': np.random.randn(self.feature_dim),
            'wall_physical': np.random.randn(self.feature_dim),
            'path_visual': np.random.randn(self.feature_dim),
            'path_physical': np.random.randn(self.feature_dim),
            'unknown': np.zeros(self.feature_dim)
        }
        
        # Normalize
        for key in self.state_vectors:
            if key != 'unknown':
                self.state_vectors[key] /= np.linalg.norm(self.state_vectors[key])
    
    def _get_node_vector(self, node: MemoryNode) -> np.ndarray:
        """Create feature vector for a memory node."""
        vector = np.zeros(self.feature_dim)
        
        # Base vector for visited location
        vector += 0.3 * self.state_vectors['visited']
        
        # Add experience vectors
        for direction, exp in node.experiences.items():
            weight = 0.1
            
            if exp.visual == ExperienceType.VISUAL_WALL:
                vector += weight * self.state_vectors['wall_visual']
            elif exp.visual == ExperienceType.VISUAL_PATH:
                vector += weight * self.state_vectors['path_visual']
                
            if exp.physical == ExperienceType.PHYSICAL_BLOCKED:
                vector += weight * self.state_vectors['wall_physical']
            elif exp.physical == ExperienceType.PHYSICAL_PASSED:
                vector += weight * self.state_vectors['path_physical']
        
        # Position encoding
        pos_encoding = np.array([
            np.sin(node.position[0] / 10),
            np.cos(node.position[0] / 10),
            np.sin(node.position[1] / 10),
            np.cos(node.position[1] / 10)
        ])
        
        if len(pos_encoding) < self.feature_dim:
            pos_encoding = np.pad(pos_encoding, (0, self.feature_dim - len(pos_encoding)))
        else:
            pos_encoding = pos_encoding[:self.feature_dim]
        
        vector += 0.2 * pos_encoding
        
        return vector / (np.linalg.norm(vector) + 1e-8)
    
    def _update_physical_experience(self, from_pos: Tuple[int, int], action: int, to_pos: Tuple[int, int]):
        """Update physical experience based on movement result."""
        if from_pos in self.memory_nodes:
            node = self.memory_nodes[from_pos]
            # Check if we actually moved
            blocked = (from_pos == to_pos)
            node.update_physical(action, blocked, self.timestep)
            
            if blocked:
                print(f"  Physical experience: Hit wall when trying {['up', 'right', 'down', 'left'][action]}")
    
    def _update_visual_experience(self, position: Tuple[int, int], observation: MazeObservation):
        """Update visual experience from current position."""
        if position not in self.memory_nodes:
            self.memory_nodes[position] = MemoryNode(position=position)
        
        node = self.memory_nodes[position]
        node.visits += 1
        
        # Update visual experience for each direction
        for direction in range(4):
            is_wall = direction not in observation.possible_moves
            node.update_visual(direction, is_wall, self.timestep)
        
        # Update vector
        node.vector = self._get_node_vector(node)
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action based on visual and physical experiences."""
        current_pos = observation.position
        self.timestep += 1
        
        # Update physical experience from last move
        if self.last_position is not None and self.last_action is not None:
            self._update_physical_experience(self.last_position, self.last_action, current_pos)
        
        # Update visual experience at current position
        self._update_visual_experience(current_pos, observation)
        
        # Check for goal
        if observation.is_goal and self.goal_position is None:
            self.goal_position = current_pos
            print(f"🎯 Goal discovered at {current_pos}!")
        
        # Get current node
        current_node = self.memory_nodes[current_pos]
        
        # Evaluate each possible action
        action_scores = {}
        
        for action in range(4):  # Check all 4 directions
            exp = current_node.experiences[action]
            
            # Skip if we're confident it's blocked
            if exp.is_passable is False and exp.confidence > 0.9:
                continue
            
            # Skip if not in possible moves (visual check)
            if action not in observation.possible_moves:
                continue
            
            delta = maze.ACTIONS[action]
            next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # GED (movement cost) - reduced for familiar paths
            if next_pos in self.memory_nodes:
                next_node = self.memory_nodes[next_pos]
                # Reduce cost based on familiarity (visits)
                ged = 1.0 / (1.0 + 0.2 * next_node.visits)
                # Minimum cost of 0.3 for very familiar paths
                ged = max(0.3, ged)
            else:
                ged = 1.0  # Full cost for unknown positions
            
            # IG based on experiences
            ig = current_node.get_information_gain(action)
            
            # Bonus for unvisited positions
            if next_pos not in self.memory_nodes:
                ig += 1.5
            else:
                # Reduce IG based on visits
                next_node = self.memory_nodes[next_pos]
                ig += 0.5 / (next_node.visits + 1)
            
            # Goal bonus
            if self.goal_position and next_pos == self.goal_position:
                ig += 10.0
            
            # geDIG objective
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = f
            
            logger.debug(f"  Action {['up', 'right', 'down', 'left'][action]}: "
                        f"exp={exp.is_passable}, conf={exp.confidence:.2f}, "
                        f"IG={ig:.2f}, f={f:.2f}")
        
        if not action_scores:
            # Forced to try a blocked direction
            for action in observation.possible_moves:
                action_scores[action] = 0
        
        if not action_scores:
            return 0
        
        # Choose best action
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a])
        
        # Store geDIG values for best action
        if best_action in action_scores:
            # Recalculate components for the chosen action
            delta = maze.ACTIONS[best_action]
            next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Recalculate GED with familiarity
            if next_pos in self.memory_nodes:
                next_node = self.memory_nodes[next_pos]
                ged = 1.0 / (1.0 + 0.2 * next_node.visits)
                ged = max(0.3, ged)
            else:
                ged = 1.0
            
            # Calculate IG for chosen action
            exp = current_node.experiences[best_action]
            ig = current_node.get_information_gain(best_action)
            if next_pos not in self.memory_nodes:
                ig += 1.5
            else:
                next_node = self.memory_nodes[next_pos]
                ig += 0.5 / (next_node.visits + 1)
            if self.goal_position and next_pos == self.goal_position:
                ig += 10.0
                
            f = self.w_ged * ged - self.k_ig * ig
            
            self.gediq_history.append({
                'step': self.timestep,
                'action': best_action,
                'ged': ged,
                'ig': ig,
                'f': f,
                'position': current_pos
            })
        
        # Store for next update
        self.last_position = current_pos
        self.last_action = best_action
        
        # Exploration
        if np.random.random() < self.config.exploration_epsilon:
            return np.random.choice(list(action_scores.keys()))
        
        return best_action
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        total_visual_walls = 0
        total_physical_blocks = 0
        mismatches = 0
        
        for node in self.memory_nodes.values():
            for exp in node.experiences.values():
                if exp.visual == ExperienceType.VISUAL_WALL:
                    total_visual_walls += 1
                if exp.physical == ExperienceType.PHYSICAL_BLOCKED:
                    total_physical_blocks += 1
                if (exp.visual == ExperienceType.VISUAL_PATH and 
                    exp.physical == ExperienceType.PHYSICAL_BLOCKED):
                    mismatches += 1
        
        return {
            'total_positions': len(self.memory_nodes),
            'total_visual_walls': total_visual_walls,
            'total_physical_blocks': total_physical_blocks,
            'visual_physical_mismatches': mismatches,
            'timesteps': self.timestep
        }