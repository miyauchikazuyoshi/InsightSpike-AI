"""Blind experience navigator - no visual cheating, pure physical exploration."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum

from ..environments.maze import MazeObservation
from ..config.maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


class ExperienceType(Enum):
    """Types of experiences at a position."""
    PHYSICAL_BLOCKED = "physical_blocked"  # Tried to move but blocked
    PHYSICAL_PASSED = "physical_passed"  # Successfully moved through
    UNKNOWN = "unknown"  # Not yet experienced


@dataclass
class BlindDirectionalExperience:
    """Physical-only experience in a specific direction."""
    direction: int  # 0-3 (up, right, down, left)
    physical: ExperienceType = ExperienceType.UNKNOWN
    attempts: int = 0  # Number of times tried
    last_update: int = 0  # When last updated
    
    @property
    def is_passable(self) -> Optional[bool]:
        """Determine if passable based on physical experience only."""
        if self.physical == ExperienceType.PHYSICAL_PASSED:
            return True
        elif self.physical == ExperienceType.PHYSICAL_BLOCKED:
            return False
        return None  # Unknown - must try!
    
    @property
    def confidence(self) -> float:
        """Confidence in our knowledge about this direction."""
        if self.physical != ExperienceType.UNKNOWN:
            return 1.0  # Physical experience is certain
        return 0.0  # No knowledge yet


@dataclass
class BlindMemoryNode:
    """Memory node with physical experiences only - no visual cheating."""
    position: Tuple[int, int]
    experiences: Dict[int, BlindDirectionalExperience] = field(default_factory=dict)
    visits: int = 0
    vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # Initialize experiences for all 4 directions
        for d in range(4):
            if d not in self.experiences:
                self.experiences[d] = BlindDirectionalExperience(direction=d)
    
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
            # Must try to know!
            ig = 2.0
        else:
            # Low IG for already experienced
            ig = 0.1 / (exp.attempts + 1)
        
        return ig


class BlindExperienceNavigator:
    """Navigator using only physical memory - no visual cheating."""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.memory_nodes: Dict[Tuple[int, int], BlindMemoryNode] = {}
        self.current_position = None
        self.last_position = None
        self.last_action = None
        self.timestep = 0
        self.goal_position = None
        self.wall_hits = 0  # Track how many times we hit walls
        
        # Feature embedding
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0
        self.k_ig = 2.0
        
        # Track geDIG values
        self.gediq_history = []
        
    def _init_embedder(self):
        """Initialize feature embedder."""
        np.random.seed(42)
        
        # Base vectors for different states
        self.state_vectors = {
            'visited': np.random.randn(self.feature_dim),
            'blocked': np.random.randn(self.feature_dim),
            'passed': np.random.randn(self.feature_dim),
            'unknown': np.zeros(self.feature_dim)
        }
        
        # Normalize
        for key in self.state_vectors:
            if key != 'unknown':
                self.state_vectors[key] /= np.linalg.norm(self.state_vectors[key])
    
    def _get_node_vector(self, node: BlindMemoryNode) -> np.ndarray:
        """Create feature vector for a memory node."""
        vector = np.zeros(self.feature_dim)
        
        # Base vector for visited location
        vector += 0.3 * self.state_vectors['visited']
        
        # Add experience vectors
        for direction, exp in node.experiences.items():
            weight = 0.2
            
            if exp.physical == ExperienceType.PHYSICAL_BLOCKED:
                vector += weight * self.state_vectors['blocked']
            elif exp.physical == ExperienceType.PHYSICAL_PASSED:
                vector += weight * self.state_vectors['passed']
        
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
                self.wall_hits += 1
                logger.debug(f"  💥 Hit wall when trying {['up', 'right', 'down', 'left'][action]} (total hits: {self.wall_hits})")
            else:
                logger.debug(f"  ✓ Successfully moved {['up', 'right', 'down', 'left'][action]}")
    
    def _ensure_node_exists(self, position: Tuple[int, int]):
        """Ensure a memory node exists for the position."""
        if position not in self.memory_nodes:
            self.memory_nodes[position] = BlindMemoryNode(position=position)
        
        node = self.memory_nodes[position]
        node.visits += 1
        node.vector = self._get_node_vector(node)
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action based purely on physical experiences - no visual cheating!"""
        current_pos = observation.position
        self.timestep += 1
        
        # Update physical experience from last move
        if self.last_position is not None and self.last_action is not None:
            self._update_physical_experience(self.last_position, self.last_action, current_pos)
        
        # Ensure current position has a node
        self._ensure_node_exists(current_pos)
        
        # Check for goal
        if observation.is_goal and self.goal_position is None:
            self.goal_position = current_pos
            logger.info(f"🎯 Goal discovered at {current_pos}!")
        
        # Get current node
        current_node = self.memory_nodes[current_pos]
        
        # Evaluate each possible action
        action_scores = {}
        
        logger.debug(f"\n📍 Position {current_pos} (visit #{current_node.visits})")
        
        for action in range(4):  # Check all 4 directions
            exp = current_node.experiences[action]
            
            # Skip if we're certain it's blocked
            if exp.is_passable is False:
                logger.debug(f"  ❌ {['up', 'right', 'down', 'left'][action]}: Known blocked")
                continue
            
            # Calculate where this action leads
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
            
            status = "unknown" if exp.physical == ExperienceType.UNKNOWN else "passable" if exp.is_passable else "blocked"
            logger.debug(f"  {['up', 'right', 'down', 'left'][action]}: status={status}, GED={ged:.2f}, IG={ig:.2f}, f={f:.2f}")
        
        if not action_scores:
            # All known directions are blocked - must retry a blocked direction
            logger.debug("  ⚠️  All directions blocked! Retrying...")
            for action in range(4):
                exp = current_node.experiences[action]
                if exp.physical == ExperienceType.PHYSICAL_BLOCKED:
                    action_scores[action] = 0  # Try again with least recently tried
        
        if not action_scores:
            return 0  # Fallback
        
        # Choose best action (lowest geDIG score)
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a])
        
        logger.debug(f"  → Choosing: {['up', 'right', 'down', 'left'][best_action]}")
        
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
        
        # NO exploration randomness - we want to see true performance
        return best_action
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        total_blocks = 0
        total_passes = 0
        unknown_count = 0
        
        for node in self.memory_nodes.values():
            for exp in node.experiences.values():
                if exp.physical == ExperienceType.PHYSICAL_BLOCKED:
                    total_blocks += 1
                elif exp.physical == ExperienceType.PHYSICAL_PASSED:
                    total_passes += 1
                else:
                    unknown_count += 1
        
        return {
            'total_positions': len(self.memory_nodes),
            'total_wall_hits': self.wall_hits,
            'physical_blocks_learned': total_blocks,
            'physical_passes_learned': total_passes,
            'unknown_directions': unknown_count,
            'timesteps': self.timestep
        }