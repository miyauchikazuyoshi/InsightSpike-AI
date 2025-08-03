"""Structured action memory navigator with advanced similarity metrics."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from enum import Enum
from collections import defaultdict

from ..environments.maze import MazeObservation
from ..config.maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


class ActionResult(Enum):
    """Result of an action attempt."""
    BLOCKED = "blocked"
    PASSED = "passed"
    UNKNOWN = "unknown"


@dataclass
class ActionNode:
    """A single action attempt from position A to position B."""
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    result: ActionResult
    attempts: int = 0
    vector: Optional[np.ndarray] = None
    
    @property
    def action_key(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Unique key for this action."""
        return (self.from_pos, self.to_pos)
    
    @property
    def direction(self) -> Tuple[int, int]:
        """Get movement direction."""
        dx = self.to_pos[1] - self.from_pos[1]
        dy = self.to_pos[0] - self.from_pos[0]
        return (dx, dy)
    
    @property
    def entropy(self) -> float:
        """Entropy of this action node."""
        if self.result == ActionResult.UNKNOWN:
            return 1.0
        elif self.result == ActionResult.BLOCKED:
            return 0.5
        else:  # PASSED
            return 0.1


class StructuredActionNavigator:
    """Navigator with structured action memory and similarity metrics."""
    
    # Direction mappings
    DIRECTIONS = {
        0: (0, -1),  # Up
        1: (1, 0),   # Right
        2: (0, 1),   # Down
        3: (-1, 0)   # Left
    }
    
    DIRECTION_NAMES = {
        (0, -1): "up",
        (1, 0): "right",
        (0, 1): "down",
        (-1, 0): "left"
    }
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.action_nodes: Dict[Tuple[Tuple[int, int], Tuple[int, int]], ActionNode] = {}
        self.current_position = None
        self.timestep = 0
        self.goal_position = None
        
        # Feature embedding
        self.feature_dim = config.feature_dim
        self._init_embedder()
        
        # geDIG coefficients
        self.w_ged = 1.0
        self.k_ig = 2.0
        
        # Track metrics
        self.wall_hits = 0
        self.gediq_history = []
        
        # Global knowledge patterns
        self.direction_stats = defaultdict(lambda: {"passed": 0, "blocked": 0})
        
        # Initialize without starting knowledge (no cheat!)
        logger.info("Initialized Structured Action Navigator (no prior knowledge)")
        
    def _init_embedder(self):
        """Initialize feature embedder."""
        np.random.seed(42)
        
        # Base vectors
        self.result_vectors = {
            ActionResult.PASSED: np.random.randn(self.feature_dim),
            ActionResult.BLOCKED: np.random.randn(self.feature_dim),
            ActionResult.UNKNOWN: np.zeros(self.feature_dim)
        }
        
        for key in [ActionResult.PASSED, ActionResult.BLOCKED]:
            self.result_vectors[key] /= np.linalg.norm(self.result_vectors[key])
        
        # Direction vectors
        self.direction_vectors = {}
        for direction in self.DIRECTIONS.values():
            self.direction_vectors[direction] = np.random.randn(self.feature_dim)
            self.direction_vectors[direction] /= np.linalg.norm(self.direction_vectors[direction])
    
    def _get_action_vector(self, node: ActionNode) -> np.ndarray:
        """Create feature vector for an action node."""
        vector = np.zeros(self.feature_dim)
        
        # Result component
        vector += 0.6 * self.result_vectors[node.result]
        
        # Direction component
        if node.direction in self.direction_vectors:
            vector += 0.2 * self.direction_vectors[node.direction]
        
        # Position encoding
        pos_encoding = np.array([
            np.sin(node.from_pos[0] / 10),
            np.cos(node.from_pos[0] / 10),
            np.sin(node.from_pos[1] / 10),
            np.cos(node.from_pos[1] / 10)
        ])
        
        if len(pos_encoding) < self.feature_dim:
            pos_encoding = np.pad(pos_encoding, (0, self.feature_dim - len(pos_encoding)))
        else:
            pos_encoding = pos_encoding[:self.feature_dim]
        
        vector += 0.2 * pos_encoding
        
        return vector / (np.linalg.norm(vector) + 1e-8)
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _structural_similarity(self, action1: ActionNode, action2: ActionNode) -> float:
        """Calculate structural similarity between two actions."""
        # Same action = perfect similarity
        if action1.action_key == action2.action_key:
            return 1.0
        
        similarity = 0.0
        
        # 1. Direction similarity (most important)
        if action1.direction == action2.direction:
            similarity += 0.5  # Same direction = high similarity
            
            # Bonus if same result
            if action1.result == action2.result and action1.result != ActionResult.UNKNOWN:
                similarity += 0.2
        
        # 2. Spatial proximity
        distance = self._manhattan_distance(action1.from_pos, action2.from_pos)
        if distance < 5:  # Nearby actions
            proximity_score = 0.3 * (1.0 - distance / 5.0)
            similarity += proximity_score
        
        # 3. Result transferability
        if action1.result != ActionResult.UNKNOWN and action2.result == ActionResult.UNKNOWN:
            # Can we transfer knowledge?
            if action1.direction == action2.direction:
                # Same direction, high transfer potential
                similarity += 0.1
        
        return min(similarity, 1.0)
    
    def _find_similar_actions(self, action_node: ActionNode, 
                             min_similarity: float = 0.3) -> List[Tuple[ActionNode, float]]:
        """Find similar actions using structural similarity."""
        similar = []
        
        for other_node in self.action_nodes.values():
            if other_node.action_key == action_node.action_key:
                continue
            
            similarity = self._structural_similarity(action_node, other_node)
            
            if similarity >= min_similarity:
                similar.append((other_node, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def _update_global_knowledge(self, node: ActionNode):
        """Update global knowledge patterns."""
        if node.result != ActionResult.UNKNOWN:
            direction_name = self.DIRECTION_NAMES.get(node.direction, "unknown")
            if node.result == ActionResult.PASSED:
                self.direction_stats[direction_name]["passed"] += 1
            else:
                self.direction_stats[direction_name]["blocked"] += 1
    
    def _get_direction_confidence(self, direction: Tuple[int, int]) -> float:
        """Get confidence for a direction based on global stats."""
        direction_name = self.DIRECTION_NAMES.get(direction, "unknown")
        stats = self.direction_stats[direction_name]
        
        total = stats["passed"] + stats["blocked"]
        if total == 0:
            return 0.5  # Neutral
        
        return stats["passed"] / total
    
    def _get_or_create_action_node(self, from_pos: Tuple[int, int], 
                                   to_pos: Tuple[int, int]) -> ActionNode:
        """Get existing action node or create new one."""
        key = (from_pos, to_pos)
        
        if key not in self.action_nodes:
            node = ActionNode(
                from_pos=from_pos,
                to_pos=to_pos,
                result=ActionResult.UNKNOWN
            )
            node.vector = self._get_action_vector(node)
            self.action_nodes[key] = node
            
        return self.action_nodes[key]
    
    def _update_action_result(self, from_pos: Tuple[int, int], 
                             to_pos: Tuple[int, int], 
                             moved: bool):
        """Update action node with movement result."""
        node = self._get_or_create_action_node(from_pos, to_pos)
        
        # Update result
        old_result = node.result
        node.result = ActionResult.PASSED if moved else ActionResult.BLOCKED
        node.attempts += 1
        
        # Update vector if result changed
        if old_result != node.result:
            node.vector = self._get_action_vector(node)
            
        # Update global knowledge
        self._update_global_knowledge(node)
        
        if not moved:
            self.wall_hits += 1
            logger.info(f"Action {from_pos} → {to_pos}: BLOCKED")
        else:
            logger.info(f"Action {from_pos} → {to_pos}: PASSED")
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action using structured memory."""
        current_pos = observation.position
        self.timestep += 1
        
        # Update result from last action
        if hasattr(self, 'last_action') and hasattr(self, 'last_target'):
            if self.current_position and current_pos != self.current_position:
                # We moved successfully
                self._update_action_result(self.current_position, current_pos, moved=True)
            elif self.current_position and current_pos == self.current_position:
                # We didn't move (hit wall)
                self._update_action_result(self.current_position, self.last_target, moved=False)
        
        self.current_position = current_pos
        
        # Check for goal
        if observation.is_goal and self.goal_position is None:
            self.goal_position = current_pos
            logger.info(f"🎯 Goal discovered at {current_pos}!")
        
        # Evaluate each possible action
        action_scores = {}
        
        logger.debug(f"\n📍 Position {current_pos}")
        
        for action in range(4):
            # Calculate target position
            delta = maze.ACTIONS[action]
            to_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Get or create action node
            action_node = self._get_or_create_action_node(current_pos, to_pos)
            
            # Skip if we know it's blocked (but allow occasional retry)
            if action_node.result == ActionResult.BLOCKED:
                # Allow retry after many attempts to avoid getting stuck
                if action_node.attempts < 5 or self.timestep % 50 == 0:
                    logger.debug(f"  Action {action}: BLOCKED (retrying)")
                else:
                    logger.debug(f"  Action {action}: BLOCKED (skipping)")
                    continue
            
            # Calculate base GED (movement cost)
            ged = 0.3 + 0.7 * action_node.entropy
            
            # Calculate base IG (information gain)
            ig = 2.0 * action_node.entropy
            
            # Strong bonus for unvisited actions (exploration)
            if action_node.result == ActionResult.UNKNOWN:
                ig += 2.0  # Increased from 1.0
                
                # Use structural similarity to estimate value
                similar_actions = self._find_similar_actions(action_node)
                
                passed_similar = 0
                blocked_similar = 0
                total_weight = 0
                
                for similar, similarity in similar_actions[:5]:  # Top 5 similar
                    weight = similarity
                    total_weight += weight
                    
                    if similar.result == ActionResult.PASSED:
                        passed_similar += weight
                    elif similar.result == ActionResult.BLOCKED:
                        blocked_similar += weight
                
                if total_weight > 0:
                    # Adjust IG based on similar actions
                    success_rate = passed_similar / total_weight
                    ig *= (0.5 + success_rate)  # Boost if similar actions succeeded
                    
                    logger.debug(f"    Similar actions suggest {success_rate:.1%} success rate")
                
                # Global direction confidence (weakened to avoid over-reliance)
                dir_confidence = self._get_direction_confidence(action_node.direction)
                # Only apply mild influence (0.8 to 1.2 range)
                ig *= (0.8 + 0.4 * dir_confidence)
            
            # Goal bonus
            if self.goal_position and to_pos == self.goal_position:
                ig += 10.0
            
            # geDIG objective
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = (f, action_node)
            
            logger.debug(f"  Action {action} ({self.DIRECTION_NAMES.get(action_node.direction, '?')}): "
                        f"GED={ged:.2f}, IG={ig:.2f}, f={f:.2f}")
        
        if not action_scores:
            # All directions blocked - retry
            logger.debug("  All known directions blocked, retrying...")
            for action in range(4):
                delta = maze.ACTIONS[action]
                to_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                action_node = self._get_or_create_action_node(current_pos, to_pos)
                if action_node.result == ActionResult.BLOCKED:
                    action_scores[action] = (0, action_node)
        
        if not action_scores:
            return 0  # Fallback
        
        # Choose best action
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a][0])
        best_score, best_node = action_scores[best_action]
        
        # Store for next update
        self.last_action = best_action
        self.last_target = best_node.to_pos
        
        # Store history
        self.gediq_history.append({
            'step': self.timestep,
            'action': best_action,
            'from_pos': current_pos,
            'to_pos': best_node.to_pos,
            'f': best_score
        })
        
        logger.debug(f"  → Choosing: {self.DIRECTION_NAMES.get(best_node.direction, '?')}")
        
        return best_action
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get navigator metrics."""
        total_actions = len(self.action_nodes)
        blocked_actions = sum(1 for n in self.action_nodes.values() 
                             if n.result == ActionResult.BLOCKED)
        passed_actions = sum(1 for n in self.action_nodes.values() 
                            if n.result == ActionResult.PASSED)
        unknown_actions = sum(1 for n in self.action_nodes.values() 
                             if n.result == ActionResult.UNKNOWN)
        
        avg_entropy = np.mean([n.entropy for n in self.action_nodes.values()])
        
        # Global knowledge stats
        global_stats = {}
        for direction, stats in self.direction_stats.items():
            total = stats["passed"] + stats["blocked"]
            if total > 0:
                global_stats[direction] = {
                    "success_rate": stats["passed"] / total,
                    "total_attempts": total
                }
        
        return {
            'total_action_nodes': total_actions,
            'blocked_actions': blocked_actions,
            'passed_actions': passed_actions,
            'unknown_actions': unknown_actions,
            'average_entropy': avg_entropy,
            'wall_hits': self.wall_hits,
            'timesteps': self.timestep,
            'global_knowledge': global_stats
        }