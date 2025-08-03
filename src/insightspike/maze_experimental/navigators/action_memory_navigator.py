"""Action-based memory navigator that memorizes each movement attempt as a node."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from enum import Enum

from ...environments.maze import MazeObservation
from ..maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


class ActionResult(Enum):
    """Result of an action attempt."""
    BLOCKED = "blocked"     # Hit a wall
    PASSED = "passed"       # Successfully moved
    UNKNOWN = "unknown"     # Not yet attempted


@dataclass
class ActionNode:
    """A single action attempt from position A to position B."""
    from_pos: Tuple[int, int]    # Starting position (a1, a2)
    to_pos: Tuple[int, int]      # Target position (b1, b2)
    result: ActionResult          # What happened
    attempts: int = 0             # Number of times tried
    vector: Optional[np.ndarray] = None
    
    @property
    def action_key(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Unique key for this action."""
        return (self.from_pos, self.to_pos)
    
    @property
    def entropy(self) -> float:
        """Entropy of this action node.
        - Unknown: High entropy (1.0)
        - Blocked: Medium entropy (0.5)
        - Passed: Low entropy (0.1)
        """
        if self.result == ActionResult.UNKNOWN:
            return 1.0
        elif self.result == ActionResult.BLOCKED:
            return 0.5
        else:  # PASSED
            return 0.1


class ActionMemoryNavigator:
    """Navigator using action-based memory nodes."""
    
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
        self.w_ged = 1.0  # Weight for GED
        self.k_ig = 2.0   # Weight for IG
        
        # Track metrics
        self.wall_hits = 0
        self.gediq_history = []
        
        # Initialize with starting moves (as you suggested)
        self._init_starting_knowledge()
        
    def _init_embedder(self):
        """Initialize feature embedder for action nodes."""
        np.random.seed(42)
        
        # Base vectors for different results
        self.result_vectors = {
            ActionResult.PASSED: np.random.randn(self.feature_dim),
            ActionResult.BLOCKED: np.random.randn(self.feature_dim), 
            ActionResult.UNKNOWN: np.zeros(self.feature_dim)
        }
        
        # Normalize
        for key in [ActionResult.PASSED, ActionResult.BLOCKED]:
            self.result_vectors[key] /= np.linalg.norm(self.result_vectors[key])
        
        # Direction vectors
        self.direction_vectors = {
            (0, -1): np.random.randn(self.feature_dim),  # Up
            (1, 0): np.random.randn(self.feature_dim),   # Right
            (0, 1): np.random.randn(self.feature_dim),   # Down
            (-1, 0): np.random.randn(self.feature_dim)   # Left
        }
        
        for key in self.direction_vectors:
            self.direction_vectors[key] /= np.linalg.norm(self.direction_vectors[key])
    
    def _init_starting_knowledge(self):
        """Initialize with basic movement knowledge from (1,1)."""
        # Starting position movements
        start = (1, 1)
        
        # Create unknown action nodes for all 4 directions
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            to_pos = (start[0] + dy, start[1] + dx)
            action_node = ActionNode(
                from_pos=start,
                to_pos=to_pos,
                result=ActionResult.UNKNOWN
            )
            action_node.vector = self._get_action_vector(action_node)
            self.action_nodes[action_node.action_key] = action_node
            
            logger.info(f"Initialized action: {start} → {to_pos} (unknown)")
    
    def _get_action_vector(self, node: ActionNode) -> np.ndarray:
        """Create feature vector for an action node."""
        vector = np.zeros(self.feature_dim)
        
        # Result component (main entropy carrier)
        vector += 0.6 * self.result_vectors[node.result]
        
        # Direction component
        dx = node.to_pos[1] - node.from_pos[1]
        dy = node.to_pos[0] - node.from_pos[0]
        if (dx, dy) in self.direction_vectors:
            vector += 0.2 * self.direction_vectors[(dx, dy)]
        
        # Position encoding (from position)
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
            
        if not moved:
            self.wall_hits += 1
            logger.info(f"Action {from_pos} → {to_pos}: BLOCKED (entropy: {node.entropy:.2f})")
        else:
            logger.info(f"Action {from_pos} → {to_pos}: PASSED (entropy: {node.entropy:.2f})")
    
    def _find_similar_actions(self, action_node: ActionNode, 
                             similarity_threshold: float = 0.7) -> List[ActionNode]:
        """Find similar action nodes based on vector similarity."""
        similar = []
        
        for other_node in self.action_nodes.values():
            if other_node.action_key == action_node.action_key:
                continue
                
            # Cosine similarity
            similarity = np.dot(action_node.vector, other_node.vector)
            
            if similarity >= similarity_threshold:
                similar.append(other_node)
                
        return similar
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action based on action memory nodes."""
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
        
        for action in range(4):  # 4 directions
            # Calculate target position
            delta = maze.ACTIONS[action]
            to_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Get or create action node
            action_node = self._get_or_create_action_node(current_pos, to_pos)
            
            # Skip if we know it's blocked
            if action_node.result == ActionResult.BLOCKED:
                logger.debug(f"  Action {action}: {current_pos} → {to_pos} is BLOCKED")
                continue
            
            # Calculate GED (movement cost)
            # Lower entropy = lower cost (we prefer known paths)
            ged = 0.3 + 0.7 * action_node.entropy
            
            # Calculate IG (information gain)
            # Higher entropy = higher information gain
            ig = 2.0 * action_node.entropy
            
            # Bonus for unvisited actions
            if action_node.result == ActionResult.UNKNOWN:
                ig += 1.0
                
            # Find similar actions to estimate value
            similar_actions = self._find_similar_actions(action_node)
            for similar in similar_actions:
                if similar.result == ActionResult.PASSED:
                    ig += 0.3  # Bonus if similar actions succeeded
                elif similar.result == ActionResult.BLOCKED:
                    ig -= 0.2  # Penalty if similar actions failed
            
            # Goal bonus
            if self.goal_position and to_pos == self.goal_position:
                ig += 10.0
            
            # geDIG objective: f = w*GED - k*IG
            f = self.w_ged * ged - self.k_ig * ig
            action_scores[action] = (f, action_node)
            
            logger.debug(f"  Action {action}: {current_pos} → {to_pos}, "
                        f"entropy={action_node.entropy:.2f}, GED={ged:.2f}, "
                        f"IG={ig:.2f}, f={f:.2f}")
        
        if not action_scores:
            # All directions blocked - retry least recent
            for action in range(4):
                delta = maze.ACTIONS[action]
                to_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                action_node = self._get_or_create_action_node(current_pos, to_pos)
                if action_node.result == ActionResult.BLOCKED:
                    action_scores[action] = (0, action_node)
        
        if not action_scores:
            return 0  # Fallback
        
        # Choose best action (lowest geDIG score)
        best_action = min(action_scores.keys(), key=lambda a: action_scores[a][0])
        best_score, best_node = action_scores[best_action]
        
        # Store for next update
        self.last_action = best_action
        self.last_target = best_node.to_pos
        
        # Store geDIG history
        self.gediq_history.append({
            'step': self.timestep,
            'action': best_action,
            'from_pos': current_pos,
            'to_pos': best_node.to_pos,
            'entropy': best_node.entropy,
            'f': best_score
        })
        
        logger.debug(f"  → Choosing action {best_action}: {current_pos} → {best_node.to_pos}")
        
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
        
        return {
            'total_action_nodes': total_actions,
            'blocked_actions': blocked_actions,
            'passed_actions': passed_actions,
            'unknown_actions': unknown_actions,
            'average_entropy': avg_entropy,
            'wall_hits': self.wall_hits,
            'timesteps': self.timestep
        }