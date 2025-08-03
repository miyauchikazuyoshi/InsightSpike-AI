"""Simple action memory navigator - focus on core concept."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..environments.maze import MazeObservation
from ..config.maze_config import MazeNavigatorConfig


logger = logging.getLogger(__name__)


@dataclass
class ActionMemory:
    """Simple action memory: A→B movement."""
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    success: bool  # True if moved, False if blocked
    attempts: int = 1
    
    @property
    def key(self):
        return (self.from_pos, self.to_pos)
    
    @property
    def direction(self):
        dx = self.to_pos[1] - self.from_pos[1]
        dy = self.to_pos[0] - self.from_pos[0]
        return (dx, dy)


class SimpleActionNavigator:
    """Simple navigator using action memories."""
    
    def __init__(self, config: MazeNavigatorConfig):
        self.config = config
        self.memories: Dict[Tuple, ActionMemory] = {}
        self.current_pos = None
        self.timestep = 0
        self.wall_hits = 0
        self.goal_pos = None
        
        # For tracking last action
        self.last_action = None
        self.last_target = None
        
        logger.info("Initialized Simple Action Navigator")
    
    def _remember_action(self, from_pos: Tuple[int, int], 
                        to_pos: Tuple[int, int], 
                        success: bool):
        """Remember an action result."""
        key = (from_pos, to_pos)
        
        if key in self.memories:
            # Update existing memory
            memory = self.memories[key]
            memory.attempts += 1
            # Only update if we have new information
            if memory.success != success:
                logger.warning(f"Contradiction: {from_pos}→{to_pos} was {memory.success}, now {success}")
        else:
            # Create new memory
            self.memories[key] = ActionMemory(from_pos, to_pos, success)
            
        if not success:
            self.wall_hits += 1
            logger.info(f"Blocked: {from_pos} → {to_pos}")
        else:
            logger.info(f"Success: {from_pos} → {to_pos}")
    
    def _find_similar_memories(self, from_pos: Tuple[int, int], 
                              to_pos: Tuple[int, int]) -> List[ActionMemory]:
        """Find similar action memories."""
        target_dir = (to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])
        similar = []
        
        for memory in self.memories.values():
            # Same direction?
            if memory.direction == target_dir:
                similar.append(memory)
        
        return similar
    
    def _estimate_success_probability(self, from_pos: Tuple[int, int], 
                                     to_pos: Tuple[int, int]) -> float:
        """Estimate probability of success based on similar actions."""
        key = (from_pos, to_pos)
        
        # If we already tried this exact action
        if key in self.memories:
            memory = self.memories[key]
            if memory.success:
                return 0.95  # High confidence it will work
            else:
                # Maybe retry occasionally
                return 0.05 if memory.attempts < 3 else 0.01
        
        # Find similar actions
        similar = self._find_similar_memories(from_pos, to_pos)
        
        if not similar:
            return 0.5  # Unknown = neutral
        
        # Calculate success rate from similar actions
        successes = sum(1 for m in similar if m.success)
        total = len(similar)
        
        # Weighted by distance
        weighted_success = 0.0
        total_weight = 0.0
        
        for memory in similar:
            # Distance between starting positions
            dist = abs(memory.from_pos[0] - from_pos[0]) + abs(memory.from_pos[1] - from_pos[1])
            weight = 1.0 / (1.0 + dist * 0.2)  # Closer = more relevant
            
            if memory.success:
                weighted_success += weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_success / total_weight
        else:
            return 0.5
    
    def decide_action(self, observation: MazeObservation, maze) -> int:
        """Decide action based on memories."""
        current_pos = observation.position
        self.timestep += 1
        
        # Update memory from last action
        if self.last_action is not None and self.current_pos is not None:
            # Did we move?
            moved = (current_pos != self.current_pos)
            self._remember_action(self.current_pos, self.last_target, moved)
        
        self.current_pos = current_pos
        
        # Check for goal
        if observation.is_goal and self.goal_pos is None:
            self.goal_pos = current_pos
            logger.info(f"🎯 Goal discovered at {current_pos}!")
        
        # Evaluate each possible action
        action_values = {}
        
        for action in range(4):
            delta = maze.ACTIONS[action]
            to_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Estimate success probability
            success_prob = self._estimate_success_probability(current_pos, to_pos)
            
            # Information gain (higher for unknown/uncertain)
            uncertainty = min(success_prob, 1.0 - success_prob) * 2  # 0 to 1
            ig = 1.0 + 2.0 * uncertainty  # 1 to 3
            
            # Add novelty bonus for unvisited state transitions
            if (current_pos, to_pos) not in self.memories:
                ig += 3.0  # Strong bonus for new transitions
            
            # Movement cost (lower for likely success)
            ged = 2.0 - success_prob  # 1 to 2
            
            # Goal bonus
            if self.goal_pos and to_pos == self.goal_pos:
                ig += 10.0
            
            # geDIG score (lower is better)
            score = ged - 2.0 * ig
            action_values[action] = score
            
            logger.debug(f"  Action {action}: success_prob={success_prob:.2f}, "
                        f"IG={ig:.2f}, GED={ged:.2f}, score={score:.2f}")
        
        # Choose best action
        if action_values:
            best_action = min(action_values.keys(), key=lambda a: action_values[a])
        else:
            best_action = 0
        
        # Store for next update
        delta = maze.ACTIONS[best_action]
        self.last_action = best_action
        self.last_target = (current_pos[0] + delta[0], current_pos[1] + delta[1])
        
        return best_action
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        total = len(self.memories)
        successful = sum(1 for m in self.memories.values() if m.success)
        blocked = total - successful
        
        # Direction statistics
        dir_stats = {}
        for memory in self.memories.values():
            dir_name = {(0,-1): 'up', (1,0): 'right', (0,1): 'down', (-1,0): 'left'}.get(memory.direction, 'other')
            if dir_name not in dir_stats:
                dir_stats[dir_name] = {'success': 0, 'blocked': 0}
            
            if memory.success:
                dir_stats[dir_name]['success'] += 1
            else:
                dir_stats[dir_name]['blocked'] += 1
        
        return {
            'total_memories': total,
            'successful_actions': successful,
            'blocked_actions': blocked,
            'wall_hits': self.wall_hits,
            'timesteps': self.timestep,
            'direction_stats': dir_stats
        }