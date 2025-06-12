"""
Maze Environment Implementation of Generic Interfaces
===================================================

Concrete implementations of the generic interfaces for maze navigation tasks.
This serves as a reference implementation and bridge for existing maze code.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .generic_interfaces import (
    ActionSpace, EnvironmentInterface, EnvironmentState, InsightDetectorInterface,
    InsightMoment, RewardNormalizer, StateEncoder, TaskType
)


class MazeEnvironmentAdapter(EnvironmentInterface):
    """Adapter for existing maze environments to generic interface"""
    
    def __init__(self, maze_size: int = 10, wall_density: float = 0.25):
        self.size = maze_size
        self.wall_density = wall_density
        self.current_pos = (0, 0)
        self.goal_pos = (maze_size - 1, maze_size - 1)
        self.visited_states = set()
        self.step_count = 0
        self.episode_count = 0
        
        # Generate maze
        self.maze = self._generate_maze()
        
    def _generate_maze(self) -> np.ndarray:
        """Generate a random maze with guaranteed path"""
        maze = np.zeros((self.size, self.size))
        
        # Add random walls
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                if random.random() < self.wall_density:
                    maze[i, j] = 1
        
        # Ensure start and goal are clear
        maze[0, 0] = 0
        maze[self.size - 1, self.size - 1] = 0
        
        return maze
    
    def get_state(self) -> EnvironmentState:
        """Get current maze state"""
        state_data = {
            'position': self.current_pos,
            'maze': self.maze,
            'goal': self.goal_pos,
            'visited': self.visited_states
        }
        
        return EnvironmentState(
            state_data=state_data,
            environment_type="maze_navigation",
            task_type=TaskType.NAVIGATION,
            state_shape=(self.size, self.size),
            step_count=self.step_count,
            episode_count=self.episode_count,
            metadata={
                'maze_size': self.size,
                'exploration_ratio': len(self.visited_states) / (self.size * self.size)
            }
        )
    
    def get_action_space(self) -> ActionSpace:
        """Get maze action space (4 discrete directions)"""
        return ActionSpace(
            action_type="discrete",
            action_dim=4,
            discrete_actions=["up", "right", "down", "left"]
        )
    
    def step(self, action: int) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """Execute action in maze"""
        self.step_count += 1
        
        # Define moves: up, right, down, left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        
        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        
        # Check bounds and walls
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and
            self.maze[new_pos[0], new_pos[1]] == 0):
            self.current_pos = new_pos
            self.visited_states.add(new_pos)
            reward = -0.01  # Small step penalty
            
            # Goal reward
            if new_pos == self.goal_pos:
                reward = 100.0
                done = True
            else:
                done = False
        else:
            # Wall collision
            reward = -1.0
            done = False
        
        info = {
            'collision': new_pos != self.current_pos,
            'exploration_ratio': len(self.visited_states) / (self.size * self.size),
            'distance_to_goal': abs(self.current_pos[0] - self.goal_pos[0]) + 
                               abs(self.current_pos[1] - self.goal_pos[1])
        }
        
        return self.get_state(), reward, done, info
    
    def reset(self) -> EnvironmentState:
        """Reset maze environment"""
        self.current_pos = (0, 0)
        self.visited_states = {(0, 0)}
        self.step_count = 0
        self.episode_count += 1
        return self.get_state()
    
    def get_task_type(self) -> TaskType:
        """Get task type"""
        return TaskType.NAVIGATION


class MazeStateEncoder(StateEncoder):
    """State encoder for maze environments"""
    
    def __init__(self, maze_size: int):
        self.maze_size = maze_size
        self.encoding_dim = maze_size * maze_size + 4  # maze + position + goal
    
    def encode_state(self, state: EnvironmentState) -> np.ndarray:
        """Encode maze state to vector"""
        state_data = state.state_data
        
        # Flatten maze
        maze_flat = state_data['maze'].flatten()
        
        # Position encoding
        pos_x, pos_y = state_data['position']
        goal_x, goal_y = state_data['goal']
        
        # Combine all features
        encoded = np.concatenate([
            maze_flat,
            [pos_x / self.maze_size, pos_y / self.maze_size, 
             goal_x / self.maze_size, goal_y / self.maze_size]
        ])
        
        return encoded
    
    def get_encoding_dim(self) -> int:
        """Get encoding dimension"""
        return self.encoding_dim
    
    def decode_state(self, encoded_state: np.ndarray) -> EnvironmentState:
        """Decode state (simplified implementation)"""
        # This is a simplified decoder - in practice you might want more sophisticated decoding
        maze_part = encoded_state[:-4].reshape((self.maze_size, self.maze_size))
        pos_part = encoded_state[-4:]
        
        pos_x = int(pos_part[0] * self.maze_size)
        pos_y = int(pos_part[1] * self.maze_size)
        goal_x = int(pos_part[2] * self.maze_size)
        goal_y = int(pos_part[3] * self.maze_size)
        
        state_data = {
            'position': (pos_x, pos_y),
            'maze': maze_part,
            'goal': (goal_x, goal_y),
            'visited': set()  # Cannot reconstruct visited states from encoding
        }
        
        return EnvironmentState(
            state_data=state_data,
            environment_type="maze_navigation",
            task_type=TaskType.NAVIGATION,
            state_shape=(self.maze_size, self.maze_size)
        )


class MazeRewardNormalizer(RewardNormalizer):
    """Reward normalizer for maze environments"""
    
    def __init__(self, maze_size: int):
        self.maze_size = maze_size
        # Expected bounds: wall penalty to goal reward
        self.min_reward = -1.0
        self.max_reward = 100.0
    
    def normalize_reward(self, reward: float, context: Dict[str, Any]) -> float:
        """Normalize maze reward to [-1, 1] range"""
        # Clamp and normalize
        clamped = max(self.min_reward, min(self.max_reward, reward))
        normalized = 2 * (clamped - self.min_reward) / (self.max_reward - self.min_reward) - 1
        return normalized
    
    def get_reward_bounds(self) -> Tuple[float, float]:
        """Get reward bounds"""
        return (self.min_reward, self.max_reward)


class MazeInsightDetector(InsightDetectorInterface):
    """Insight detector for maze navigation"""
    
    def __init__(self, maze_size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(TaskType.NAVIGATION, config)
        self.maze_size = maze_size
        
        # Maze-specific thresholds
        self.dged_threshold = config.get('dged_threshold', -0.3) if config else -0.3
        self.dig_threshold = config.get('dig_threshold', 1.0) if config else 1.0
        
        # Tracking for calculations
        self.exploration_history = []
        self.reward_history = []
        self.state_visit_count = {}
    
    def detect_insight(
        self, 
        current_state: EnvironmentState,
        action: Any,
        reward: float,
        next_state: EnvironmentState,
        context: Dict[str, Any]
    ) -> Optional[InsightMoment]:
        """Detect insight in maze navigation"""
        
        # Update internal tracking
        self.update_context(current_state, action, reward)
        
        # Calculate metrics
        dged = self.calculate_dged(context)
        dig = self.calculate_dig(context)
        
        # Check insight conditions
        insight_detected = False
        insight_type = ""
        description = ""
        
        # Primary condition: efficiency drop + high info gain
        if dged < self.dged_threshold and dig > self.dig_threshold:
            insight_detected = True
            insight_type = "strategic_breakthrough"
            description = f"Navigation strategy breakthrough: ΔGED={dged:.3f}, ΔIG={dig:.3f}"
        
        # Goal discovery condition
        elif reward > 50:  # Goal reward threshold
            insight_detected = True
            insight_type = "goal_discovery"
            description = f"Goal discovery insight: reached target with reward={reward:.1f}"
        
        # Exploration insight
        elif dig > 2.0:
            current_pos = next_state.state_data['position']
            if current_pos not in self.state_visit_count:
                insight_detected = True
                insight_type = "exploration_insight"
                description = f"Exploration insight: discovered new valuable area at {current_pos}"
        
        if insight_detected:
            insight = InsightMoment(
                episode=next_state.episode_count,
                step=next_state.step_count,
                insight_type=insight_type,
                description=description,
                dged_value=dged,
                dig_value=dig,
                confidence=min(abs(dged) + dig, 1.0),
                performance_impact=dig * 0.1,
                state=next_state,
                action=action,
                reward=reward,
                detection_method="maze_navigation"
            )
            
            self.insight_history.append(insight)
            return insight
        
        return None
    
    def calculate_dged(self, context: Dict[str, Any]) -> float:
        """Calculate ΔGED for maze navigation"""
        if len(self.reward_history) < 5:
            return 0.0
        
        # Calculate exploration efficiency change
        recent_efficiency = np.mean(self.reward_history[-5:])
        overall_efficiency = np.mean(self.reward_history)
        
        return recent_efficiency - overall_efficiency
    
    def calculate_dig(self, context: Dict[str, Any]) -> float:
        """Calculate ΔIG for maze navigation"""
        base_gain = max(self.reward_history[-1] if self.reward_history else 0, 0.1)
        
        # Exploration factor
        unique_states = len(set(self.exploration_history))
        total_states = len(self.exploration_history)
        exploration_factor = unique_states / max(total_states, 1)
        
        # Distance to goal factor (from context)
        distance_factor = 1.0
        if 'distance_to_goal' in context:
            max_distance = 2 * self.maze_size
            distance_factor = 1.0 - (context['distance_to_goal'] / max_distance)
        
        return base_gain * exploration_factor * (1 + distance_factor)
    
    def update_context(self, state: EnvironmentState, action: Any, reward: float):
        """Update context for maze navigation"""
        current_pos = state.state_data['position']
        
        self.exploration_history.append(current_pos)
        self.reward_history.append(reward)
        self.state_visit_count[current_pos] = self.state_visit_count.get(current_pos, 0) + 1
        
        # Keep history bounded
        if len(self.exploration_history) > 1000:
            self.exploration_history = self.exploration_history[-500:]
            self.reward_history = self.reward_history[-500:]


# Export concrete implementations
__all__ = [
    "MazeEnvironmentAdapter",
    "MazeStateEncoder", 
    "MazeRewardNormalizer",
    "MazeInsightDetector"
]
