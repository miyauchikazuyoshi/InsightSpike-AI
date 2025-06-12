"""
Agent Factory and Environment Adapters
======================================

Factory methods and adapters to easily create InsightSpike agents
for different environments and domains.
"""

from typing import Any, Dict, Optional

from ..interfaces.generic_interfaces import TaskType
from ..interfaces.maze_implementation import (
    MazeEnvironmentAdapter, MazeInsightDetector, MazeRewardNormalizer,
    MazeStateEncoder
)
from .generic_agent import GenericInsightSpikeAgent


class InsightSpikeAgentFactory:
    """Factory for creating InsightSpike agents for different domains"""
    
    @staticmethod
    def create_maze_agent(
        agent_id: str = "InsightSpike-Maze",
        maze_size: int = 10,
        wall_density: float = 0.25,
        config: Optional[Dict[str, Any]] = None
    ) -> GenericInsightSpikeAgent:
        """Create an InsightSpike agent for maze navigation"""
        
        # Create environment and components
        environment = MazeEnvironmentAdapter(maze_size, wall_density)
        state_encoder = MazeStateEncoder(maze_size)
        reward_normalizer = MazeRewardNormalizer(maze_size)
        insight_detector = MazeInsightDetector(maze_size, config)
        
        # Create agent
        agent = GenericInsightSpikeAgent(
            agent_id=agent_id,
            environment=environment,
            insight_detector=insight_detector,
            state_encoder=state_encoder,
            reward_normalizer=reward_normalizer,
            config=config
        )
        
        return agent
    
    @staticmethod
    def create_custom_agent(
        agent_id: str,
        environment,
        insight_detector,
        state_encoder,
        reward_normalizer,
        config: Optional[Dict[str, Any]] = None
    ) -> GenericInsightSpikeAgent:
        """Create an InsightSpike agent with custom components"""
        
        return GenericInsightSpikeAgent(
            agent_id=agent_id,
            environment=environment,
            insight_detector=insight_detector,
            state_encoder=state_encoder,
            reward_normalizer=reward_normalizer,
            config=config
        )
    
    @staticmethod
    def get_default_config(task_type: TaskType) -> Dict[str, Any]:
        """Get default configuration for different task types"""
        
        configs = {
            TaskType.NAVIGATION: {
                'learning_rate': 0.15,
                'exploration_rate': 0.4,
                'exploration_decay': 0.995,
                'min_exploration': 0.05,
                'dged_threshold': -0.3,
                'dig_threshold': 1.0
            },
            TaskType.OPTIMIZATION: {
                'learning_rate': 0.1,
                'exploration_rate': 0.3,
                'exploration_decay': 0.99,
                'min_exploration': 0.1,
                'dged_threshold': -0.5,
                'dig_threshold': 1.5
            },
            TaskType.GAME_PLAYING: {
                'learning_rate': 0.2,
                'exploration_rate': 0.5,
                'exploration_decay': 0.998,
                'min_exploration': 0.02,
                'dged_threshold': -0.2,
                'dig_threshold': 0.8
            },
            TaskType.CUSTOM: {
                'learning_rate': 0.1,
                'exploration_rate': 0.3,
                'exploration_decay': 0.995,
                'min_exploration': 0.05,
                'dged_threshold': -0.3,
                'dig_threshold': 1.0
            }
        }
        
        return configs.get(task_type, configs[TaskType.CUSTOM])


class AgentConfigBuilder:
    """Builder pattern for agent configuration"""
    
    def __init__(self):
        self.config = {}
    
    def learning_rate(self, lr: float):
        """Set learning rate"""
        self.config['learning_rate'] = lr
        return self
    
    def exploration_params(self, rate: float, decay: float, min_rate: float):
        """Set exploration parameters"""
        self.config['exploration_rate'] = rate
        self.config['exploration_decay'] = decay
        self.config['min_exploration'] = min_rate
        return self
    
    def insight_thresholds(self, dged_threshold: float, dig_threshold: float):
        """Set insight detection thresholds"""
        self.config['dged_threshold'] = dged_threshold
        self.config['dig_threshold'] = dig_threshold
        return self
    
    def memory_capacity(self, capacity: int):
        """Set memory capacity"""
        self.config['memory_capacity'] = capacity
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build configuration dictionary"""
        return self.config.copy()


# Convenience functions for quick agent creation
def create_maze_agent(
    maze_size: int = 10,
    agent_config: Optional[Dict[str, Any]] = None
) -> GenericInsightSpikeAgent:
    """Quick function to create a maze agent"""
    return InsightSpikeAgentFactory.create_maze_agent(
        maze_size=maze_size,
        config=agent_config
    )


def create_configured_maze_agent(
    maze_size: int = 10,
    learning_rate: float = 0.15,
    exploration_rate: float = 0.4,
    dged_threshold: float = -0.3,
    dig_threshold: float = 1.0
) -> GenericInsightSpikeAgent:
    """Create maze agent with specific parameters"""
    
    config = (AgentConfigBuilder()
              .learning_rate(learning_rate)
              .exploration_params(exploration_rate, 0.995, 0.05)
              .insight_thresholds(dged_threshold, dig_threshold)
              .build())
    
    return create_maze_agent(maze_size, config)


# Export factory and convenience functions
__all__ = [
    "InsightSpikeAgentFactory",
    "AgentConfigBuilder",
    "create_maze_agent",
    "create_configured_maze_agent"
]
