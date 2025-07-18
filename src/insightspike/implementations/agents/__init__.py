"""
InsightSpike Agents Module
=========================

Provides different agent implementations for various use cases.
"""

# Import factory functions
from .agent_factory import (
    AgentConfigBuilder,
    InsightSpikeAgentFactory,
    create_maze_agent,
    create_qa_agent,
)
from .configurable_agent import AgentConfig, AgentMode, ConfigurableAgent
from .generic_agent import GenericInsightSpikeAgent

# Import main agents
from .main_agent import CycleResult, MainAgent

# Define what's available when importing from this module
__all__ = [
    # Core agents
    "MainAgent",
    "ConfigurableAgent", 
    "GenericInsightSpikeAgent",
    
    # Configuration
    "AgentConfig",
    "AgentMode",
    "CycleResult",
    
    # Factory functions
    "create_qa_agent",
    "create_maze_agent",
    "InsightSpikeAgentFactory",
    "AgentConfigBuilder",
]