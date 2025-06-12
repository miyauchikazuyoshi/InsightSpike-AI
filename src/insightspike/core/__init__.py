"""Core InsightSpike-AI modules"""

# Configuration and management
from .config_manager import ConfigManager

# Data structures and framework
try:
    from .experiment_framework import (
        BaseExperiment,
        ExperimentConfig, 
        ExperimentResult,
        PerformanceMetrics,
        ExperimentSuite,
        create_simple_experiment_config,
        create_performance_metrics
    )
    EXPERIMENT_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Experiment framework not available: {e}")
    EXPERIMENT_FRAMEWORK_AVAILABLE = False

# Layer implementations
try:
    from .layers import *
    LAYERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some core layers not available: {e}")
    LAYERS_AVAILABLE = False

# Import agents
try:
    from .agents.main_agent import MainAgent, CycleResult
    MAIN_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MainAgent not available: {e}")
    MAIN_AGENT_AVAILABLE = False

# Import generic agents and factory
try:
    from .agents.generic_agent import (
        GenericInsightSpikeAgent, GenericMemoryManager, GenericReasoner
    )
    from .agents.agent_factory import (
        InsightSpikeAgentFactory, AgentConfigBuilder,
        create_maze_agent, create_configured_maze_agent
    )
    GENERIC_AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Generic agents not available: {e}")
    GENERIC_AGENTS_AVAILABLE = False

# Import standalone reasoner
try:
    from .reasoners.standalone_l3 import (
        StandaloneL3GraphReasoner, create_standalone_reasoner,
        analyze_documents_simple
    )
    STANDALONE_REASONER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Standalone reasoner not available: {e}")
    STANDALONE_REASONER_AVAILABLE = False

# Import interfaces
try:
    from .interfaces import *
    INTERFACES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some interfaces not available: {e}")
    INTERFACES_AVAILABLE = False

# Export list
base_exports = ["ConfigManager"]

if EXPERIMENT_FRAMEWORK_AVAILABLE:
    base_exports.extend([
        "BaseExperiment", "ExperimentConfig", "ExperimentResult", 
        "PerformanceMetrics", "ExperimentSuite",
        "create_simple_experiment_config", "create_performance_metrics"
    ])

if MAIN_AGENT_AVAILABLE:
    base_exports.append("MainAgent")

if GENERIC_AGENTS_AVAILABLE:
    base_exports.extend([
        "GenericInsightSpikeAgent", "GenericMemoryManager", "GenericReasoner",
        "InsightSpikeAgentFactory", "AgentConfigBuilder",
        "create_maze_agent", "create_configured_maze_agent"
    ])

if STANDALONE_REASONER_AVAILABLE:
    base_exports.extend([
        "StandaloneL3GraphReasoner", "create_standalone_reasoner",
        "analyze_documents_simple"
    ])

__all__ = base_exports
