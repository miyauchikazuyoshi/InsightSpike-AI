"""Core InsightSpike-AI modules"""

# Configuration and management - removed to avoid circular import

# Data structures and framework - removed after refactoring
EXPERIMENT_FRAMEWORK_AVAILABLE = False

# Layer implementations - moved to implementations.layers
LAYERS_AVAILABLE = False

# Import agents - moved to implementations.agents
MAIN_AGENT_AVAILABLE = False

# Import generic agents and factory - removed after refactoring
GENERIC_AGENTS_AVAILABLE = False

# Import standalone reasoner - skip in LITE_MODE
import os

LITE_MODE = os.getenv("INSIGHTSPIKE_LITE_MODE", "0") == "1"

# Standalone reasoner - moved to tools.standalone
StandaloneL3GraphReasoner = None
create_standalone_reasoner = None
analyze_documents_simple = None
STANDALONE_REASONER_AVAILABLE = False

# Import interfaces - moved to core.base
INTERFACES_AVAILABLE = False

# Export list
base_exports = []

if EXPERIMENT_FRAMEWORK_AVAILABLE:
    base_exports.extend(
        [
            "BaseExperiment",
            "ExperimentConfig",
            "ExperimentResult",
            "PerformanceMetrics",
            "ExperimentSuite",
            "create_simple_experiment_config",
            "create_performance_metrics",
        ]
    )

if MAIN_AGENT_AVAILABLE:
    base_exports.append("MainAgent")

if GENERIC_AGENTS_AVAILABLE:
    base_exports.extend(
        [
            "GenericInsightSpikeAgent",
            "GenericMemoryManager",
            "GenericReasoner",
            "InsightSpikeAgentFactory",
            "AgentConfigBuilder",
            "create_maze_agent",
            "create_configured_maze_agent",
        ]
    )

if STANDALONE_REASONER_AVAILABLE:
    base_exports.extend(
        [
            "StandaloneL3GraphReasoner",
            "create_standalone_reasoner",
            "analyze_documents_simple",
        ]
    )

__all__ = base_exports
