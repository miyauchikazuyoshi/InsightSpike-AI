"""InsightSpike package metadata"""
import importlib.util
import os


class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.8.0"


# Check if we're in LITE_MODE to skip heavy dependencies
LITE_MODE = os.environ.get("INSIGHTSPIKE_LITE_MODE", "0") == "1"

# Legacy compatibility exports - import the config.py file specifically
from .config import get_config

# Export new main agent for easy access
if not LITE_MODE:
    try:
        from .implementations.agents.main_agent import CycleResult, MainAgent
    except ImportError:
        # Define a placeholder if main_agent is not available
        class MainAgent:
            def __init__(self):
                pass

            def initialize(self):
                return False

            def process_question(self, question, **kwargs):
                return {"response": "MainAgent not available", "success": False}

        class CycleResult:
            def __init__(self, **kwargs):
                pass

else:
    # In lite mode, always use placeholder
    class MainAgent:
        def __init__(self):
            pass

        def initialize(self):
            return False

        def process_question(self, question, **kwargs):
            return {"response": "MainAgent not available (lite mode)", "success": False}

    class CycleResult:
        def __init__(self, **kwargs):
            pass


# New unified layer exports (recommended for new code)
if not LITE_MODE:
    try:
        from .implementations.layers.layer1_error_monitor import ErrorMonitor
        from .implementations.layers.layer2_memory_manager import L2MemoryManager
        from .implementations.layers.layer4_llm_interface import get_llm_provider
    except ImportError:
        # Fallback if core layers are not available
        ErrorMonitor = None
        L2MemoryManager = None
        get_llm_provider = None
else:
    # In lite mode, skip heavy dependencies
    ErrorMonitor = None
    L2MemoryManager = None
    get_llm_provider = None

# Optional Layer3 with PyTorch dependency - skip in LITE_MODE
if not LITE_MODE:
    try:
        from .implementations.layers.layer3_graph_reasoner import L3GraphReasoner
    except ImportError:
        L3GraphReasoner = None
else:
    L3GraphReasoner = None

# Generic agent system exports
if not LITE_MODE:
    try:
        from .core.base.generic_interfaces import (
            EnvironmentInterface,
            InsightMoment,
            TaskType,
        )
        from .implementations.agents.agent_factory import (
            AgentConfigBuilder,
            InsightSpikeAgentFactory,
            create_configured_maze_agent,
            create_maze_agent,
        )
        from .implementations.agents.generic_agent import GenericInsightSpikeAgent
    except ImportError:
        # Fallback if generic agents are not available
        GenericInsightSpikeAgent = None
        InsightSpikeAgentFactory = None
        create_maze_agent = None
        create_configured_maze_agent = None
        AgentConfigBuilder = None
        TaskType = None
        EnvironmentInterface = None
        InsightMoment = None
else:
    # In lite mode, skip heavy dependencies
    GenericInsightSpikeAgent = None
    InsightSpikeAgentFactory = None
    create_maze_agent = None
    create_configured_maze_agent = None
    AgentConfigBuilder = None
    TaskType = None
    EnvironmentInterface = None
    InsightMoment = None

# Standalone reasoner export - skip in LITE_MODE
if not LITE_MODE:
    try:
        from .tools.standalone.standalone_l3 import (
            StandaloneL3GraphReasoner,
            analyze_documents_simple,
            create_standalone_reasoner,
        )
    except ImportError:
        # Fallback if standalone reasoner is not available
        StandaloneL3GraphReasoner = None
        create_standalone_reasoner = None
        analyze_documents_simple = None
else:
    StandaloneL3GraphReasoner = None
    create_standalone_reasoner = None
    analyze_documents_simple = None
    create_standalone_reasoner = None
    analyze_documents_simple = None

from .config import get_config

# Import the unified config system
from .config.legacy_config import Config


# Create a legacy config module object for backward compatibility
class LegacyConfigModule:
    def __init__(self):
        from .config.legacy_config import get_legacy_config

        legacy = get_legacy_config()
        for key, value in legacy.items():
            setattr(self, key, value)

    def timestamp(self):
        from datetime import datetime

        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


config = LegacyConfigModule()

# Legacy module exports for compatibility - new organized structure
from . import utils
from .detection import eureka_spike
from .metrics import graph_metrics

# Version info
__version__ = About.VERSION


# Export About as a module attribute for backward compatibility
def about():
    """Return package information as a dictionary."""
    return {"name": About.NAME, "version": About.VERSION}


# Main exports
__all__ = [
    "MainAgent",
    "CycleResult",
    "get_config",
    "Config",
    "About",
    "about",
    "graph_metrics",
    "eureka_spike",
    "config",
    "utils",
    # Layer exports
    "ErrorMonitor",
    "L2MemoryManager",
    "get_llm_provider",
    "L3GraphReasoner",
    # Generic agent system exports
    "GenericInsightSpikeAgent",
    "InsightSpikeAgentFactory",
    "create_maze_agent",
    "create_configured_maze_agent",
    "AgentConfigBuilder",
    "TaskType",
    "EnvironmentInterface",
    "InsightMoment",
    # Standalone reasoner exports
    "StandaloneL3GraphReasoner",
    "create_standalone_reasoner",
    "analyze_documents_simple",
]
