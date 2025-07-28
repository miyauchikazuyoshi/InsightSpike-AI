"""
Core Components
==============

Core business logic components following clean architecture.
"""

# Import core data structures
from .episode import Episode
from .structures import CycleResult, GraphMetrics

# Import core components (lazy loading to avoid circular imports)
try:
    from .reasoning_engine import ReasoningEngine
    REASONING_ENGINE_AVAILABLE = True
except ImportError:
    ReasoningEngine = None
    REASONING_ENGINE_AVAILABLE = False

try:
    from .memory_controller import MemoryController
    MEMORY_CONTROLLER_AVAILABLE = True
except ImportError:
    MemoryController = None
    MEMORY_CONTROLLER_AVAILABLE = False

try:
    from .response_generator import ResponseGenerator
    RESPONSE_GENERATOR_AVAILABLE = True
except ImportError:
    ResponseGenerator = None
    RESPONSE_GENERATOR_AVAILABLE = False

# Legacy compatibility flags
EXPERIMENT_FRAMEWORK_AVAILABLE = False
LAYERS_AVAILABLE = False
MAIN_AGENT_AVAILABLE = False
GENERIC_AGENTS_AVAILABLE = False
STANDALONE_REASONER_AVAILABLE = False
INTERFACES_AVAILABLE = False

# Export list
__all__ = [
    "Episode",
    "CycleResult",
    "GraphMetrics",
]

if REASONING_ENGINE_AVAILABLE:
    __all__.append("ReasoningEngine")

if MEMORY_CONTROLLER_AVAILABLE:
    __all__.append("MemoryController")

if RESPONSE_GENERATOR_AVAILABLE:
    __all__.append("ResponseGenerator")
