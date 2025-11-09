"""
InsightSpike Core Layers
=======================

4-Layer neurobiologically-inspired architecture.
"""

# Layer 1: Error Monitor (Cerebellum analog)
from .layer1_error_monitor import ErrorMonitor
from .layer2_compatibility import CompatibleL2MemoryManager

# Layer 2: Memory Manager (Hippocampus + Locus Coeruleus analog)
from .layer2_memory_manager import L2MemoryManager, MemoryConfig, MemoryMode

# Layer 3: Graph Reasoner (Prefrontal Cortex analog)
try:
    from .layer3_graph_reasoner import L3GraphReasoner

    GRAPH_REASONER_AVAILABLE = True
except ImportError:
    L3GraphReasoner = None
    GRAPH_REASONER_AVAILABLE = False

# Layer 4: Language Interface (Broca's/Wernicke's areas analog)
# Heavy imports are deferred to attribute access to avoid import-time issues
# in lightweight test environments that stub out torch/torch_geometric.

# Supporting components (lightweight)
from .scalable_graph_builder import ScalableGraphBuilder

# from .layer4_prompt_builder import PromptBuilder  # Temporarily disabled due to missing interfaces module


__all__ = [
    # Layer 1
    "ErrorMonitor",
    # Layer 2
    "L2MemoryManager",
    "MemoryConfig",
    "MemoryMode",
    "CompatibleL2MemoryManager",
    # Layer 3
    "L3GraphReasoner",
    "GRAPH_REASONER_AVAILABLE",
    # Layer 4 (lazy)
    "L4LLMInterface",
    "LLMConfig",
    "LLMProviderType",
    "get_llm_provider",
    # "PromptBuilder",  # Temporarily disabled
    # Supporting
    "ScalableGraphBuilder",
]


def __getattr__(name):  # PEP 562 lazy attribute access for heavy L4 imports
    if name in {"L4LLMInterface", "LLMConfig", "LLMProviderType", "get_llm_provider"}:
        try:
            from .layer4_llm_interface import (
                L4LLMInterface as _L4,
                LLMConfig as _Cfg,
                LLMProviderType as _PT,
                get_llm_provider as _get,
            )
            globals().update({
                "L4LLMInterface": _L4,
                "LLMConfig": _Cfg,
                "LLMProviderType": _PT,
                "get_llm_provider": _get,
            })
            return globals()[name]
        except Exception as e:
            raise AttributeError(f"Could not load {name}: {e}")
    raise AttributeError(name)
