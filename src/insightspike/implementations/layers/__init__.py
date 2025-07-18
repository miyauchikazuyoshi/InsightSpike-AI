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
from .layer4_llm_interface import (
    L4LLMInterface,
    LLMConfig,
    LLMProviderType,
    get_llm_provider,
)
from .layer4_prompt_builder import PromptBuilder

# Supporting components
from .scalable_graph_builder import ScalableGraphBuilder

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
    
    # Layer 4
    "L4LLMInterface",
    "LLMConfig",
    "LLMProviderType", 
    "get_llm_provider",
    "PromptBuilder",
    
    # Supporting
    "ScalableGraphBuilder",
]