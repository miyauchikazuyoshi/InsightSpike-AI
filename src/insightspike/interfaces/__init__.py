"""
InsightSpike Interfaces
======================

Clean protocol definitions for all major components.
Using Python's Protocol for structural subtyping (duck typing with type hints).
"""

from .agent import IAgent, IMemoryAgent
from .datastore import IDataStore, IEpisodeStore
from .embedder import IEmbedder
from .graph import IGraphBuilder, IGraphAnalyzer
from .llm import ILLMProvider
from .memory import IMemoryManager, IMemorySearch

__all__ = [
    # Agent interfaces
    "IAgent",
    "IMemoryAgent",
    # DataStore interfaces
    "IDataStore",
    "IEpisodeStore",
    # Embedder interface
    "IEmbedder",
    # Graph interfaces
    "IGraphBuilder",
    "IGraphAnalyzer",
    # LLM interface
    "ILLMProvider",
    # Memory interfaces
    "IMemoryManager",
    "IMemorySearch",
]