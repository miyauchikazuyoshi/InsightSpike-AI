"""Backward compatibility wrapper stub.

Provides minimal API expected by tests that import BackwardCompatibleWrapper.

Behavior: delegates to IntegratedVectorGraphIndex while exposing legacy-named
methods/attributes. This is intentionally lightweight until the full
refactored index pipeline is restored.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
from .integrated_vector_graph_index import IntegratedVectorGraphIndex

class BackwardCompatibleWrapper:
    def __init__(self, dimension: int = 384):
        self._index = IntegratedVectorGraphIndex(dimension=dimension)

    # Legacy aliases
    @property
    def size(self) -> int:
        return self._index.size

    def add(self, vec, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        return self._index.add_vector(vec, text=text, metadata=metadata)

    def add_episode(self, episode):  # passthrough
        return self._index.add_episode(episode)

    def search(self, query_vec, top_k: int = 5):
        return self._index.search(query_vec, top_k=top_k)

    def export(self):
        return self._index.export()

__all__ = ["BackwardCompatibleWrapper"]
