"""Backward compatibility wrapper stub.

Provides minimal API expected by tests that import BackwardCompatibleWrapper.

Behavior: delegates to IntegratedVectorGraphIndex while exposing legacy-named
methods/attributes. This is intentionally lightweight until the full
refactored index pipeline is restored.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from .integrated_vector_graph_index import IntegratedVectorGraphIndex

class BackwardCompatibleWrapper:
    def __init__(self, index_or_dimension: Any = 384):
        # Accept either an existing index instance or a dimension to construct one
        if isinstance(index_or_dimension, IntegratedVectorGraphIndex):
            self._index = index_or_dimension
        else:
            self._index = IntegratedVectorGraphIndex(dimension=int(index_or_dimension))

    # Legacy aliases
    @property
    def size(self) -> int:
        return self._index.size

    def add(self, vec, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        return self._index.add_vector(vec, text=text, metadata=metadata)

    def add_episode(self, episode):  # passthrough
        return self._index.add_episode(episode)

    def search(self, query_vec, top_k: int = 5):
        res = self._index.search(query_vec, top_k=top_k)
        # Normalize to list of (idx, score) for legacy callers
        if isinstance(res, tuple) and len(res) == 2:
            idxs, scores = res
            return list(zip(idxs, scores))
        return res

    # Legacy finder returning (indices, scores)
    def find_similar(self, query_vec, k: int = 10, namespace: str = "vectors") -> Tuple[List[int], List[float]]:
        return self._index.find_similar(query_vec, k=k)

    # Compatibility alias
    def search_vectors(self, query_vec, k: int = 10, namespace: str = "vectors") -> Tuple[List[int], List[float]]:
        return self.find_similar(query_vec, k=k, namespace=namespace)

    def export(self):
        return self._index.export()

    # Episodic helpers used by enhanced datastore
    def save_episodes(self, episodes: List[Dict[str, Any]], namespace: str = "episodes") -> bool:
        try:
            for ep in episodes:
                # Support either {'vec': ...} or {'embedding': ...}
                vec = ep.get("vec", ep.get("embedding"))
                text = ep.get("text", "")
                meta = ep.get("metadata", {})
                self._index.add_vector(vec, text=text, metadata=meta)
            return True
        except Exception:
            return False

    def load_episodes(self, namespace: str = "episodes") -> List[Dict[str, Any]]:
        # Reconstruct simple dicts from the index content
        exported = self._index.export()
        vecs = exported.get("vectors", [])
        texts = exported.get("texts", [])
        metas = exported.get("metadata", [])
        out: List[Dict[str, Any]] = []
        for i in range(len(vecs)):
            out.append({
                "id": str(i),
                "text": texts[i] if i < len(texts) else "",
                "vec": vecs[i],
                "metadata": metas[i] if i < len(metas) else {},
            })
        return out

__all__ = ["BackwardCompatibleWrapper"]
