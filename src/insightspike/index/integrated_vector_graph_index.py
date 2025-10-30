"""Lightweight stub for IntegratedVectorGraphIndex.

This stub is provided to allow test collection and backward compatibility
checks to run in environments where the full integrated index implementation
has been temporarily removed/refactored.

Design goals:
- Minimal state (vectors + optional metadata)
- Deterministic, dependency-light (numpy optional)
- API surface matching usages in tests (add, search, size)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

try:  # numpy optional for deterministic fallback
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore

class IntegratedVectorGraphIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectors: List[List[float]] = []
        self._texts: List[str] = []
        self._meta: List[Dict[str, Any]] = []

    # Backward compatible alias used in migration helper
    @property
    def size(self) -> int:  # matches expected attribute
        return len(self._vectors)

    def add_vector(self, vec, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        if _np is not None and not isinstance(vec, list):  # normalize to list
            vec = list(map(float, vec))
        elif isinstance(vec, list):
            vec = [float(x) for x in vec]
        self._vectors.append(vec)
        self._texts.append(text)
        self._meta.append(metadata or {})
        return len(self._vectors) - 1

    # Legacy style convenience
    def add_episode(self, episode):  # episode expected to have vec + text
        vec = getattr(episode, 'vec', None)
        text = getattr(episode, 'text', '')
        meta = {k: getattr(episode, k) for k in ('c', 'metadata') if hasattr(episode, k)}
        return self.add_vector(vec, text=text, metadata=meta)

    def search(self, query_vec, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self._vectors:
            return []
        # Simple cosine similarity fallback
        if _np is None:
            return [(i, 0.0) for i in range(min(top_k, len(self._vectors)))]
        q = _np.array(query_vec, dtype=float)
        qn = q / (_np.linalg.norm(q) + 1e-12)
        mat = _np.array(self._vectors, dtype=float)
        norms = _np.linalg.norm(mat, axis=1) + 1e-12
        sims = (mat @ qn) / norms
        ranking = list(enumerate(sims.tolist()))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking[:top_k]

    # Basic export used by migration
    def export(self) -> Dict[str, Any]:
        return {"vectors": self._vectors, "texts": self._texts, "metadata": self._meta}

__all__ = ["IntegratedVectorGraphIndex"]
