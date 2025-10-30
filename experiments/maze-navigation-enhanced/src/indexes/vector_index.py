"""Vector index abstraction (Phase 4 scaffold).

Design goals:
- Provide a minimal interface for adding/searching episode embeddings.
- Initial implementation is a linear scan (InMemoryIndex) producing identical
  results to existing heap-based top-K (used only if injected; otherwise
  Navigator keeps current logic for backward compatibility).
- Future: ANN backends (Faiss / hnswlib) will implement the same interface.
"""
from __future__ import annotations
from typing import Protocol, List, Tuple, Iterable, Optional, Sequence, Any
import numpy as np

class VectorIndex(Protocol):
    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        """Add vectors with associated integer ids (no duplicates expected)."""
        ...
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Return list of (id, distance) sorted by ascending distance."""
        ...
    def remove(self, ids: Sequence[int]) -> None:
        """Optional: remove vectors by id (best-effort; silent if not present)."""
        ...
    def __len__(self) -> int: ...

class InMemoryIndex:
    """Naive in-memory L2 index (linear scan).

    Stores (id -> vector) and performs np.linalg.norm over all vectors.
    Suitable for correctness parity tests before swapping in ANN.
    """
    def __init__(self, dim: Optional[int] = None):
        self._dim = dim
        self._vectors: dict[int, np.ndarray] = {}
    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        if self._dim is None and vectors.size:
            self._dim = vectors.shape[1]
        for i, vid in enumerate(ids):
            self._vectors[int(vid)] = vectors[i]
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if top_k <= 0:
            return []
        results: List[Tuple[int,float]] = []
        for vid, vec in self._vectors.items():
            # defensive: ensure dimension match
            if self._dim is not None and vec.shape[0] != self._dim:
                continue
            dist = float(np.linalg.norm(query - vec))
            results.append((vid, dist))
        results.sort(key=lambda x: x[1])
        if len(results) > top_k:
            return results[:top_k]
        return results
    def remove(self, ids: Sequence[int]) -> None:
        for vid in ids:
            self._vectors.pop(int(vid), None)
    def __len__(self) -> int:
        return len(self._vectors)
    def clear(self) -> None:
        self._vectors.clear()

__all__ = ["VectorIndex", "InMemoryIndex"]
