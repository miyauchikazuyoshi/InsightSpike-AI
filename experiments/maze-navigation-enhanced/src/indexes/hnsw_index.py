"""HNSWLib ANN index implementation (Phase 6).

Optional dependency: hnswlib
Usage:
    from indexes.hnsw_index import HNSWLibIndex
    index = HNSWLibIndex(dim=128)
    index.add([1,2,3], np.random.randn(3,128).astype('float32'))
    results = index.search(query_vec, top_k=5)
Falls back gracefully if hnswlib is not installed (ImportError on construction).
"""
from __future__ import annotations
from typing import Sequence, List, Tuple, Optional
import numpy as np

try:
    import hnswlib  # type: ignore
except Exception as e:  # pragma: no cover - dependency might be absent
    hnswlib = None  # type: ignore

class HNSWLibIndex:
    """Thin wrapper around hnswlib.Index implementing VectorIndex protocol.

    Notes:
        - Uses L2 space (consistent with current linear reference).
        - Dynamically resizes when adding more elements than current capacity.
        - remove() implemented via mark_deleted (vectors are not reclaimed until rebuild).
    """
    def __init__(self, dim: int, max_elements: int = 10000, M: int = 16, ef_construction: int = 100, ef_search: int = 100):
        if hnswlib is None:  # pragma: no cover
            raise ImportError("hnswlib not installed. Install with `pip install hnswlib` or extras 'ann'.")
        self._dim = dim
        self._index = hnswlib.Index(space='l2', dim=dim)
        self._index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
        self._index.set_ef(ef_search)
        self._count = 0
        self._max_elements = max_elements
    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        if not len(ids):
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"Vector shape mismatch: expected (*,{self._dim}) got {vectors.shape}")
        needed = self._count + len(ids)
        if needed > self._max_elements:
            # resize (round up by 50%)
            new_cap = max(needed, int(self._max_elements * 1.5))
            self._index.resize_index(new_cap)
            self._max_elements = new_cap
        self._index.add_items(vectors, ids)
        self._count += len(ids)
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if top_k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(f"Query dimension mismatch {q.shape[1]} != {self._dim}")
        labels, distances = self._index.knn_query(q, k=min(top_k, self._count))
        return [(int(l), float(d)) for l, d in zip(labels[0], distances[0])]
    def remove(self, ids: Sequence[int]) -> None:
        for vid in ids:
            try:
                self._index.mark_deleted(int(vid))
                self._count -= 1
            except Exception:
                pass
    def __len__(self) -> int:
        return self._count
    def set_ef(self, ef: int) -> None:
        try:
            self._index.set_ef(ef)
        except Exception:
            pass

__all__ = ["HNSWLibIndex"]
