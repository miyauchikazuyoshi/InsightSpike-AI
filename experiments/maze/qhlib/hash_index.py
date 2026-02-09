"""Layer 0: Vector hash index for O(1) revisit detection.

Quantizes raw vectors into hash buckets for approximate exact-match lookup.
For maze: resolution = 1/maze_size (one cell).
For RAG:  resolution = embedding granularity (e.g. 0.05).
"""

from itertools import product
from typing import Dict, List, Tuple

import numpy as np

Node = Tuple[int, int, int]


class VectorHashIndex:
    """Quantized hash index for fast revisit detection."""

    def __init__(self, resolution: float = 0.05):
        self.res = float(max(1e-9, resolution))
        self._buckets: Dict[tuple, List[Tuple[Node, np.ndarray]]] = {}
        self._count = 0

    def _quantize(self, vec: np.ndarray) -> tuple:
        return tuple((vec / self.res).astype(int))

    def add(self, node_id: Node, raw_vector: np.ndarray) -> None:
        """Register a node with its raw vector."""
        key = self._quantize(raw_vector)
        self._buckets.setdefault(key, []).append((node_id, raw_vector.copy()))
        self._count += 1

    def lookup(
        self,
        query_vector: np.ndarray,
        theta_revisit: float = 0.95,
    ) -> List[Tuple[Node, float]]:
        """O(1) hash lookup + raw cosine similarity check."""
        key = self._quantize(query_vector)
        candidates = self._buckets.get(key, [])
        if not candidates:
            return []

        qn = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        results = []
        for node_id, stored_vec in candidates:
            vn = stored_vec / (np.linalg.norm(stored_vec) + 1e-9)
            sim = float(np.dot(qn, vn))
            if sim >= theta_revisit:
                results.append((node_id, sim))

        results.sort(key=lambda x: -x[1])
        return results

    def lookup_with_neighbors(
        self,
        query_vector: np.ndarray,
        theta_revisit: float = 0.95,
    ) -> List[Tuple[Node, float]]:
        """Search adjacent bins too. Prevents misses at quantization boundaries."""
        key = self._quantize(query_vector)
        dim = len(key)

        all_candidates = []
        for offset in product([-1, 0, 1], repeat=dim):
            nkey = tuple(k + o for k, o in zip(key, offset))
            all_candidates.extend(self._buckets.get(nkey, []))

        if not all_candidates:
            return []

        qn = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        seen: set = set()
        results = []
        for node_id, stored_vec in all_candidates:
            if node_id in seen:
                continue
            seen.add(node_id)
            vn = stored_vec / (np.linalg.norm(stored_vec) + 1e-9)
            sim = float(np.dot(qn, vn))
            if sim >= theta_revisit:
                results.append((node_id, sim))

        results.sort(key=lambda x: -x[1])
        return results

    @property
    def size(self) -> int:
        return self._count
