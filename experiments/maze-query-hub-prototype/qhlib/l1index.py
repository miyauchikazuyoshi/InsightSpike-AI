from __future__ import annotations

"""
Simple weighted-L2 index for Layer1-style vector prefiltering.

Stores pre-weighted vectors (element-wise multiply by provided weights) and
performs plain L2 distance for search, which is equivalent to weighted L2
in the original space.

This is a naive linear-scan top-K implementation suitable for prototyping.
"""

from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np

Node = Tuple[int, int, int]


class WeightedL2Index:
    def __init__(self, dim: int, weights: Sequence[float]):
        self._dim = int(dim)
        self._w = np.asarray(weights, dtype=float).reshape(-1)
        if self._w.shape[0] != self._dim:
            raise ValueError("weights dim mismatch")
        self._vecs: Dict[Node, np.ndarray] = {}

    def _pre(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.shape[0] != self._dim:
            raise ValueError("vector dim mismatch")
        return self._w * v

    def add(self, node_id: Node, abs_vector: np.ndarray) -> None:
        self._vecs[node_id] = self._pre(abs_vector)

    def search(self, query_abs_vector: np.ndarray, top_k: int) -> List[Tuple[Node, float]]:
        if top_k <= 0 or not self._vecs:
            return []
        q = self._pre(query_abs_vector)
        dists: List[Tuple[Node, float]] = []
        for nid, v in self._vecs.items():
            try:
                d = float(np.linalg.norm(q - v))
            except Exception:
                continue
            dists.append((nid, d))
        dists.sort(key=lambda x: x[1])
        if len(dists) > top_k:
            dists = dists[:top_k]
        return dists

