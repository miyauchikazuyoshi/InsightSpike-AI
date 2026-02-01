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
    def __init__(self, dim: int, weights: Sequence[float], bin_width: float = 0.05):
        self._dim = int(dim)
        self._w = np.asarray(weights, dtype=float).reshape(-1)
        if self._w.shape[0] != self._dim:
            raise ValueError("weights dim mismatch")
        # Preweighted vectors
        self._vecs: Dict[Node, np.ndarray] = {}
        # Precomputed norms of preweighted vectors
        self._norms: Dict[Node, float] = {}
        # Simple norm bins for radius filtering (triangle inequality)
        self._bins: Dict[int, List[Node]] = {}
        self._bin_w = float(max(1e-9, bin_width))

    def _pre(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.shape[0] != self._dim:
            raise ValueError("vector dim mismatch")
        return self._w * v

    def add(self, node_id: Node, abs_vector: np.ndarray) -> None:
        v = self._pre(abs_vector)
        self._vecs[node_id] = v
        n = float(np.linalg.norm(v))
        self._norms[node_id] = n
        b = int(n / self._bin_w)
        self._bins.setdefault(b, []).append(node_id)

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

    def search_radius(self, query_abs_vector: np.ndarray, radius: float, top_k: int) -> List[Tuple[Node, float]]:
        """Return up to top_k nodes within weighted L2 radius around query.

        Uses norm bins and triangle inequality: ||q-v|| >= | ||q|| - ||v|| |.
        """
        if top_k <= 0 or not self._vecs:
            return []
        r = float(max(0.0, radius))
        q = self._pre(query_abs_vector)
        qn = float(np.linalg.norm(q))
        # Compute candidate bins
        bmin = int(max(0.0, (qn - r)) / self._bin_w)
        bmax = int((qn + r) / self._bin_w)
        cand_nodes: List[Node] = []
        for b in range(bmin, bmax + 1):
            cand_nodes.extend(self._bins.get(b, []))
        # Fallback to full scan if bin range yields nothing
        if not cand_nodes:
            return self.search(query_abs_vector, top_k)
        pairs: List[Tuple[Node, float]] = []
        for nid in cand_nodes:
            v = self._vecs.get(nid)
            if v is None:
                continue
            # Triangle inequality quick check (tighten filter):
            vn = self._norms.get(nid, 0.0)
            if abs(qn - vn) > r:
                continue
            try:
                d = float(np.linalg.norm(q - v))
            except Exception:
                continue
            if d <= r:
                pairs.append((nid, d))
        pairs.sort(key=lambda x: x[1])
        return pairs[:top_k] if len(pairs) > top_k else pairs
