"""NetworkX backend for F-eval components.

Used by maze and RAG experiments. Operates on nx.Graph / nx.DiGraph.

Extracts and unifies implementations from:
  - gedig_scoring.py: _local_ged(), _local_entropy(), _local_sp_gain()
  - algorithms/core/metrics.py: normalized_ged(), entropy_ig()
"""

from __future__ import annotations

import math
from typing import Any, Optional, Set

import networkx as nx
import numpy as np


# ─── Graph Snapshot ──────────────────────────────────────────────

class NxGraphSnapshot:
    """NetworkX graph wrapped as a GraphSnapshot.

    Works with both nx.Graph and nx.DiGraph.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._edge_set: Optional[Set] = None

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def edge_set(self) -> Set:
        if self._edge_set is None:
            self._edge_set = set(self.graph.edges())
        return self._edge_set


# ─── EPC (Graph Edit Cost) ───────────────────────────────────────

class NxEPC:
    """Graph Edit Cost via edge set symmetric difference.

    ΔEPC = |E_added ∪ E_removed| / max(|E_before|, |E_after|, 1)

    Unifies:
      - gedig_scoring.py _local_ged()
      - algorithms/core/metrics.py normalized_ged()
    """

    def compute(self, before: NxGraphSnapshot, after: NxGraphSnapshot) -> float:
        e_before = before.edge_set()
        e_after = after.edge_set()

        added = e_after - e_before
        removed = e_before - e_after

        denominator = max(len(e_before), len(e_after), 1)
        return (len(added) + len(removed)) / denominator


# ─── Entropy ─────────────────────────────────────────────────────

class NxEntropy:
    """Shannon entropy change of node feature distributions.

    ΔH = H(after) - H(before)

    Computes entropy from node degree distribution by default.
    Can use custom feature_key for feature-based entropy.

    Unifies:
      - gedig_scoring.py _local_entropy()
      - algorithms/core/metrics.py entropy_ig()
    """

    def __init__(self, feature_key: Optional[str] = None):
        self.feature_key = feature_key

    def compute(self, before: NxGraphSnapshot, after: NxGraphSnapshot) -> float:
        h_before = self._entropy(before)
        h_after = self._entropy(after)
        return h_after - h_before

    def _entropy(self, snapshot: NxGraphSnapshot) -> float:
        """Compute Shannon entropy of the degree distribution."""
        g = snapshot.graph
        if g.number_of_nodes() == 0:
            return 0.0

        if self.feature_key:
            # Feature-based entropy
            vals = [
                g.nodes[n].get(self.feature_key, 0.0)
                for n in g.nodes()
            ]
        else:
            # Degree-based entropy
            vals = [float(d) for _, d in g.degree()]

        total = sum(vals) or 1.0
        probs = [v / total for v in vals if v > 0]

        if not probs:
            return 0.0

        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(max(len(probs), 1))

        return entropy / max_entropy if max_entropy > 0 else 0.0


# ─── Structure Potential (SP / Betti) ────────────────────────────

class NxSP:
    """Shortest-path efficiency change.

    ΔSP = SP(after) - SP(before)

    Uses sampled pairs for scalability.

    Unifies:
      - gedig_scoring.py _local_sp_gain()
      - algorithms/core/metrics.py compute_sp_gain_norm()
    """

    def __init__(self, n_pairs: int = 20, seed: int = 42):
        self.n_pairs = n_pairs
        self.seed = seed

    def compute(self, before: NxGraphSnapshot, after: NxGraphSnapshot) -> float:
        sp_before = self._avg_efficiency(before)
        sp_after = self._avg_efficiency(after)
        return sp_after - sp_before

    def _avg_efficiency(self, snapshot: NxGraphSnapshot) -> float:
        """Average path efficiency over sampled node pairs."""
        g = snapshot.graph
        nodes = list(g.nodes())
        n = len(nodes)
        if n < 2:
            return 0.0

        rng = np.random.RandomState(self.seed)
        pairs = []
        for _ in range(min(self.n_pairs, n * (n - 1) // 2)):
            i, j = rng.choice(n, size=2, replace=False)
            pairs.append((nodes[i], nodes[j]))

        total_eff = 0.0
        for u, v in pairs:
            try:
                d = nx.shortest_path_length(g, u, v)
                total_eff += 1.0 / d
            except nx.NetworkXNoPath:
                pass  # disconnected → efficiency 0

        return total_eff / max(len(pairs), 1)


class NxBetti:
    """First Betti number (β₁ = E - V + C) change.

    β₁ measures the number of independent cycles in the graph.
    SP's essence is "structural short-circuiting via holes" → β₁.

    ΔB = β₁(after) - β₁(before)
    """

    def compute(self, before: NxGraphSnapshot, after: NxGraphSnapshot) -> float:
        b_before = self._betti_1(before)
        b_after = self._betti_1(after)
        return b_after - b_before

    def _betti_1(self, snapshot: NxGraphSnapshot) -> float:
        """Compute β₁ = E - V + C (normalized by max edges)."""
        g = snapshot.graph
        if isinstance(g, nx.DiGraph):
            ug = g.to_undirected()
        else:
            ug = g

        e = ug.number_of_edges()
        v = ug.number_of_nodes()
        c = nx.number_connected_components(ug)

        if v == 0:
            return 0.0

        beta_1 = e - v + c
        max_edges = v * (v - 1) // 2 or 1
        return beta_1 / max_edges
