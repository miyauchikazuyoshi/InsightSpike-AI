"""RAG adapter: maps document graphs to unified geDIG F-eval.

Wraps AGHT's per-edge QKV attention and AG/DG classification
using the unified core.

Usage:
    from gedig.adapters.rag import RAGFEval

    f_eval = RAGFEval()
    # Per-edge F-eval (QKV style)
    edge_f = f_eval.compute_edge_f(cost=0.3, q_score=0.8, k_score=0.6, f_lambda=1.0)
    # AG/DG classification
    result = f_eval.classify_edges(edge_f_values, percentile=0.3)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx

from gedig.core.ag_dg import PercentileClassifier
from gedig.core.protocols import AGDGResult, FEvalResult
from gedig.core.message_passing import AttentionPropagator
from gedig.backends.networkx_backend import NxGraphSnapshot, NxBetti


class RAGFEval:
    """F-eval for RAG document graphs (AGHT style).

    In RAG, F-eval operates at the per-edge level:
      f(e) = cost(e) - λ · dot(Q, K) / √d_k

    This is different from maze/transformer which compare before/after
    graph states. Instead, each edge gets an individual f-value, and
    the AG/DG classification uses a percentile threshold.

    Parameters
    ----------
    f_lambda : float
        Weight for query relevance in f-value computation.
    percentile : float
        AG/DG classification percentile (default 0.3 = bottom 30% are AG).
    d_k : float
        QKV scaling dimension.
    """

    def __init__(
        self,
        f_lambda: float = 1.0,
        percentile: float = 0.3,
        d_k: float = 3.0,
    ):
        self.f_lambda = f_lambda
        self.classifier = PercentileClassifier(percentile=percentile)
        self.d_k = d_k
        self.propagator = AttentionPropagator()

    def compute_edge_f(
        self,
        cost: float,
        q_vec: list[float],
        k_vec: list[float],
    ) -> float:
        """Compute per-edge F-value using QKV dot product.

        f = cost - λ · dot(Q, K) / √d_k

        Parameters
        ----------
        cost : float
            Base edge cost (from edge type).
        q_vec : list[float]
            Q vector (query-dependent features of source node).
        k_vec : list[float]
            K vector (intrinsic features of target node).

        Returns
        -------
        float : f-value (lower = more AG-like)
        """
        import math
        dot = sum(q * k for q, k in zip(q_vec, k_vec))
        alpha = dot / math.sqrt(self.d_k)
        return cost - self.f_lambda * alpha

    def classify_edges(
        self,
        edge_f_values: Dict[Any, float],
    ) -> AGDGResult:
        """Classify edges as AG/DG using percentile threshold.

        Parameters
        ----------
        edge_f_values : dict
            edge_id → f_value mapping.

        Returns
        -------
        AGDGResult with ag/dg edge lists and threshold.
        """
        return self.classifier.classify(edge_f_values)

    def compute_graph_betti(self, graph: nx.Graph) -> float:
        """Compute β₁ of a document graph.

        Useful for tracking structural properties across experiments.
        """
        snap = NxGraphSnapshot(graph)
        betti = NxBetti()
        return betti._betti_1(snap)

    def propagate(
        self,
        graph: nx.DiGraph,
        initial_relevance: Dict[Any, float],
        n_iterations: int = 2,
        alpha: float = 0.3,
    ) -> Dict[Any, float]:
        """Attention-weighted message passing on document graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Document graph with edge 'flow' attributes.
        initial_relevance : dict
            node_id → initial relevance score.
        n_iterations : int
            Number of propagation rounds.
        alpha : float
            Mixing weight for neighbor aggregation.

        Returns
        -------
        dict : node_id → propagated relevance
        """
        self.propagator.alpha = alpha
        return self.propagator.propagate(
            graph, initial_relevance,
            n_iterations=n_iterations,
        )
