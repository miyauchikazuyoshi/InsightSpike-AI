"""Layer 1: Attention-weighted graph walk for candidate generation.

Scans edges from revisit nodes where attention > theta.
Computational cost: O(degree) — independent of total memory size.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

Node = Tuple[int, int, int]


class AttentionGraphWalker:
    """Walk graph edges filtered by attention threshold."""

    def __init__(self, theta: float = 0.3, alpha: float = 0.5):
        self.theta = theta   # attention threshold
        self.alpha = alpha   # attention exponent for effective_score

    def get_candidates(
        self,
        graph: nx.Graph,
        revisit_nodes: List[Tuple[Node, float]],
        query_vector: np.ndarray,
        weight_vector: np.ndarray,
    ) -> List[Dict]:
        """Extract candidates from revisit node neighbors with attention > theta.

        Returns candidates sorted by effective_score (descending).
        """
        candidates = []
        seen: set = set()

        for revisit_node, raw_sim in revisit_nodes:
            if revisit_node not in graph:
                continue
            for neighbor in graph.neighbors(revisit_node):
                if neighbor in seen:
                    continue
                seen.add(neighbor)

                edge_data = graph[revisit_node][neighbor]
                attention = float(edge_data.get("attention", 0.0))
                if attention < self.theta:
                    continue

                neighbor_vec = graph.nodes[neighbor].get("abs_vector")
                if neighbor_vec is None:
                    neighbor_vec = graph.nodes[neighbor].get("vector")
                if neighbor_vec is None:
                    continue
                neighbor_arr = np.asarray(neighbor_vec, dtype=float)

                w_sim = self._weighted_similarity(
                    query_vector, neighbor_arr, weight_vector
                )
                effective_score = (attention ** self.alpha) * w_sim

                candidates.append(
                    {
                        "node_id": neighbor,
                        "attention": attention,
                        "weighted_similarity": w_sim,
                        "effective_score": effective_score,
                        "source_revisit_node": revisit_node,
                        "edge_type": edge_data.get("edge_type", "unknown"),
                    }
                )

        candidates.sort(key=lambda x: -x["effective_score"])
        return candidates

    @staticmethod
    def _weighted_similarity(q: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        n = min(len(q), len(v), len(w))
        qw = q[:n] * w[:n]
        vw = v[:n] * w[:n]
        dot = float(np.dot(qw, vw))
        nq = float(np.linalg.norm(qw))
        nv = float(np.linalg.norm(vw))
        if nq < 1e-9 or nv < 1e-9:
            return 0.0
        return max(0.0, dot / (nq * nv))
