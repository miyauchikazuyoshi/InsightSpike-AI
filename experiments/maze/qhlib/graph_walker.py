"""Layer 1: Attention-weighted graph walk for candidate generation.

Scans edges from revisit nodes where attention > theta.

Scoring modes:
- legacy:  attention^α × cosine_sim × σ(propagated/τ)
- 3att:    ag_attention × σ(-dg_attention/τ_dg) × σ(propagated/τ_r)
           (Three-attention: relevance × confidence × value)

Computational cost: O(degree) — independent of total memory size.
"""

import math
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

Node = Tuple[int, int, int]

_PROPAGATED_DIM = 9


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class AttentionGraphWalker:
    """Walk graph edges filtered by attention threshold and DG quality gate."""

    def __init__(self, theta: float = 0.3, alpha: float = 0.5,
                 dg_gate_tau: float = 1.0,
                 tau_dg_3att: float = 0.3, tau_reward: float = 0.3,
                 score_mode: str = "legacy"):
        self.theta = theta           # attention threshold
        self.alpha = alpha           # attention exponent for effective_score
        self.dg_gate_tau = dg_gate_tau  # DG gate temperature (legacy)
        self.tau_dg_3att = tau_dg_3att  # DG confidence temperature (3att)
        self.tau_reward = tau_reward    # Reward value temperature (3att)
        self.score_mode = score_mode   # 'legacy' or '3att'

    def get_candidates(
        self,
        graph: nx.Graph,
        revisit_nodes: List[Tuple[Node, float]],
        query_vector: np.ndarray,
        weight_vector: np.ndarray,
    ) -> List[Dict]:
        """Extract candidates from revisit node neighbors with attention > theta.

        Both legacy and 3-attention scores are computed and returned.
        Sorting uses the score selected by score_mode.
        """
        candidates = []
        seen: set = set()
        tau = self.dg_gate_tau
        tau_d = self.tau_dg_3att
        tau_r = self.tau_reward

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

                # Propagated value (shared by both scoring modes)
                propagated = 0.0
                if len(neighbor_arr) > _PROPAGATED_DIM:
                    propagated = float(neighbor_arr[_PROPAGATED_DIM])

                # --- Legacy score ---
                dg_gate = _sigmoid(propagated / tau) if tau > 0 else (1.0 if propagated >= 0 else 0.0)
                w_sim = self._weighted_similarity(
                    query_vector, neighbor_arr, weight_vector
                )
                effective_score = (attention ** self.alpha) * w_sim * dg_gate

                # --- 3-attention score ---
                # relevance: ag_attention (historical similarity at connection time)
                ag_att = float(edge_data.get("ag_attention", attention))
                # confidence: σ(-dg_attention / τ_dg)
                #   dg_attention = g0 (negative = improvement = high confidence)
                dg_att = float(edge_data.get("dg_attention", 0.0))
                dg_conf = _sigmoid(-dg_att / tau_d) if tau_d > 0 else 0.5
                # value: σ(propagated / τ_r)
                rw_val = _sigmoid(propagated / tau_r) if tau_r > 0 else 0.5
                score_3att = ag_att * dg_conf * rw_val

                candidates.append(
                    {
                        "node_id": neighbor,
                        "attention": attention,
                        "weighted_similarity": w_sim,
                        "dg_gate": dg_gate,
                        "propagated": propagated,
                        "effective_score": effective_score,
                        "score_3att": score_3att,
                        "ag_attention": ag_att,
                        "dg_confidence": dg_conf,
                        "reward_value": rw_val,
                        "source_revisit_node": revisit_node,
                        "edge_type": edge_data.get("edge_type", "unknown"),
                    }
                )

        sort_key = "score_3att" if self.score_mode == "3att" else "effective_score"
        candidates.sort(key=lambda x: -x[sort_key])
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
