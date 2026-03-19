"""Message passing abstractions.

Two styles unified:
  - QLearningPropagator: max-neighbor (maze sleep)
  - AttentionPropagator: flow-weighted average (RAG AGHT)
"""

from __future__ import annotations

from typing import Any, Dict


class QLearningPropagator:
    """Q-learning style value propagation (maze sleep phase).

    propagated(n) = reward(n) + gamma * max(propagated(neighbor))

    Parameters
    ----------
    gamma : float
        Discount factor. Default 0.95.
    """

    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma

    def propagate(
        self,
        graph: Any,  # nx.Graph
        node_values: Dict[Any, float],
        n_iterations: int = 50,
        reward_key: str = "reward",
        **kwargs: Any,
    ) -> Dict[Any, float]:
        """Propagate values through graph using max-neighbor rule.

        Parameters
        ----------
        graph : nx.Graph
            Graph with node attributes.
        node_values : dict
            Initial node values (reward).
        n_iterations : int
            Number of propagation iterations.

        Returns
        -------
        dict : node_id → propagated value
        """
        propagated = dict(node_values)

        for _ in range(n_iterations):
            new_vals = {}
            for node in graph.nodes():
                reward = node_values.get(node, 0.0)
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    max_nbr = max(propagated.get(n, 0.0) for n in neighbors)
                else:
                    max_nbr = 0.0
                new_vals[node] = reward + self.gamma * max_nbr
            propagated = new_vals

        return propagated


class AttentionPropagator:
    """Attention-weighted message passing (RAG AGHT).

    rel(n) = (1-alpha) * rel(n) + alpha * weighted_avg(flow * rel(neighbor))

    Parameters
    ----------
    alpha : float
        Mixing weight for neighbor aggregation. Default 0.3.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def propagate(
        self,
        graph: Any,  # nx.DiGraph
        node_values: Dict[Any, float],
        n_iterations: int = 2,
        flow_key: str = "flow",
        **kwargs: Any,
    ) -> Dict[Any, float]:
        """Propagate values using attention-weighted averaging.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph with edge 'flow' attributes.
        node_values : dict
            Initial node relevance scores.
        n_iterations : int
            Number of propagation rounds.

        Returns
        -------
        dict : node_id → propagated relevance
        """
        relevance = dict(node_values)

        for _ in range(n_iterations):
            new_rel = {}
            for node in graph.nodes():
                preds = list(graph.predecessors(node))
                if not preds:
                    new_rel[node] = relevance.get(node, 0.0)
                    continue

                total_flow = 0.0
                weighted_sum = 0.0
                for pred in preds:
                    flow = graph[pred][node].get(flow_key, 0.0)
                    weighted_sum += flow * relevance.get(pred, 0.0)
                    total_flow += flow

                if total_flow > 1e-8:
                    agg = weighted_sum / total_flow
                else:
                    agg = 0.0

                new_rel[node] = (
                    (1 - self.alpha) * relevance.get(node, 0.0)
                    + self.alpha * agg
                )
            relevance = new_rel

        return relevance
