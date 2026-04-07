"""Maze adapter: maps spatial graphs to unified geDIG F-eval.

Wraps the maze experiment's hop-series evaluation pattern:
  for h in 0..max_hops:
    g(h) = FEval.compute(graph_before, graph_after_h)
    if g(h) meets AG/DG threshold → commit

Usage:
    from gedig.adapters.maze import MazeFEval

    f_eval = MazeFEval(lambda_param=1.0, gamma=0.5)
    result = f_eval.evaluate_hop(before_graph, after_graph)
    # result.f_value → g-value for this hop
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx

from gedig.core.f_eval import FEval
from gedig.core.ag_dg import ThresholdClassifier
from gedig.core.protocols import FEvalResult, AGDGResult
from gedig.core.message_passing import QLearningPropagator
from gedig.backends.networkx_backend import (
    NxGraphSnapshot,
    NxEPC,
    NxEntropy,
    NxSP,
    NxBetti,
)


class MazeFEval:
    """F-eval for maze spatial graphs.

    Wraps the hop-series evaluation: computes F for each candidate
    edge addition at distance h from the agent.

    Parameters
    ----------
    lambda_param : float
        Weight for (H + γ·B) term.
    gamma : float
        Weight for B relative to H.
    use_betti : bool
        If True, use β₁ instead of SP.
    theta_ag : float
        AG fire threshold (g > theta → commit).
    theta_dg : float
        DG fire threshold (g < theta → explore further).
    """

    def __init__(
        self,
        lambda_param: float = 1.0,
        gamma: float = 0.5,
        use_betti: bool = False,
        theta_ag: float = 0.0,
        theta_dg: float = float("inf"),
    ):
        sp = NxBetti() if use_betti else NxSP()
        self.f_eval = FEval(
            epc=NxEPC(),
            entropy=NxEntropy(),
            sp=sp,
            lambda_param=lambda_param,
            gamma=gamma,
        )
        self.classifier = ThresholdClassifier(
            theta_ag=theta_ag,
            theta_dg=theta_dg,
        )
        self.propagator = QLearningPropagator()

    def evaluate_hop(
        self,
        before_graph: nx.Graph,
        after_graph: nx.Graph,
    ) -> FEvalResult:
        """Compute F between before and after adding hop-h edges.

        Parameters
        ----------
        before_graph : nx.Graph
            Graph state before edge addition.
        after_graph : nx.Graph
            Graph state after adding candidate edges at hop h.

        Returns
        -------
        FEvalResult with f_value (= g-value in maze terminology).
        """
        before = NxGraphSnapshot(before_graph)
        after = NxGraphSnapshot(after_graph)
        return self.f_eval.compute(before, after)

    def evaluate_multihop(
        self,
        base_graph: nx.Graph,
        hop_graphs: list[nx.Graph],
    ) -> list[FEvalResult]:
        """Evaluate F across all hops in a hop-series.

        Parameters
        ----------
        base_graph : nx.Graph
            Graph state before any candidate edges.
        hop_graphs : list[nx.Graph]
            Graph states after adding edges at each hop distance.

        Returns
        -------
        list[FEvalResult] : F-eval result for each hop.
        """
        before = NxGraphSnapshot(base_graph)
        results = []
        for hop_graph in hop_graphs:
            after = NxGraphSnapshot(hop_graph)
            results.append(self.f_eval.compute(before, after))
        return results

    def classify_step(
        self,
        g0: float,
        gmin_mh: float,
        best_hop: int,
    ) -> Dict[str, Any]:
        """Classify a maze step as AG fire, DG fire, or neither.

        Parameters
        ----------
        g0 : float
            Hop-0 g-value (immediate connection quality).
        gmin_mh : float
            Min g-value across multi-hop evaluation.
        best_hop : int
            Best hop index (0 = no multi-hop benefit).

        Returns
        -------
        dict with ag_fire, dg_fire, best_hop.
        """
        ag_fire = g0 > self.classifier.theta_ag
        dg_fire = best_hop >= 1 and gmin_mh < self.classifier.theta_dg

        return {
            "ag_fire": ag_fire,
            "dg_fire": dg_fire,
            "best_hop": best_hop,
            "g0": g0,
            "gmin_mh": gmin_mh,
        }

    def sleep_propagate(
        self,
        graph: nx.Graph,
        rewards: Dict[Any, float],
        gamma: float = 0.95,
        n_iterations: int = 50,
    ) -> Dict[Any, float]:
        """Sleep phase: Q-learning style reward propagation.

        propagated(n) = reward(n) + γ·max(propagated(neighbor))

        Parameters
        ----------
        graph : nx.Graph
            Maze graph.
        rewards : dict
            node → reward value.
        gamma : float
            Discount factor.
        n_iterations : int
            Propagation iterations.

        Returns
        -------
        dict : node → propagated value.
        """
        self.propagator.gamma = gamma
        return self.propagator.propagate(
            graph, rewards,
            n_iterations=n_iterations,
        )
