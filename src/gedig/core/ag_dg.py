"""AG/DG edge classification.

AG (Assertion Graph): confirmed edges, low uncertainty
DG (Differential Graph): uncertain edges, information gaps

Two classifiers:
  - PercentileClassifier: dynamic threshold from data (RAG, transformer)
  - ThresholdClassifier: fixed threshold (maze)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .protocols import AGDGResult


class PercentileClassifier:
    """Classify edges as AG/DG using a percentile-based threshold.

    Used by RAG (30th percentile on f-values) and transformer.
    Edges below the threshold are AG (confirmed), above are DG (uncertain).

    Parameters
    ----------
    percentile : float
        Fraction of edges classified as AG. Default 0.3 (30th percentile).
    """

    def __init__(self, percentile: float = 0.3):
        self.percentile = percentile

    def classify(
        self,
        edge_scores: Dict[Any, float],
    ) -> AGDGResult:
        """Classify edges based on percentile threshold.

        Parameters
        ----------
        edge_scores : dict
            Mapping of edge_id → f_value (lower = more AG-like).

        Returns
        -------
        AGDGResult with ag_edges, dg_edges, and threshold.
        """
        if not edge_scores:
            return AGDGResult()

        sorted_vals = sorted(edge_scores.values())
        idx = int(len(sorted_vals) * self.percentile)
        threshold = sorted_vals[min(idx, len(sorted_vals) - 1)]

        ag_edges = []
        dg_edges = []
        for edge_id, fv in edge_scores.items():
            if fv < threshold:
                ag_edges.append(edge_id)
            else:
                dg_edges.append(edge_id)

        return AGDGResult(
            n_ag=len(ag_edges),
            n_dg=len(dg_edges),
            threshold=threshold,
            ag_edges=ag_edges,
            dg_edges=dg_edges,
        )


class ThresholdClassifier:
    """Classify edges as AG/DG using fixed thresholds.

    Used by maze: ag_fire = g0 > theta_AG, dg_fire = gmin < theta_DG.

    Parameters
    ----------
    theta_ag : float
        AG fire threshold. Scores above this are AG.
    theta_dg : float
        DG fire threshold. Scores below this are DG.
    """

    def __init__(
        self,
        theta_ag: float = 0.0,
        theta_dg: float = float("inf"),
    ):
        self.theta_ag = theta_ag
        self.theta_dg = theta_dg

    def classify(
        self,
        edge_scores: Dict[Any, float],
    ) -> AGDGResult:
        """Classify edges based on fixed thresholds.

        For maze: scores are g-values (higher = better structural gain).
        AG: g > theta_ag (high quality → commit)
        DG: g < theta_dg (low quality → explore further)
        """
        ag_edges = []
        dg_edges = []
        for edge_id, g in edge_scores.items():
            if g > self.theta_ag:
                ag_edges.append(edge_id)
            elif g < self.theta_dg:
                dg_edges.append(edge_id)

        return AGDGResult(
            n_ag=len(ag_edges),
            n_dg=len(dg_edges),
            threshold=self.theta_ag,
            ag_edges=ag_edges,
            dg_edges=dg_edges,
            metadata={"theta_ag": self.theta_ag, "theta_dg": self.theta_dg},
        )
