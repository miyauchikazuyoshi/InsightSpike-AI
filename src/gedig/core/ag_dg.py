"""Compatibility shims for historical AG/DG edge-classifier names.

The old API used AG/DG names for score-based edge partitions. AG and DG now
refer only to the Attention Gate and Decision Gate event pair; new edge code
uses :mod:`gedig.core.edge_partition`.
"""

from __future__ import annotations

from typing import Any, Dict

from .edge_partition import (
    PercentileEdgePartitioner,
    ThresholdEdgePartitioner,
)
from .protocols import AGDGResult, EdgePartitionResult


def _legacy_result(partition: EdgePartitionResult) -> AGDGResult:
    return AGDGResult(
        n_ag=partition.n_low_score,
        n_dg=partition.n_high_score,
        threshold=partition.threshold,
        ag_edges=partition.low_score_edges,
        dg_edges=partition.high_score_edges,
        metadata={
            **partition.metadata,
            "legacy_edge_labels": True,
        },
    )


class PercentileClassifier:
    """Deprecated wrapper around :class:`PercentileEdgePartitioner`."""

    def __init__(self, percentile: float = 0.3):
        self.percentile = percentile

    def classify(
        self,
        edge_scores: Dict[Any, float],
    ) -> AGDGResult:
        """Return the historical field layout for a score partition."""

        partitioner = PercentileEdgePartitioner(self.percentile)
        return _legacy_result(partitioner.partition(edge_scores))


class ThresholdClassifier:
    """Deprecated edge-partition wrapper with historical parameter names."""

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
        """Return the historical field layout for a score partition."""

        partition = ThresholdEdgePartitioner(
            upper_threshold=self.theta_ag,
            lower_threshold=self.theta_dg,
        ).partition(edge_scores)
        return AGDGResult(
            n_ag=partition.n_high_score,
            n_dg=partition.n_low_score,
            threshold=partition.threshold,
            ag_edges=partition.high_score_edges,
            dg_edges=partition.low_score_edges,
            metadata={
                **partition.metadata,
                "legacy_edge_labels": True,
                "theta_ag": self.theta_ag,
                "theta_dg": self.theta_dg,
            },
        )
