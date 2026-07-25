"""Score-based edge partitioning, distinct from AG/DG event gates."""

from __future__ import annotations

from typing import Any, Dict

from .protocols import EdgePartitionResult


class PercentileEdgePartitioner:
    """Partition edges at a percentile of their scalar scores."""

    def __init__(self, percentile: float = 0.3):
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("percentile must be between 0 and 1")
        self.percentile = percentile

    def partition(
        self,
        edge_scores: Dict[Any, float],
    ) -> EdgePartitionResult:
        if not edge_scores:
            return EdgePartitionResult()

        sorted_values = sorted(edge_scores.values())
        index = int(len(sorted_values) * self.percentile)
        threshold = sorted_values[min(index, len(sorted_values) - 1)]
        low_score_edges = []
        high_score_edges = []
        for edge_id, score in edge_scores.items():
            if score < threshold:
                low_score_edges.append(edge_id)
            else:
                high_score_edges.append(edge_id)

        return EdgePartitionResult(
            threshold=threshold,
            low_score_edges=low_score_edges,
            high_score_edges=high_score_edges,
        )


class ThresholdEdgePartitioner:
    """Partition edges using separate upper and lower thresholds."""

    def __init__(
        self,
        upper_threshold: float = 0.0,
        lower_threshold: float = float("inf"),
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def partition(
        self,
        edge_scores: Dict[Any, float],
    ) -> EdgePartitionResult:
        low_score_edges = []
        middle_score_edges = []
        high_score_edges = []
        for edge_id, score in edge_scores.items():
            if score > self.upper_threshold:
                high_score_edges.append(edge_id)
            elif score < self.lower_threshold:
                low_score_edges.append(edge_id)
            else:
                middle_score_edges.append(edge_id)

        return EdgePartitionResult(
            threshold=self.upper_threshold,
            low_score_edges=low_score_edges,
            high_score_edges=high_score_edges,
            middle_score_edges=middle_score_edges,
            metadata={
                "upper_threshold": self.upper_threshold,
                "lower_threshold": self.lower_threshold,
            },
        )


__all__ = [
    "PercentileEdgePartitioner",
    "ThresholdEdgePartitioner",
]
