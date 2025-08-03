"""
Graph Analyzer
==============

Analyzes graph structures for metrics and insights.
Separated from L3GraphReasoner to follow Single Responsibility Principle.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """Analyzes graph structures and calculates metrics."""

    def __init__(self, config=None):
        self.config = config or {}

    def calculate_metrics(
        self,
        current_graph: Data,  # PyTorch Geometric Data only
        previous_graph: Optional[Data],
        delta_ged_func,
        delta_ig_func,
    ) -> Dict[str, float]:
        """Calculate ΔGED and ΔIG metrics between PyG graphs."""
        if previous_graph is None:
            return {
                "delta_ged": 0.0,
                "delta_ig": 0.0,
                "graph_size_current": current_graph.num_nodes if current_graph else 0,
                "graph_size_previous": 0,
            }

        try:
            # Prepare config for multihop if available
            kwargs = {}
            if self.config and 'graph' in self.config:
                graph_config = self.config['graph']
                if 'metrics' in graph_config:
                    kwargs['config'] = {'metrics': graph_config['metrics']}
            
            # Calculate metrics directly on PyG Data objects
            ged = delta_ged_func(previous_graph, current_graph, **kwargs)
            
            # Calculate information gain using node features
            if hasattr(previous_graph, "x") and hasattr(current_graph, "x"):
                ig = delta_ig_func(previous_graph, current_graph, **kwargs)
            else:
                ig = 0.0

            return {
                "delta_ged": float(ged),
                "delta_ig": float(ig),
                "graph_size_current": current_graph.num_nodes,
                "graph_size_previous": previous_graph.num_nodes,
            }

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {
                "delta_ged": 0.0,
                "delta_ig": 0.0,
                "graph_size_current": current_graph.num_nodes if current_graph else 0,
                "graph_size_previous": previous_graph.num_nodes
                if previous_graph
                else 0,
            }

    def detect_spike(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> bool:
        """Detect if current state represents an insight spike."""
        # GED is negative for improvement, so check if it's below (more negative than) threshold
        high_ged = metrics.get("delta_ged", 0) < thresholds.get("ged", -0.5)
        high_ig = metrics.get("delta_ig", 0) > thresholds.get("ig", 0.2)
        low_conflict = conflicts.get("total", 1.0) < thresholds.get("conflict", 0.5)

        return high_ged and high_ig and low_conflict

    def assess_quality(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> float:
        """Assess overall quality of reasoning process."""
        # Combine multiple factors
        # GED is negative for improvement, so we negate it to get positive score
        ged_score = abs(metrics.get("delta_ged", 0))
        ig_score = metrics.get("delta_ig", 0)

        # Average of both scores
        metric_score = (ged_score + ig_score) / 2
        conflict_penalty = conflicts.get("total", 0)

        quality = max(0.0, min(1.0, metric_score - conflict_penalty))
        return float(quality)
