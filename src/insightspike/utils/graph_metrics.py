"""
Graph Metrics - ΔGED and ΔIG Calculation Utilities
================================================

Implements graph edit distance and information gain metrics for insight detection.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    from torch_geometric.data import Data

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch/PyG not available, using fallback implementations")

__all__ = ["delta_ged", "delta_ig", "GraphMetricsCalculator"]


class GraphMetricsCalculator:
    """Calculator for graph-based insight metrics."""

    def __init__(self, config=None):
        from ..config import get_config

        self.config = config or get_config()

    def calculate_delta_ged(self, graph1: Any, graph2: Any) -> float:
        """Calculate change in graph edit distance."""
        if not TORCH_AVAILABLE:
            return self._fallback_ged(graph1, graph2)

        try:
            if graph1 is None or graph2 is None:
                return 0.0

            # Simple approximation: difference in structural properties
            ged1 = self._graph_complexity(graph1)
            ged2 = self._graph_complexity(graph2)

            return abs(ged2 - ged1)

        except Exception as e:
            logging.getLogger(__name__).error(f"ΔGED calculation failed: {e}")
            return 0.0

    def calculate_delta_ig(self, graph1: Any, graph2: Any) -> float:
        """Calculate change in information gain."""
        if not TORCH_AVAILABLE:
            return self._fallback_ig(graph1, graph2)

        try:
            if graph1 is None or graph2 is None:
                return 0.0

            # Information content approximation
            ig1 = self._information_content(graph1)
            ig2 = self._information_content(graph2)

            return max(0.0, ig2 - ig1)  # Only positive information gain

        except Exception as e:
            logging.getLogger(__name__).error(f"ΔIG calculation failed: {e}")
            return 0.0

    def _graph_complexity(self, graph: Any) -> float:
        """Calculate graph structural complexity."""
        if not hasattr(graph, "num_nodes") or graph.num_nodes == 0:
            return 0.0

        num_nodes = graph.num_nodes
        num_edges = graph.edge_index.size(1) // 2 if hasattr(graph, "edge_index") else 0

        # Complexity based on nodes, edges, and density
        density = num_edges / max(1, num_nodes * (num_nodes - 1) / 2)
        complexity = np.log(max(1, num_nodes)) + np.log(max(1, num_edges)) + density

        return float(complexity)

    def _information_content(self, graph: Any) -> float:
        """Calculate information content of graph."""
        if not hasattr(graph, "x") or graph.x.size(0) == 0:
            return 0.0

        try:
            # Use feature entropy as information measure
            features = graph.x.cpu().numpy()

            # Calculate entropy of feature distributions
            total_entropy = 0.0
            for i in range(features.shape[1]):
                feature_col = features[:, i]
                # Discretize continuous features
                hist, _ = np.histogram(feature_col, bins=10)
                probs = hist / np.sum(hist + 1e-10)
                probs = probs[probs > 0]  # Remove zero probabilities

                if len(probs) > 0:
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    total_entropy += entropy

            return float(total_entropy)

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Information content calculation failed: {e}"
            )
            return 0.0

    def _fallback_ged(self, graph1: Any, graph2: Any) -> float:
        """Fallback GED calculation without PyTorch."""
        # Simple structural difference
        if graph1 is None or graph2 is None:
            return 0.0

        # Use basic properties if available
        try:
            size1 = getattr(
                graph1, "num_nodes", len(graph1) if hasattr(graph1, "__len__") else 1
            )
            size2 = getattr(
                graph2, "num_nodes", len(graph2) if hasattr(graph2, "__len__") else 1
            )

            return abs(size2 - size1) / max(size1, size2, 1)

        except:
            return 0.5  # Default moderate difference

    def _fallback_ig(self, graph1: Any, graph2: Any) -> float:
        """Fallback IG calculation without PyTorch."""
        # Simple size-based information gain
        if graph1 is None:
            return 1.0 if graph2 is not None else 0.0
        if graph2 is None:
            return 0.0

        try:
            size1 = getattr(
                graph1, "num_nodes", len(graph1) if hasattr(graph1, "__len__") else 1
            )
            size2 = getattr(
                graph2, "num_nodes", len(graph2) if hasattr(graph2, "__len__") else 1
            )

            # Information gain proportional to size increase
            gain = max(0.0, (size2 - size1) / max(size1, 1))
            return min(1.0, gain)  # Cap at 1.0

        except:
            return 0.3  # Default moderate gain


# Global calculator instance
_calculator = None


def get_calculator():
    """Get global metrics calculator."""
    global _calculator
    if _calculator is None:
        _calculator = GraphMetricsCalculator()
    return _calculator


def delta_ged(graph1: Any, graph2: Any) -> float:
    """Calculate ΔGED between two graphs."""
    calculator = get_calculator()
    return calculator.calculate_delta_ged(graph1, graph2)


def delta_ig(graph1: Any, graph2: Any) -> float:
    """Calculate ΔIG between two graphs."""
    calculator = get_calculator()
    return calculator.calculate_delta_ig(graph1, graph2)
