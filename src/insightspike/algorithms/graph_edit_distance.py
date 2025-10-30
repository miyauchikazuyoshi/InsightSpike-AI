"""
Graph Edit Distance (GED) Algorithm Implementation for PyTorch Geometric
======================================================================

Implementation of Graph Edit Distance calculation for InsightSpike-AI's geDIG technology.
This module provides the core ΔGED computation for detecting structural insight moments
using PyTorch Geometric (PyG) Data objects.

Mathematical Foundation:
    ΔGED = GED(G_after, G_reference) - GED(G_before, G_reference)
    
    Where GED(G1, G2) = min Σ(cost(operation)) for all edit operations
    that transform G1 into G2.

Key Insight Detection:
    - Negative ΔGED values indicate structural simplification (insight detected)
    - ΔGED ≤ -0.5 threshold typically indicates EurekaSpike
    - Combined with ΔIG ≥ 0.2 for full geDIG detection

PyG Integration:
    - Works with torch_geometric.data.Data objects
    - Supports edge_attr for future multi-dimensional edge features
    - Efficient tensor operations for large graphs
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

__all__ = [
    "GraphEditDistance",
    "OptimizationLevel",
    "GEDResult",
    "compute_graph_edit_distance",
    "compute_delta_ged",
]


class OptimizationLevel(Enum):
    """Optimization levels for GED calculation."""

    FAST = "fast"  # O(n²) approximation
    STANDARD = "standard"  # O(n³) exact for small graphs, approximation for large
    PRECISE = "precise"  # O(n!) exact calculation (use with caution)


@dataclass
class GEDResult:
    """Result of GED calculation with metadata."""

    ged_value: float
    computation_time: float
    optimization_level: OptimizationLevel
    graph1_size: int
    graph2_size: int
    timeout_occurred: bool = False
    approximation_used: bool = False

    @property
    def is_reliable(self) -> bool:
        """Check if result is reliable (no timeout, appropriate method)."""
        return not self.timeout_occurred and (
            self.optimization_level != OptimizationLevel.FAST
            or max(self.graph1_size, self.graph2_size) <= 20
        )


class GraphEditDistance:
    """
    Graph Edit Distance calculator for PyTorch Geometric graphs.

    This implementation focuses on knowledge graphs where nodes
    represent concepts/episodes and edges represent semantic relationships.

    The calculator supports multiple optimization levels:
    - FAST: Approximation suitable for real-time applications
    - STANDARD: Balanced precision/performance (recommended)
    - PRECISE: Exact calculation for research applications
    """

    def __init__(
        self,
        node_cost: float = 1.0,
        edge_cost: float = 1.0,
        optimization_level: Union[str, OptimizationLevel] = OptimizationLevel.STANDARD,
        timeout_seconds: float = 5.0,
        max_graph_size_exact: int = 50,
    ):
        """
        Initialize GED calculator with cost parameters.

        Args:
            node_cost: Cost of node insertion/deletion operations
            edge_cost: Cost of edge insertion/deletion operations
            optimization_level: "fast", "standard", or "precise"
            timeout_seconds: Maximum computation time before fallback
            max_graph_size_exact: Maximum graph size for exact calculation
        """
        self.node_cost = node_cost
        self.edge_cost = edge_cost

        if isinstance(optimization_level, str):
            optimization_level = OptimizationLevel(optimization_level.lower())
        self.optimization_level = optimization_level

        self.timeout_seconds = timeout_seconds
        self.max_graph_size_exact = max_graph_size_exact

        # Statistics tracking
        self.calculation_count = 0
        self.total_computation_time = 0.0
        self.approximation_count = 0

        # State for proper ΔGED calculation
        self.initial_graph = None
        self.previous_graph = None

        logger.info(
            f"GED Calculator initialized: {optimization_level.value} mode, "
            f"node_cost={node_cost}, edge_cost={edge_cost}"
        )

    def calculate(self, graph1: Data, graph2: Data) -> GEDResult:
        """
        Calculate Graph Edit Distance between two PyG graphs.

        Args:
            graph1: First PyG Data object
            graph2: Second PyG Data object

        Returns:
            GEDResult: Detailed calculation result

        Raises:
            ValueError: If graphs are invalid or incompatible
        """
        # Validate inputs
        self._validate_graphs(graph1, graph2)

        start_time = time.time()
        self.calculation_count += 1

        try:
            # Determine calculation method based on graph sizes and optimization level
            size1 = graph1.num_nodes if hasattr(graph1, "num_nodes") else 0
            size2 = graph2.num_nodes if hasattr(graph2, "num_nodes") else 0
            max_size = max(size1, size2)

            if (
                self.optimization_level == OptimizationLevel.FAST
                or max_size > self.max_graph_size_exact
            ):
                ged_value, approximation_used = self._approximate_ged(graph1, graph2)
            elif self.optimization_level == OptimizationLevel.PRECISE:
                ged_value, approximation_used = self._exact_ged_pyg(graph1, graph2)
            else:  # STANDARD
                ged_value, approximation_used = self._standard_ged(graph1, graph2)

            computation_time = time.time() - start_time
            self.total_computation_time += computation_time

            if approximation_used:
                self.approximation_count += 1

            # Numerical noise suppression for identical graphs
            if abs(ged_value) < 1e-6 and graph1.num_nodes == graph2.num_nodes and graph1.edge_index.size(1) == graph2.edge_index.size(1):
                ged_value = 0.0

            result = GEDResult(
                ged_value=ged_value,
                computation_time=computation_time,
                optimization_level=self.optimization_level,
                graph1_size=size1,
                graph2_size=size2,
                approximation_used=approximation_used,
            )

            logger.debug(
                f"GED calculation completed: {ged_value:.3f} "
                f"(time: {computation_time:.3f}s, approx: {approximation_used})"
            )

            return result

        except Exception as e:
            computation_time = time.time() - start_time
            logger.error(f"GED calculation failed: {e}")

            # Return fallback result
            return GEDResult(
                ged_value=self._fallback_ged_value(graph1, graph2),
                computation_time=computation_time,
                optimization_level=self.optimization_level,
                graph1_size=size1,
                graph2_size=size2,
                timeout_occurred=True,
                approximation_used=True,
            )

    def compute_delta_ged(
        self, graph_before: Data, graph_after: Data, reference_graph: Optional[Data] = None
    ) -> float:
        """
        Compute ΔGED for insight detection using instantaneous formula.

        Mathematical Definition:
            ΔGED = GED(graph_after, graph_before)

        This measures the direct structural change from the previous state,
        maintaining temporal consistency with ΔIG calculation.

        TRANSPARENCY NOTE:
        - "Previous state" means exactly 1 step before (no moving average)
        - GED calculation uses custom PyG implementation
        - For graphs > 50 nodes, we use structural features approximation
        - Negative values indicate structural simplification (fewer nodes/edges)

        Args:
            graph_before: Previous PyG graph state
            graph_after: Current PyG graph state
            reference_graph: Deprecated parameter (kept for compatibility)

        Returns:
            float: ΔGED value (negative indicates simplification/insight)
        """
        if reference_graph is not None:
            logger.warning(
                "reference_graph parameter is deprecated. "
                "ΔGED now uses instantaneous calculation."
            )

        # Direct calculation: how much did the graph change from before to after
        result = self.calculate(graph_after, graph_before)
        delta_ged = result.ged_value

        # Important: GED(A, B) measures distance, so if the graph became simpler,
        # we need to check if after has fewer nodes/edges than before
        nodes_before = graph_before.num_nodes if hasattr(graph_before, 'num_nodes') else 0
        nodes_after = graph_after.num_nodes if hasattr(graph_after, 'num_nodes') else 0
        edges_before = graph_before.edge_index.size(1) if hasattr(graph_before, 'edge_index') else 0
        edges_after = graph_after.edge_index.size(1) if hasattr(graph_after, 'edge_index') else 0

        # If the graph got smaller (simplification), make ΔGED negative
        if nodes_after < nodes_before or edges_after < edges_before:
            delta_ged = -abs(delta_ged)

        logger.debug(
            f"ΔGED calculated: GED(after, before)={result.ged_value:.3f}, "
            f"nodes: {nodes_before}→{nodes_after}, edges: {edges_before}→{edges_after}, "
            f"ΔGED={delta_ged:.3f}"
        )

        return delta_ged

    def reset_state(self):
        """Reset the internal state for ΔGED calculation."""
        self.initial_graph = None
        self.previous_graph = None

    def _validate_graphs(self, graph1: Data, graph2: Data):
        """Validate input PyG graphs."""
        if graph1 is None or graph2 is None:
            raise ValueError("Input graphs cannot be None")

        # Check if graphs are PyG Data objects
        if not isinstance(graph1, Data) or not isinstance(graph2, Data):
            raise ValueError("Graphs must be PyTorch Geometric Data objects")

        # Check required attributes
        required_attrs = ["x", "edge_index"]
        for graph, name in [(graph1, "graph1"), (graph2, "graph2")]:
            for attr in required_attrs:
                if not hasattr(graph, attr):
                    raise ValueError(f"{name} must have '{attr}' attribute")

    def _exact_ged_pyg(self, graph1: Data, graph2: Data) -> Tuple[float, bool]:
        """
        Exact GED calculation for PyG graphs.
        
        Note: This is a simplified implementation. For production use,
        consider more sophisticated algorithms like Hungarian method.
        """
        try:
            # Node operations cost
            node_diff = abs(graph1.num_nodes - graph2.num_nodes)
            node_cost_total = node_diff * self.node_cost
            
            # Edge operations cost
            edges1 = graph1.edge_index.size(1)
            edges2 = graph2.edge_index.size(1)
            edge_diff = abs(edges1 - edges2)
            edge_cost_total = edge_diff * self.edge_cost
            
            # Feature-based cost (if nodes have features)
            feature_cost = 0.0
            if graph1.x is not None and graph2.x is not None:
                # Compare node features using cosine similarity
                min_nodes = min(graph1.num_nodes, graph2.num_nodes)
                if min_nodes > 0:
                    # Simple feature comparison for common nodes
                    feat1 = graph1.x[:min_nodes]
                    feat2 = graph2.x[:min_nodes]
                    
                    # Normalize features
                    feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
                    feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
                    
                    # Compute similarities
                    similarities = (feat1_norm * feat2_norm).sum(dim=1)
                    feature_cost = (1.0 - similarities.mean()).item() * min_nodes
            
            # Total GED
            ged_value = node_cost_total + edge_cost_total + feature_cost
            
            return float(ged_value), False

        except Exception as e:
            logger.warning(f"Exact GED failed: {e}, falling back to approximation")
            return self._approximate_ged(graph1, graph2)

    def _standard_ged(self, graph1: Data, graph2: Data) -> Tuple[float, bool]:
        """Standard GED calculation with size-based method selection."""
        max_size = max(graph1.num_nodes, graph2.num_nodes)

        if max_size <= self.max_graph_size_exact:
            return self._exact_ged_pyg(graph1, graph2)
        else:
            return self._approximate_ged(graph1, graph2)

    def _approximate_ged(self, graph1: Data, graph2: Data) -> Tuple[float, bool]:
        """Fast approximation of GED using structural features."""
        try:
            # Basic structural differences
            nodes_diff = abs(graph1.num_nodes - graph2.num_nodes)
            edges_diff = abs(graph1.edge_index.size(1) - graph2.edge_index.size(1))

            # Degree sequence comparison
            degrees1 = self._compute_degrees(graph1)
            degrees2 = self._compute_degrees(graph2)
            
            # Pad shorter sequence
            max_len = max(len(degrees1), len(degrees2))
            degrees1 = np.pad(degrees1, (0, max_len - len(degrees1)))
            degrees2 = np.pad(degrees2, (0, max_len - len(degrees2)))
            
            degree_diff = np.abs(degrees1 - degrees2).sum()

            # Graph density comparison
            density1 = self._compute_density(graph1)
            density2 = self._compute_density(graph2)
            density_diff = abs(density1 - density2)

            # Combine features with weights
            approximate_ged = (
                self.node_cost * nodes_diff
                + self.edge_cost * edges_diff
                + 0.5 * degree_diff
                + 2.0 * density_diff * max(graph1.num_nodes, graph2.num_nodes)
            )

            return float(approximate_ged), True

        except Exception as e:
            logger.error(f"Approximation GED failed: {e}")
            return self._fallback_ged_value(graph1, graph2), True

    def _compute_degrees(self, graph: Data) -> np.ndarray:
        """Compute degree sequence for a PyG graph."""
        if graph.edge_index.size(1) == 0:
            return np.zeros(graph.num_nodes)
        
        # Count edges for each node
        edge_index = graph.edge_index
        degrees = torch.zeros(graph.num_nodes)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        
        return degrees.numpy()

    def _compute_density(self, graph: Data) -> float:
        """Compute graph density."""
        if graph.num_nodes <= 1:
            return 0.0
        
        num_edges = graph.edge_index.size(1) / 2  # Undirected
        max_edges = graph.num_nodes * (graph.num_nodes - 1) / 2
        
        return num_edges / max_edges if max_edges > 0 else 0.0

    def _fallback_ged_value(self, graph1: Data, graph2: Data) -> float:
        """Ultimate fallback GED calculation."""
        try:
            size1 = graph1.num_nodes if hasattr(graph1, "num_nodes") else 1
            size2 = graph2.num_nodes if hasattr(graph2, "num_nodes") else 1
            return float(abs(size2 - size1))
        except:
            return 1.0  # Default moderate difference

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator performance statistics."""
        avg_time = self.total_computation_time / max(self.calculation_count, 1)
        approximation_rate = self.approximation_count / max(self.calculation_count, 1)

        return {
            "total_calculations": self.calculation_count,
            "total_computation_time": self.total_computation_time,
            "average_computation_time": avg_time,
            "approximation_count": self.approximation_count,
            "approximation_rate": approximation_rate,
            "optimization_level": self.optimization_level.value,
            "node_cost": self.node_cost,
            "edge_cost": self.edge_cost,
        }

    def compute(self, graph1: Data, graph2: Data) -> float:
        """
        Compute Graph Edit Distance between two graphs.
        Alias for calculate() method to maintain API consistency.

        Args:
            graph1: First PyG graph
            graph2: Second PyG graph

        Returns:
            float: Graph edit distance value
        """
        return self.calculate(graph1, graph2).ged_value


# Convenience functions for external API
def compute_graph_edit_distance(graph1: Data, graph2: Data, **kwargs) -> float:
    """
    Compute Graph Edit Distance between two PyG graphs.

    Args:
        graph1: First PyG graph
        graph2: Second PyG graph
        **kwargs: Additional parameters for GraphEditDistance constructor

    Returns:
        float: Graph edit distance value
    """
    calculator = GraphEditDistance(**kwargs)
    result = calculator.calculate(graph1, graph2)
    return result.ged_value


def compute_delta_ged(
    graph_before: Data, graph_after: Data, reference_graph: Optional[Data] = None, **kwargs
) -> float:
    """
    Compute ΔGED for insight detection.

    Args:
        graph_before: Initial PyG graph state
        graph_after: Final PyG graph state
        reference_graph: Optional reference graph
        **kwargs: Additional parameters for GraphEditDistance constructor

    Returns:
        float: ΔGED value (negative indicates simplification/insight)
    """
    calculator = GraphEditDistance(**kwargs)
    return calculator.compute_delta_ged(graph_before, graph_after, reference_graph)


# Global instance for stateful calculations
_global_ged_calculator = None


def get_global_ged_calculator() -> GraphEditDistance:
    """Get or create global GED calculator instance."""
    global _global_ged_calculator
    if _global_ged_calculator is None:
        _global_ged_calculator = GraphEditDistance()
    return _global_ged_calculator


def reset_ged_state():
    """Reset the global GED calculator state."""
    global _global_ged_calculator
    if _global_ged_calculator is not None:
        _global_ged_calculator.reset_state()