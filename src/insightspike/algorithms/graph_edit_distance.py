"""
Graph Edit Distance (GED) Algorithm Implementation
=================================================

Implementation of Graph Edit Distance calculation for InsightSpike-AI's geDIG technology.
This module provides the core ΔGED computation for detecting structural insight moments.

Mathematical Foundation:
    ΔGED = GED(G_after, G_reference) - GED(G_before, G_reference)
    
    Where GED(G1, G2) = min Σ(cost(operation)) for all edit operations
    that transform G1 into G2.

Key Insight Detection:
    - Negative ΔGED values indicate structural simplification (insight detected)
    - ΔGED ≤ -0.5 threshold typically indicates EurekaSpike
    - Combined with ΔIG ≥ 0.2 for full geDIG detection

References:
    - Riesen, K., & Bunke, H. (2009). Approximate graph edit distance computation by means of bipartite graph matching.
    - NetworkX Documentation: https://networkx.org/documentation/stable/reference/algorithms/graph_edit_distance.html
"""

import logging
from typing import Any, Dict, Optional, Union, Tuple
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    # Create mock for graceful fallback
    class MockGraph:
        def __init__(self):
            self.nodes = lambda: []
            self.edges = lambda: []
            
    nx = type('MockNetworkX', (), {
        'Graph': MockGraph,
        'graph_edit_distance': lambda g1, g2, **kwargs: 0.0
    })

logger = logging.getLogger(__name__)

__all__ = [
    "GraphEditDistance", 
    "OptimizationLevel", 
    "GEDResult",
    "compute_graph_edit_distance",
    "compute_delta_ged"
]


class OptimizationLevel(Enum):
    """Optimization levels for GED calculation."""
    FAST = "fast"           # O(n²) approximation
    STANDARD = "standard"   # O(n³) exact for small graphs, approximation for large
    PRECISE = "precise"     # O(n!) exact calculation (use with caution)


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
            self.optimization_level != OptimizationLevel.FAST or 
            max(self.graph1_size, self.graph2_size) <= 20
        )


class GraphEditDistance:
    """
    Graph Edit Distance calculator for measuring conceptual changes.
    
    This implementation focuses on educational concept graphs where nodes
    represent concepts and edges represent relationships between concepts.
    
    The calculator supports multiple optimization levels:
    - FAST: Approximation suitable for real-time applications
    - STANDARD: Balanced precision/performance (recommended)
    - PRECISE: Exact calculation for research applications
    """
    
    def __init__(self, 
                 node_cost: float = 1.0,
                 edge_cost: float = 1.0,
                 optimization_level: Union[str, OptimizationLevel] = OptimizationLevel.STANDARD,
                 timeout_seconds: float = 5.0,
                 max_graph_size_exact: int = 50):
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
        
        logger.info(f"GED Calculator initialized: {optimization_level.value} mode, "
                   f"node_cost={node_cost}, edge_cost={edge_cost}")
    
    def calculate(self, graph1: Any, graph2: Any) -> GEDResult:
        """
        Calculate Graph Edit Distance between two graphs.
        
        Args:
            graph1: First graph (NetworkX Graph or compatible)
            graph2: Second graph (NetworkX Graph or compatible)
            
        Returns:
            GEDResult: Detailed calculation result
            
        Raises:
            ValueError: If graphs are invalid or incompatible
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, using fallback calculation")
            return self._fallback_calculation(graph1, graph2)
            
        # Validate inputs
        self._validate_graphs(graph1, graph2)
        
        start_time = time.time()
        self.calculation_count += 1
        
        try:
            # Determine calculation method based on graph sizes and optimization level
            size1 = len(graph1.nodes()) if hasattr(graph1, 'nodes') else 0
            size2 = len(graph2.nodes()) if hasattr(graph2, 'nodes') else 0
            max_size = max(size1, size2)
            
            if (self.optimization_level == OptimizationLevel.FAST or 
                max_size > self.max_graph_size_exact):
                ged_value, approximation_used = self._approximate_ged(graph1, graph2)
            elif self.optimization_level == OptimizationLevel.PRECISE:
                ged_value, approximation_used = self._exact_ged(graph1, graph2)
            else:  # STANDARD
                ged_value, approximation_used = self._standard_ged(graph1, graph2)
            
            computation_time = time.time() - start_time
            self.total_computation_time += computation_time
            
            if approximation_used:
                self.approximation_count += 1
            
            result = GEDResult(
                ged_value=ged_value,
                computation_time=computation_time,
                optimization_level=self.optimization_level,
                graph1_size=size1,
                graph2_size=size2,
                approximation_used=approximation_used
            )
            
            logger.debug(f"GED calculation completed: {ged_value:.3f} "
                        f"(time: {computation_time:.3f}s, approx: {approximation_used})")
            
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
                approximation_used=True
            )
    
    def compute_delta_ged(self, graph_before: Any, graph_after: Any, 
                         reference_graph: Optional[Any] = None) -> float:
        """
        Compute ΔGED for insight detection.
        
        Mathematical Definition:
            If reference_graph provided:
                ΔGED = GED(graph_after, reference) - GED(graph_before, reference)
            Else:
                ΔGED = GED(graph_before, graph_after)
        
        Args:
            graph_before: Initial graph state
            graph_after: Final graph state  
            reference_graph: Optional reference for comparison
            
        Returns:
            float: ΔGED value (negative indicates simplification/insight)
        """
        if reference_graph is not None:
            # Calculate GED to reference for both states
            ged_before = self.calculate(graph_before, reference_graph).ged_value
            ged_after = self.calculate(graph_after, reference_graph).ged_value
            delta_ged = ged_after - ged_before
        else:
            # Direct GED between before and after
            delta_ged = self.calculate(graph_before, graph_after).ged_value
            
        logger.debug(f"ΔGED calculated: {delta_ged:.3f}")
        return delta_ged
    
    def _validate_graphs(self, graph1: Any, graph2: Any):
        """Validate input graphs."""
        if graph1 is None or graph2 is None:
            raise ValueError("Input graphs cannot be None")
            
        # Check if graphs have required methods
        required_methods = ['nodes', 'edges']
        for graph, name in [(graph1, 'graph1'), (graph2, 'graph2')]:
            for method in required_methods:
                if not hasattr(graph, method):
                    raise ValueError(f"{name} must have '{method}' method")
    
    def _exact_ged(self, graph1: Any, graph2: Any) -> Tuple[float, bool]:
        """Exact GED calculation using NetworkX."""
        try:
            ged = nx.graph_edit_distance(
                graph1, graph2,
                node_del_cost=self.node_cost,
                node_ins_cost=self.node_cost,
                edge_del_cost=self.edge_cost,
                edge_ins_cost=self.edge_cost,
                timeout=self.timeout_seconds
            )
            
            if ged is None:  # Timeout occurred
                logger.warning("GED calculation timeout, using approximation")
                return self._approximate_ged(graph1, graph2)
                
            return float(ged), False
            
        except Exception as e:
            logger.warning(f"Exact GED failed: {e}, falling back to approximation")
            return self._approximate_ged(graph1, graph2)
    
    def _standard_ged(self, graph1: Any, graph2: Any) -> Tuple[float, bool]:
        """Standard GED calculation with size-based method selection."""
        size1 = len(graph1.nodes())
        size2 = len(graph2.nodes())
        max_size = max(size1, size2)
        
        if max_size <= self.max_graph_size_exact:
            return self._exact_ged(graph1, graph2)
        else:
            return self._approximate_ged(graph1, graph2)
    
    def _approximate_ged(self, graph1: Any, graph2: Any) -> Tuple[float, bool]:
        """Fast approximation of GED using structural features."""
        try:
            # Calculate basic structural differences
            nodes_diff = abs(len(graph1.nodes()) - len(graph2.nodes()))
            edges_diff = abs(len(graph1.edges()) - len(graph2.edges()))
            
            # Degree sequence comparison
            degrees1 = sorted([degree for node, degree in graph1.degree()])
            degrees2 = sorted([degree for node, degree in graph2.degree()])
            
            # Pad shorter sequence with zeros
            max_len = max(len(degrees1), len(degrees2))
            degrees1.extend([0] * (max_len - len(degrees1)))
            degrees2.extend([0] * (max_len - len(degrees2)))
            
            degree_diff = sum(abs(d1 - d2) for d1, d2 in zip(degrees1, degrees2))
            
            # Clustering coefficient comparison
            try:
                clustering1 = nx.average_clustering(graph1)
                clustering2 = nx.average_clustering(graph2)
                clustering_diff = abs(clustering1 - clustering2)
            except:
                clustering_diff = 0.0
            
            # Combine features with weights
            approximate_ged = (
                self.node_cost * nodes_diff +
                self.edge_cost * edges_diff +
                0.5 * degree_diff +
                2.0 * clustering_diff
            )
            
            return float(approximate_ged), True
            
        except Exception as e:
            logger.error(f"Approximation GED failed: {e}")
            return self._fallback_ged_value(graph1, graph2), True
    
    def _fallback_ged_value(self, graph1: Any, graph2: Any) -> float:
        """Ultimate fallback GED calculation."""
        try:
            size1 = len(graph1.nodes()) if hasattr(graph1, 'nodes') else 1
            size2 = len(graph2.nodes()) if hasattr(graph2, 'nodes') else 1
            return float(abs(size2 - size1))
        except:
            return 1.0  # Default moderate difference
    
    def _fallback_calculation(self, graph1: Any, graph2: Any) -> GEDResult:
        """Fallback calculation when NetworkX unavailable."""
        ged_value = self._fallback_ged_value(graph1, graph2)
        
        return GEDResult(
            ged_value=ged_value,
            computation_time=0.001,  # Minimal time
            optimization_level=self.optimization_level,
            graph1_size=1,
            graph2_size=1,
            approximation_used=True
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator performance statistics."""
        avg_time = (self.total_computation_time / max(self.calculation_count, 1))
        approximation_rate = (self.approximation_count / max(self.calculation_count, 1))
        
        return {
            "total_calculations": self.calculation_count,
            "total_computation_time": self.total_computation_time,
            "average_computation_time": avg_time,
            "approximation_count": self.approximation_count,
            "approximation_rate": approximation_rate,
            "optimization_level": self.optimization_level.value,
            "node_cost": self.node_cost,
            "edge_cost": self.edge_cost
        }


# Convenience functions for external API
def compute_graph_edit_distance(graph1: Any, graph2: Any, **kwargs) -> float:
    """
    Compute Graph Edit Distance between two graphs.
    
    Args:
        graph1: First graph
        graph2: Second graph
        **kwargs: Additional parameters for GraphEditDistance constructor
        
    Returns:
        float: Graph edit distance value
    """
    calculator = GraphEditDistance(**kwargs)
    result = calculator.calculate(graph1, graph2)
    return result.ged_value


def compute_delta_ged(graph_before: Any, graph_after: Any, 
                     reference_graph: Optional[Any] = None, **kwargs) -> float:
    """
    Compute ΔGED for insight detection.
    
    Args:
        graph_before: Initial graph state
        graph_after: Final graph state
        reference_graph: Optional reference graph
        **kwargs: Additional parameters for GraphEditDistance constructor
        
    Returns:
        float: ΔGED value (negative indicates simplification/insight)
    """
    calculator = GraphEditDistance(**kwargs)
    return calculator.compute_delta_ged(graph_before, graph_after, reference_graph)