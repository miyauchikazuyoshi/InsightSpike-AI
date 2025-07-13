"""
Advanced graph metrics using high-quality GED/IG implementations
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch_geometric.data import Data

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import our advanced algorithms
from ..algorithms.graph_edit_distance import GraphEditDistance
from ..algorithms.graph_edit_distance import compute_delta_ged as nx_delta_ged
from ..algorithms.information_gain import InformationGain
from ..algorithms.pyg_adapter import PyGAdapter, PyGGraphEditDistance

logger = logging.getLogger(__name__)


class AdvancedGraphMetrics:
    """
    High-quality graph metrics using proper GED and IG calculations.

    Features:
    - Exact GED calculation for small graphs
    - Approximate GED for large graphs
    - Combined structural and embedding distance
    - Information gain based on actual entropy changes
    """

    def __init__(
        self,
        use_exact_ged: bool = True,
        ged_optimization: str = "standard",
        structural_weight: float = 0.7,
        embedding_weight: float = 0.3,
    ):
        """
        Initialize advanced metrics calculator.

        Args:
            use_exact_ged: Use NetworkX exact GED when possible
            ged_optimization: "fast", "standard", or "precise"
            structural_weight: Weight for structural component
            embedding_weight: Weight for embedding component
        """
        self.use_exact_ged = use_exact_ged
        self.structural_weight = structural_weight
        self.embedding_weight = embedding_weight

        # Initialize calculators
        self.pyg_ged = PyGGraphEditDistance(
            structural_weight=structural_weight,
            embedding_weight=embedding_weight,
            use_networkx=use_exact_ged,
        )

        self.ig_calculator = InformationGain()

        # Statistics
        self.calculation_count = 0
        self.exact_calculations = 0
        self.approximate_calculations = 0

    def delta_ged(
        self,
        old_graph: Optional[Data],
        new_graph: Optional[Data],
        reference_graph: Optional[Data] = None,
    ) -> float:
        """
        Calculate ΔGED using high-quality algorithm.

        Args:
            old_graph: Previous graph state (PyG Data)
            new_graph: Current graph state (PyG Data)
            reference_graph: Optional reference for comparison

        Returns:
            float: ΔGED value (negative indicates insight)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using fallback")
            return self._fallback_delta_ged(old_graph, new_graph)

        # Handle None cases
        if old_graph is None and new_graph is None:
            return 0.0
        elif old_graph is None:
            return 1.0  # New graph created
        elif new_graph is None:
            return -1.0  # Graph removed (simplification)

        self.calculation_count += 1

        try:
            # Use PyG-aware calculator
            delta = self.pyg_ged.compute_delta_ged(
                old_graph, new_graph, reference_graph
            )

            # Track statistics
            if (
                hasattr(self.pyg_ged, "nx_ged_calculator")
                and self.pyg_ged.nx_ged_calculator
            ):
                stats = self.pyg_ged.nx_ged_calculator.get_statistics()
                if stats["approximation_rate"] < 0.5:
                    self.exact_calculations += 1
                else:
                    self.approximate_calculations += 1

            logger.debug(f"Advanced ΔGED: {delta:.3f}")
            return float(delta)

        except Exception as e:
            logger.error(f"Advanced ΔGED calculation failed: {e}")
            return self._fallback_delta_ged(old_graph, new_graph)

    def delta_ig(
        self,
        old_graph: Optional[Data],
        new_graph: Optional[Data],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate ΔIG using proper information theory.

        Args:
            old_graph: Previous graph state
            new_graph: Current graph state
            query: Optional query for context-aware IG
            context: Additional context information

        Returns:
            float: ΔIG value (positive indicates information gain)
        """
        # Handle None cases
        if old_graph is None and new_graph is None:
            return 0.0
        elif old_graph is None:
            return 0.5  # New information
        elif new_graph is None:
            return -0.5  # Information loss

        try:
            # Extract embeddings for entropy calculation
            old_embeddings = self._extract_embeddings(old_graph)
            new_embeddings = self._extract_embeddings(new_graph)

            # Calculate information gain
            if old_embeddings is not None and new_embeddings is not None:
                # InformationGain.calculate() returns IGResult object
                result = self.ig_calculator.calculate(old_embeddings, new_embeddings)

                # Extract the IG value from result
                ig_value = result.ig_value if hasattr(result, "ig_value") else result

                logger.debug(f"Advanced ΔIG: {ig_value:.3f}")
                return float(ig_value)
            else:
                # Fallback to structural IG
                return self._structural_ig(old_graph, new_graph)

        except Exception as e:
            logger.error(f"Advanced ΔIG calculation failed: {e}")
            return self._fallback_delta_ig(old_graph, new_graph)

    def combined_insight_score(
        self,
        old_graph: Optional[Data],
        new_graph: Optional[Data],
        ged_threshold: float = -0.5,
        ig_threshold: float = 0.2,
    ) -> Tuple[float, bool, Dict[str, float]]:
        """
        Calculate combined insight score using ΔGED and ΔIG.

        Returns:
            Tuple of (score, is_insight, components)
        """
        # Calculate components
        delta_ged = self.delta_ged(old_graph, new_graph)
        delta_ig = self.delta_ig(old_graph, new_graph)

        # Normalize values
        norm_ged = -delta_ged if delta_ged < 0 else 0  # Convert negative to positive
        norm_ig = delta_ig if delta_ig > 0 else 0

        # Combined score (weighted average)
        score = 0.6 * norm_ged + 0.4 * norm_ig

        # Check if it's an insight
        is_insight = delta_ged <= ged_threshold and delta_ig >= ig_threshold

        components = {
            "delta_ged": delta_ged,
            "delta_ig": delta_ig,
            "normalized_ged": norm_ged,
            "normalized_ig": norm_ig,
            "combined_score": score,
        }

        logger.info(
            f"Insight detection: ΔGED={delta_ged:.3f}, ΔIG={delta_ig:.3f}, "
            f"Score={score:.3f}, Is_insight={is_insight}"
        )

        return score, is_insight, components

    def _extract_embeddings(self, graph: Data) -> Optional[np.ndarray]:
        """Extract node embeddings from PyG graph"""
        if not TORCH_AVAILABLE or graph is None:
            return None

        try:
            if hasattr(graph, "x") and graph.x is not None:
                return graph.x.cpu().numpy()
            return None
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def _structural_ig(self, old_graph: Data, new_graph: Data) -> float:
        """Calculate IG based on structural changes"""
        old_features = PyGAdapter.extract_structural_features(old_graph)
        new_features = PyGAdapter.extract_structural_features(new_graph)

        # Simple entropy-based calculation
        old_entropy = self._graph_entropy(old_features)
        new_entropy = self._graph_entropy(new_features)

        # Information gain is reduction in entropy
        ig = old_entropy - new_entropy
        return float(ig)

    def _graph_entropy(self, features: dict) -> float:
        """Estimate graph entropy from features"""
        n = max(features["num_nodes"], 1)
        e = features["num_edges"]

        # Shannon entropy based on edge probability
        if n > 1:
            max_edges = n * (n - 1) / 2
            p = e / max_edges if max_edges > 0 else 0

            if 0 < p < 1:
                entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
            else:
                entropy = 0
        else:
            entropy = 0

        # Add degree distribution entropy
        avg_degree = features["avg_degree"]
        if avg_degree > 0:
            degree_entropy = np.log(avg_degree + 1)
            entropy += 0.3 * degree_entropy

        return float(entropy)

    def _fallback_delta_ged(self, old_graph: Any, new_graph: Any) -> float:
        """Simple fallback for ΔGED"""
        try:
            old_nodes = old_graph.num_nodes if hasattr(old_graph, "num_nodes") else 0
            new_nodes = new_graph.num_nodes if hasattr(new_graph, "num_nodes") else 0

            old_edges = (
                old_graph.edge_index.size(1) if hasattr(old_graph, "edge_index") else 0
            )
            new_edges = (
                new_graph.edge_index.size(1) if hasattr(new_graph, "edge_index") else 0
            )

            # Simple difference
            node_diff = new_nodes - old_nodes
            edge_diff = new_edges - old_edges

            # Negative if reduction (insight)
            return float(node_diff + 0.5 * edge_diff)
        except:
            return 0.0

    def _fallback_delta_ig(self, old_graph: Any, new_graph: Any) -> float:
        """Simple fallback for ΔIG"""
        try:
            # Growth indicates information gain
            old_size = old_graph.num_nodes if hasattr(old_graph, "num_nodes") else 0
            new_size = new_graph.num_nodes if hasattr(new_graph, "num_nodes") else 0

            if old_size == 0:
                return 0.5 if new_size > 0 else 0.0

            growth_rate = (new_size - old_size) / old_size
            return float(np.tanh(growth_rate))  # Bounded [-1, 1]
        except:
            return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculation statistics"""
        total = max(self.calculation_count, 1)

        return {
            "total_calculations": self.calculation_count,
            "exact_calculations": self.exact_calculations,
            "approximate_calculations": self.approximate_calculations,
            "exact_rate": self.exact_calculations / total,
            "approximate_rate": self.approximate_calculations / total,
            "structural_weight": self.structural_weight,
            "embedding_weight": self.embedding_weight,
        }


# Convenience functions that match the existing API
_advanced_metrics = None


def get_advanced_metrics() -> AdvancedGraphMetrics:
    """Get or create singleton metrics calculator"""
    global _advanced_metrics
    if _advanced_metrics is None:
        _advanced_metrics = AdvancedGraphMetrics()
    return _advanced_metrics


def delta_ged(
    old_graph: Optional[Any], new_graph: Optional[Any], use_advanced: bool = True
) -> float:
    """
    Calculate ΔGED with optional advanced algorithm.

    This function maintains compatibility with existing code.
    """
    if use_advanced:
        return get_advanced_metrics().delta_ged(old_graph, new_graph)
    else:
        # Fallback to simple calculation
        return get_advanced_metrics()._fallback_delta_ged(old_graph, new_graph)


def delta_ig(
    old_graph: Optional[Any],
    new_graph: Optional[Any],
    query: Optional[str] = None,
    use_advanced: bool = True,
) -> float:
    """
    Calculate ΔIG with optional advanced algorithm.

    This function maintains compatibility with existing code.
    """
    if use_advanced:
        return get_advanced_metrics().delta_ig(old_graph, new_graph, query)
    else:
        # Fallback to simple calculation
        return get_advanced_metrics()._fallback_delta_ig(old_graph, new_graph)
