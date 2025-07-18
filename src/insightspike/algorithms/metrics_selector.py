"""
Metrics Algorithm Selector
=========================

Selects appropriate GED/IG calculation methods based on configuration.
"""

import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Import different implementations
from ..metrics.graph_metrics import delta_ged as simple_delta_ged
from ..metrics.graph_metrics import delta_ig as simple_delta_ig

try:
    from ..algorithms.graph_edit_distance import GraphEditDistance
    from ..algorithms.information_gain import InformationGain
    from ..metrics.advanced_graph_metrics import delta_ged as advanced_delta_ged
    from ..metrics.advanced_graph_metrics import delta_ig as advanced_delta_ig

    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    logger.warning("Advanced metrics not available")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class MetricsSelector:
    """Select and manage metric calculation methods"""

    def __init__(self, config=None):
        self.config = config
        self._ged_method = None
        self._ig_method = None
        self._initialize_methods()

    def _initialize_methods(self):
        """Initialize calculation methods based on config"""
        # Get algorithm selection from config
        ged_algo = "simple"
        ig_algo = "simple"

        if self.config:
            ged_algo = getattr(self.config, "ged_algorithm", "advanced")
            ig_algo = getattr(self.config, "ig_algorithm", "advanced")

        # Select GED method
        self._ged_method = self._select_ged_method(ged_algo)

        # Select IG method
        self._ig_method = self._select_ig_method(ig_algo)

        logger.info(f"Metrics initialized: GED={ged_algo}, IG={ig_algo}")

    def _select_ged_method(self, algorithm: str) -> Callable:
        """Select GED calculation method"""
        if algorithm == "hybrid":
            logger.info("Using hybrid GED calculation")
            return self._hybrid_ged
        elif algorithm == "networkx" and NETWORKX_AVAILABLE and ADVANCED_AVAILABLE:
            logger.info("Using NetworkX-based GED calculation")
            return self._networkx_ged
        elif algorithm == "advanced" and ADVANCED_AVAILABLE:
            logger.info("Using advanced GED calculation")
            return advanced_delta_ged
        else:
            if algorithm != "simple":
                logger.warning(
                    f"Requested GED algorithm '{algorithm}' not available, using simple"
                )
            return simple_delta_ged

    def _select_ig_method(self, algorithm: str) -> Callable:
        """Select IG calculation method"""
        if algorithm == "hybrid":
            logger.info("Using hybrid IG calculation")
            return self._hybrid_ig
        elif algorithm == "entropy" and ADVANCED_AVAILABLE:
            logger.info("Using entropy-based IG calculation")
            return self._entropy_ig
        elif algorithm == "advanced" and ADVANCED_AVAILABLE:
            logger.info("Using advanced IG calculation")
            return advanced_delta_ig
        else:
            if algorithm != "simple":
                logger.warning(
                    f"Requested IG algorithm '{algorithm}' not available, using simple"
                )
            return simple_delta_ig

    def _networkx_ged(self, graph1: Any, graph2: Any) -> float:
        """NetworkX-based GED calculation with PyG conversion"""
        try:
            from ..algorithms.pyg_adapter import PyGAdapter

            # Convert PyG to NetworkX
            nx_graph1 = PyGAdapter.pyg_to_networkx(graph1)
            nx_graph2 = PyGAdapter.pyg_to_networkx(graph2)

            # Use GraphEditDistance calculator
            if not hasattr(self, "_nx_ged_calc"):
                self._nx_ged_calc = GraphEditDistance(
                    node_cost=1.0, edge_cost=0.5, optimization_level="standard"
                )

            result = self._nx_ged_calc.calculate(nx_graph1, nx_graph2)

            # Determine sign based on simplification
            if nx_graph2.number_of_nodes() < nx_graph1.number_of_nodes():
                # Simplification detected - return negative
                return -result.ged_value
            else:
                return result.ged_value

        except Exception as e:
            logger.error(f"NetworkX GED calculation failed: {e}")
            return simple_delta_ged(graph1, graph2)

    def _entropy_ig(self, graph1: Any, graph2: Any) -> float:
        """Entropy-based IG calculation"""
        try:
            # Try improved entropy first
            try:
                from .improved_entropy_ig import ImprovedEntropyIG

                if not hasattr(self, "_improved_ig_calc"):
                    self._improved_ig_calc = ImprovedEntropyIG()
                return self._improved_ig_calc.calculate_ig(graph1, graph2)
            except ImportError:
                pass

            # Fallback to standard entropy
            if not hasattr(self, "_ig_calc"):
                self._ig_calc = InformationGain()

            # Extract embeddings
            if hasattr(graph1, "x") and hasattr(graph2, "x"):
                emb1 = (
                    graph1.x.cpu().numpy()
                    if hasattr(graph1.x, "cpu")
                    else graph1.x.numpy()
                )
                emb2 = (
                    graph2.x.cpu().numpy()
                    if hasattr(graph2.x, "cpu")
                    else graph2.x.numpy()
                )

                result = self._ig_calc.calculate(emb1, emb2)
                return result.ig_value if hasattr(result, "ig_value") else result
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Entropy IG calculation failed: {e}")
            return simple_delta_ig(graph1, graph2)

    def delta_ged(self, graph1: Any, graph2: Any) -> float:
        """Calculate ΔGED using configured method"""
        return self._ged_method(graph1, graph2)

    def delta_ig(self, graph1: Any, graph2: Any) -> float:
        """Calculate ΔIG using configured method"""
        return self._ig_method(graph1, graph2)

    def get_algorithm_info(self) -> Dict[str, str]:
        """Get information about current algorithms"""
        return {
            "ged_algorithm": getattr(self.config, "ged_algorithm", "unknown"),
            "ig_algorithm": getattr(self.config, "ig_algorithm", "unknown"),
            "advanced_available": ADVANCED_AVAILABLE,
            "networkx_available": NETWORKX_AVAILABLE,
        }

    def _hybrid_ged(self, graph1: Any, graph2: Any) -> float:
        """Hybrid GED calculation"""
        try:
            from .hybrid_metrics import HybridMetrics

            if not hasattr(self, "_hybrid_calc"):
                self._hybrid_calc = HybridMetrics(self.config)

            results = self._hybrid_calc.calculate_hybrid_metrics(graph1, graph2)
            return results["hybrid_ged"]

        except Exception as e:
            logger.error(f"Hybrid GED calculation failed: {e}")
            return self._networkx_ged(graph1, graph2)

    def _hybrid_ig(self, graph1: Any, graph2: Any) -> float:
        """Hybrid IG calculation"""
        try:
            from .hybrid_metrics import HybridMetrics

            if not hasattr(self, "_hybrid_calc"):
                self._hybrid_calc = HybridMetrics(self.config)

            results = self._hybrid_calc.calculate_hybrid_metrics(graph1, graph2)
            return results["hybrid_ig"]

        except Exception as e:
            logger.error(f"Hybrid IG calculation failed: {e}")
            return self._entropy_ig(graph1, graph2)


# Global instance
_selector = None


def get_metrics_selector(config=None) -> MetricsSelector:
    """Get or create metrics selector"""
    global _selector
    if _selector is None or config is not None:
        _selector = MetricsSelector(config)
    return _selector


# Convenience functions
def delta_ged(graph1: Any, graph2: Any, config=None) -> float:
    """Calculate ΔGED with configured algorithm"""
    selector = get_metrics_selector(config)
    return selector.delta_ged(graph1, graph2)


def delta_ig(graph1: Any, graph2: Any, config=None) -> float:
    """Calculate ΔIG with configured algorithm"""
    selector = get_metrics_selector(config)
    return selector.delta_ig(graph1, graph2)
