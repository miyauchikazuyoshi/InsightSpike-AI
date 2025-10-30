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
from ..metrics.pyg_compatible_metrics import delta_ged_pyg, delta_ig_pyg

try:
    from ..algorithms.graph_edit_distance import GraphEditDistance
    from ..algorithms.information_gain import InformationGain
    from .gedig_core import delta_ged as advanced_delta_ged
    from .gedig_core import delta_ig as advanced_delta_ig

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
        ged_algo = "pyg"  # Default to PyG-compatible
        ig_algo = "pyg"   # Default to PyG-compatible

        if self.config:
            # Handle both dict and object-style configs
            if isinstance(self.config, dict):
                # First check graph.algorithms section
                graph_config = self.config.get("graph", {})
                algorithms = graph_config.get("algorithms", {})
                ged_algo = algorithms.get("ged", graph_config.get("ged_algorithm", "pyg"))
                ig_algo = algorithms.get("ig", graph_config.get("ig_algorithm", "pyg"))
            else:
                # Object-style config (pydantic). Prefer nested GraphConfig.
                try:
                    g = getattr(self.config, "graph", None)
                    if g is not None:
                        ged_algo = getattr(g, "ged_algorithm", "advanced")
                        ig_algo = getattr(g, "ig_algorithm", "advanced")
                    else:
                        # Fallback to attributes directly on config
                        ged_algo = getattr(self.config, "ged_algorithm", "advanced")
                        ig_algo = getattr(self.config, "ig_algorithm", "advanced")
                except Exception:
                    # Safe fallback
                    ged_algo = "advanced"
                    ig_algo = "advanced"

        # Select GED method
        self._ged_method = self._select_ged_method(ged_algo)

        # Select IG method
        self._ig_method = self._select_ig_method(ig_algo)

        logger.info(f"Metrics initialized: GED={ged_algo}, IG={ig_algo}")

    def _select_ged_method(self, algorithm: str) -> Callable:
        """Select GED calculation method"""
        if algorithm == "pyg":
            logger.info("Using PyG-compatible GED calculation")
            return delta_ged_pyg
        elif algorithm == "hybrid":
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
                    f"Requested GED algorithm '{algorithm}' not available, using PyG-compatible"
                )
            return delta_ged_pyg  # Use PyG-compatible as fallback

    def _select_ig_method(self, algorithm: str) -> Callable:
        """Select IG calculation method"""
        if algorithm == "pyg":
            logger.info("Using PyG-compatible IG calculation")
            return delta_ig_pyg
        elif algorithm == "hybrid":
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
                    f"Requested IG algorithm '{algorithm}' not available, using PyG-compatible"
                )
            return delta_ig_pyg  # Use PyG-compatible as fallback

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

                # Log shapes for debugging
                logger.debug(f"IG calculation - emb1 shape: {emb1.shape}, emb2 shape: {emb2.shape}")
                result = self._ig_calc.calculate(emb1, emb2)
                return result.ig_value if hasattr(result, "ig_value") else result
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Entropy IG calculation failed: {e}")
            return simple_delta_ig(graph1, graph2)

    def delta_ig(self, graph1: Any, graph2: Any, **kwargs) -> float:
        """Compute ΔIG with variance-based raw value, then optional clip/bonus.

        Raw heuristic: ig_raw = var(before_embeddings) - var(after_embeddings) over node feature vectors (flattened).
        Positive if variance decreased (information consolidation). Then apply experimental clipping.
        Falls back to selected IG method if embeddings not available.
        """
        import os, numpy as np
        raw_val = None
        try:
            if hasattr(graph1, 'x') and hasattr(graph2, 'x') and graph1.x is not None and graph2.x is not None:
                def to_np(x):
                    if hasattr(x, 'detach'):
                        x = x.detach()
                    if hasattr(x, 'cpu'):
                        x = x.cpu()
                    return x.numpy() if hasattr(x, 'numpy') else np.array(x)
                emb1 = to_np(graph1.x)
                emb2 = to_np(graph2.x)
                # 空 or 要素ゼロの場合は 0 を返す (NaN 回避)
                if emb1.size == 0 or emb2.size == 0:
                    return 0.0
                # Flatten per-node embeddings to a single distribution dimension-wise
                var1 = float(np.var(emb1))
                var2 = float(np.var(emb2))
                raw_val = var1 - var2  # higher when variance shrinks
            else:
                # Fallback to chosen IG method
                result = self._ig_method(graph1, graph2, **kwargs)
                if not isinstance(result, (int, float)):
                    result = float(result)
                raw_val = result
        except Exception as e:
            logger.debug(f"variance ΔIG fallback due to error: {e}")
            try:
                result = self._ig_method(graph1, graph2, **kwargs)
                raw_val = float(result) if not isinstance(result, (int, float)) else result
            except Exception as e2:
                logger.debug(f"delta_ig failed with {self._ig_method.__name__}: {e2}")
                return 0.0

        # Experimental clipping + bonus (M2) under IG_CLIP_EXPERIMENTAL
        flag = os.getenv("IG_CLIP_EXPERIMENTAL", "0").lower() in ("1","true","yes","on")
        result_val = raw_val
        if flag:
            clipped = max(0.0, raw_val)
            try:
                n1 = getattr(graph1, 'num_nodes', 0) if graph1 is not None else 0
                n2 = getattr(graph2, 'num_nodes', 0) if graph2 is not None else 0
                new_nodes = max(0, n2 - n1)
            except Exception:
                new_nodes = 0
            bonus_factor = 0.01
            result_val = clipped + bonus_factor * new_nodes
            if os.getenv("INSIGHT_DEBUG_METRICS", "0").lower() in ("1","true","on","yes"):
                logger.info(f"IG_CLIP_EXPERIMENTAL: raw={raw_val:.4f} clipped={clipped:.4f} new_nodes={new_nodes} result={result_val:.4f}")
        # NaN 安全化: 数値化後に NaN / inf を 0 にクリップ
        if not isinstance(result_val, (int, float)) or result_val != result_val or result_val in (float('inf'), float('-inf')):
            return 0.0
        return result_val

    def delta_ged(self, graph1: Any, graph2: Any, **kwargs) -> float:
        """Wrapper for selected GED method (no transformation yet)."""
        try:
            result = self._ged_method(graph1, graph2, **kwargs)
            if not isinstance(result, (int, float)):
                result = float(result)
            # Calibration logging (M3 prep): compute basic possible_ged & efficiency
            import os
            if os.getenv("INSIGHT_DEBUG_METRICS", "0").lower() in ("1","true","yes","on"):
                try:
                    n1 = getattr(graph1, 'num_nodes', 0) if graph1 is not None else 0
                    n2 = getattr(graph2, 'num_nodes', 0) if graph2 is not None else 0
                    # naive upper bound: difference in node count + proportional edge diff
                    e1 = getattr(graph1, 'num_edges', None)
                    if e1 is None and hasattr(graph1, 'edge_index'):
                        e1 = int(getattr(graph1.edge_index, 'shape', [0,0])[0] or 0)
                    e2 = getattr(graph2, 'num_edges', None)
                    if e2 is None and hasattr(graph2, 'edge_index'):
                        e2 = int(getattr(graph2.edge_index, 'shape', [0,0])[0] or 0)
                    e1 = e1 or 0
                    e2 = e2 or 0
                    possible = abs(n2 - n1) + 0.5 * abs(e2 - e1)
                    if possible < 1e-6:
                        possible = 1.0
                    efficiency = result / possible
                    alpha = float(os.getenv("GED_ALPHA", "1.0") or 1.0)
                    normalized = result / alpha if alpha != 0 else result
                    # Optional EMA smoothing of efficiency
                    try:
                        window = int(os.getenv("GEDIG_EFF_EWMA", "0") or 0)
                    except ValueError:
                        window = 0
                    if window > 1:
                        decay = 2 / (window + 1)
                        prev = getattr(self, "_eff_ema", efficiency)
                        eff_ema = prev + decay * (efficiency - prev)
                        self._eff_ema = eff_ema
                    else:
                        eff_ema = efficiency
                    # Store last metrics for external diagnostics
                    self._last_ged_metrics = {
                        "raw": result,
                        "possible": possible,
                        "efficiency": efficiency,
                        "efficiency_ema": eff_ema,
                        "alpha": alpha,
                        "normalized": normalized,
                    }
                    logger.info(f"GED_CALIBRATION raw={result:.4f} possible={possible:.4f} eff={efficiency:.4f} alpha={alpha:.3f}")
                    return normalized
                except Exception as log_e:
                    logger.debug(f"GED calibration logging failed: {log_e}")
            # No debug flag: still apply alpha normalization silently
            try:
                alpha = float(os.getenv("GED_ALPHA", "1.0") or 1.0)
                if alpha != 0:
                    return result / alpha
            except Exception:
                pass
            return result
        except Exception as e:
            logger.debug(f"delta_ged failed with {self._ged_method.__name__}: {e}")
            return 0.0

    def get_algorithm_info(self) -> Dict[str, str]:
        """Get information about current algorithms"""
        # Extract algorithm names from config
        ged_algo = "unknown"
        ig_algo = "unknown"
        
        if self.config:
            if isinstance(self.config, dict):
                graph_config = self.config.get("graph", {})
                algorithms = graph_config.get("algorithms", {})
                ged_algo = algorithms.get("ged", graph_config.get("ged_algorithm", "unknown"))
                ig_algo = algorithms.get("ig", graph_config.get("ig_algorithm", "unknown"))
            else:
                ged_algo = getattr(self.config, "ged_algorithm", "unknown")
                ig_algo = getattr(self.config, "ig_algorithm", "unknown")
                
        return {
            "ged_algorithm": ged_algo,
            "ig_algorithm": ig_algo,
            "advanced_available": ADVANCED_AVAILABLE,
            "networkx_available": NETWORKX_AVAILABLE,
        }

    def get_last_ged_metrics(self) -> Optional[Dict[str, float]]:
        """Return last stored GED calibration metrics if available."""
        return getattr(self, '_last_ged_metrics', None)

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
