"""ΔGED / ΔIG utilities - Wrappers for algorithm implementations."""
import networkx as nx
import numpy as np

try:
    from ..algorithms.graph_edit_distance import GraphEditDistance, get_global_ged_calculator
    from ..algorithms.information_gain import InformationGain

    _ALGORITHMS_AVAILABLE = True
except ImportError:
    _ALGORITHMS_AVAILABLE = False

# For backward compatibility fallback
_SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

__all__ = ["delta_ged", "delta_ig"]


def delta_ged(g_old: nx.Graph, g_new: nx.Graph) -> float:
    """Calculate graph edit distance delta.

    This is a wrapper around the full GED implementation in algorithms.
    For advanced features, use GraphEditDistance directly.
    
    NOTE: This now uses the proper ΔGED calculation that maintains
    a reference graph for correct insight detection.
    """
    if not _ALGORITHMS_AVAILABLE:
        # Fallback to simple NetworkX implementation
        ged = nx.graph_edit_distance(g_old, g_new, timeout=1.0)
        return float(ged) if ged is not None else 0.0

    # Use the global GED calculator with state tracking
    calculator = get_global_ged_calculator()
    return calculator.compute_delta_ged(g_old, g_new)


def delta_ig(vecs_old: np.ndarray, vecs_new: np.ndarray, k: int = 8) -> float:
    """Calculate information gain delta between two vector sets.

    This is a wrapper that maintains backward compatibility.
    For advanced features and different entropy methods, use InformationGain directly.
    
    Returns positive value when entropy decreases (information gain).
    """
    if _ALGORITHMS_AVAILABLE:
        if vecs_old is None or vecs_new is None:
            return 0.0

        # Use the fixed InformationGain implementation
        ig_calc = InformationGain()
        result = ig_calc.calculate(vecs_old, vecs_new)
        return result.ig_value

    # Fallback implementation using sklearn directly
    if not _SKLEARN_AVAILABLE:
        return 0.0

    # Original implementation for backward compatibility
    k_old_effective = (
        min(k, vecs_old.shape[0] - 1)
        if vecs_old is not None and vecs_old.shape[0] > 1
        else 2
    )
    k_new_effective = (
        min(k, vecs_new.shape[0] - 1)
        if vecs_new is not None and vecs_new.shape[0] > 1
        else 2
    )

    if k_old_effective < 2 or k_new_effective < 2:
        return 0.0

    score_old = 0.0
    score_new = 0.0

    try:
        # Calculate silhouette score for old vectors
        if vecs_old is not None and vecs_old.shape[0] >= k_old_effective:
            kmeans_old = KMeans(
                n_clusters=k_old_effective, random_state=0, n_init="auto"
            ).fit(vecs_old)
            if len(np.unique(kmeans_old.labels_)) >= 2:
                score_old = silhouette_score(vecs_old, kmeans_old.labels_)

        # Calculate silhouette score for new vectors
        if vecs_new is not None and vecs_new.shape[0] >= k_new_effective:
            kmeans_new = KMeans(
                n_clusters=k_new_effective, random_state=0, n_init="auto"
            ).fit(vecs_new)
            if len(np.unique(kmeans_new.labels_)) >= 2:
                score_new = silhouette_score(vecs_new, kmeans_new.labels_)

    except (ValueError, Exception):
        pass

    return score_new - score_old
