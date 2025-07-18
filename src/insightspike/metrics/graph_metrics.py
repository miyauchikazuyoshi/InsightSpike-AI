"""ΔGED / ΔIG utilities - Wrappers for algorithm implementations."""
import numpy as np
import networkx as nx

try:
    from ..algorithms.graph_edit_distance import GraphEditDistance
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
    """
    if not _ALGORITHMS_AVAILABLE:
        # Fallback to simple NetworkX implementation
        ged = nx.graph_edit_distance(g_old, g_new, timeout=1.0)
        return float(ged) if ged is not None else 0.0
    
    # Use the comprehensive implementation
    ged_calc = GraphEditDistance()
    result = ged_calc.calculate(g_old, g_new)
    return result.ged_value


def delta_ig(vecs_old: np.ndarray, vecs_new: np.ndarray, k: int = 8) -> float:
    """Calculate information gain delta between two vector sets.
    
    This is a wrapper that maintains backward compatibility.
    For advanced features and different entropy methods, use InformationGain directly.
    """
    if _ALGORITHMS_AVAILABLE:
        # Use the comprehensive implementation
        ig_calc = InformationGain()
        
        if vecs_old is None or vecs_new is None:
            return 0.0
            
        # Use clustering-based method (similar to original silhouette approach)
        score_old = ig_calc.calculate_from_vectors(vecs_old, method='clustering', k=k)
        score_new = ig_calc.calculate_from_vectors(vecs_new, method='clustering', k=k)
        
        return score_new - score_old
    
    # Fallback implementation using sklearn directly
    if not _SKLEARN_AVAILABLE:
        return 0.0
    
    # Original implementation for backward compatibility
    k_old_effective = min(k, vecs_old.shape[0] - 1) if vecs_old is not None and vecs_old.shape[0] > 1 else 2
    k_new_effective = min(k, vecs_new.shape[0] - 1) if vecs_new is not None and vecs_new.shape[0] > 1 else 2
    
    if k_old_effective < 2 or k_new_effective < 2:
        return 0.0
    
    score_old = 0.0
    score_new = 0.0
    
    try:
        # Calculate silhouette score for old vectors
        if vecs_old is not None and vecs_old.shape[0] >= k_old_effective:
            kmeans_old = KMeans(n_clusters=k_old_effective, random_state=0, n_init="auto").fit(vecs_old)
            if len(np.unique(kmeans_old.labels_)) >= 2:
                score_old = silhouette_score(vecs_old, kmeans_old.labels_)
        
        # Calculate silhouette score for new vectors
        if vecs_new is not None and vecs_new.shape[0] >= k_new_effective:
            kmeans_new = KMeans(n_clusters=k_new_effective, random_state=0, n_init="auto").fit(vecs_new)
            if len(np.unique(kmeans_new.labels_)) >= 2:
                score_new = silhouette_score(vecs_new, kmeans_new.labels_)
    
    except (ValueError, Exception):
        pass
    
    return score_new - score_old