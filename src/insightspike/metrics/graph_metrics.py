"""ΔGED / ΔIG utilities"""
import networkx as nx
import numpy as np
import os

_SKLEARN_AVAILABLE = False
_silhouette_score = None
_KMeans = None

try:
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score
    from sklearn.cluster import KMeans as sklearn_KMeans
    _silhouette_score = sklearn_silhouette_score
    _KMeans = sklearn_KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    # In LITE_MODE, sklearn might not be available.
    # Functions using these components should handle their absence.
    # A warning could be logged here if needed, but for CI, failing gracefully is often preferred.
    pass

__all__ = ["delta_ged", "delta_ig"]

def delta_ged(g_old: nx.Graph, g_new: nx.Graph) -> float:
    # This function does not use sklearn
    ged = nx.graph_edit_distance(g_old, g_new, timeout=1.0)
    return float(ged) if ged is not None else 0.0

def delta_ig(vecs_old: np.ndarray, vecs_new: np.ndarray, k: int = 8) -> float:
    """Calculate information gain delta between two vector sets."""
    if not _SKLEARN_AVAILABLE:
        # If sklearn is not available (e.g., in LITE_MODE), return 0.0.
        # This allows modules importing graph_metrics to load without error.
        return 0.0

    # Ensure _KMeans and _silhouette_score are loaded if _SKLEARN_AVAILABLE is True
    if _KMeans is None or _silhouette_score is None:
        # This case should ideally not be reached if _SKLEARN_AVAILABLE is True,
        # but as a safeguard:
        return 0.0

    # Adjust k based on the number of samples
    k_old_effective = 2
    if vecs_old is not None and vecs_old.shape[0] > 1:
        k_old_effective = min(k, vecs_old.shape[0] - 1)
    
    k_new_effective = 2
    if vecs_new is not None and vecs_new.shape[0] > 1:
        k_new_effective = min(k, vecs_new.shape[0] - 1)

    # silhouette_score requires at least 2 clusters and 2 samples per cluster (implicitly)
    # KMeans n_clusters must be >= 2 for silhouette_score to be meaningful.
    if k_old_effective < 2 or k_new_effective < 2:
        return 0.0
    
    score_old = 0.0
    score_new = 0.0

    try:
        # Calculate silhouette score for old vectors
        if vecs_old is not None and vecs_old.shape[0] >= k_old_effective:
            kmeans_old = _KMeans(n_clusters=k_old_effective, random_state=0, n_init='auto').fit(vecs_old)
            labels_old = kmeans_old.labels_
            if len(np.unique(labels_old)) >= 2: # silhouette_score requires at least 2 distinct labels
                score_old = _silhouette_score(vecs_old, labels_old)
            # else, score_old remains 0.0 (single cluster or not enough distinct labels)
        
        # Calculate silhouette score for new vectors
        if vecs_new is not None and vecs_new.shape[0] >= k_new_effective:
            kmeans_new = _KMeans(n_clusters=k_new_effective, random_state=0, n_init='auto').fit(vecs_new)
            labels_new = kmeans_new.labels_
            if len(np.unique(labels_new)) >= 2:
                score_new = _silhouette_score(vecs_new, labels_new)
            # else, score_new remains 0.0

    except ValueError:
        # Handles errors like "Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)"
        # Or other KMeans/silhouette_score related ValueErrors.
        # In such cases, effectively the change in IG is treated as 0 or the scores remain 0.0.
        pass # score_old and score_new will remain 0.0 or their last valid value (0.0 here)

    return score_new - score_old
