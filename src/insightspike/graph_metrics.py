"""ΔGED / ΔIG utilities"""
import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

__all__ = ["delta_ged", "delta_ig"]

def delta_ged(g_old: nx.Graph, g_new: nx.Graph) -> float:
    ged = nx.graph_edit_distance(g_old, g_new, timeout=1.0)
    return float(ged) if ged is not None else 0.0

def delta_ig(vecs_old: np.ndarray, vecs_new: np.ndarray, k: int = 8) -> float:
    km_old = KMeans(k).fit(vecs_old)
    km_new = KMeans(k).fit(vecs_new)
    ig_old = silhouette_score(vecs_old, km_old.labels_)
    ig_new = silhouette_score(vecs_new, km_new.labels_)
    return float(ig_new - ig_old)