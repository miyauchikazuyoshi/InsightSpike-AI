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
    """Calculate information gain delta between two vector sets."""
    # データポイント数に合わせてクラスター数を調整
    k_old = min(k, vecs_old.shape[0] - 1) if vecs_old is not None and vecs_old.shape[0] > 1 else 2
    k_new = min(k, vecs_new.shape[0] - 1) if vecs_new is not None and vecs_new.shape[0] > 1 else 2
    
    # 最低でも2つのクラスターが必要
    if k_old < 2 or k_new < 2:
        return 0.0
    
    try:
        km_old = KMeans(k_old).fit(vecs_old)
        km_new = KMeans(k_new).fit(vecs_new)
        
        # シルエットスコアは少なくとも2つのクラスターと各クラスターに少なくとも1つのサンプルが必要
        ig_old = silhouette_score(vecs_old, km_old.labels_) if len(set(km_old.labels_)) > 1 else 0
        ig_new = silhouette_score(vecs_new, km_new.labels_) if len(set(km_new.labels_)) > 1 else 0
        return ig_new - ig_old
    except Exception as e:
        print(f"情報利得計算中にエラー発生: {e}")
        return 0.0
