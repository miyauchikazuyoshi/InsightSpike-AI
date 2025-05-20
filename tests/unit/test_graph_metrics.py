import sys, types
import numpy as np

class DummyKMeans:
    def __init__(self, k):
        self.k = k
        self.labels_ = None
    def fit(self, x):
        self.labels_ = [0] * len(x)
        return self

sys.modules['sklearn.cluster'] = types.SimpleNamespace(KMeans=DummyKMeans)
sys.modules['sklearn.metrics'] = types.SimpleNamespace(silhouette_score=lambda x, labels=None: 0.5)
sys.modules['sklearn.metrics.pairwise'] = types.SimpleNamespace()

import networkx as nx
from insightspike import graph_metrics


def test_delta_ged():
    g1 = nx.Graph(); g1.add_edge(0,1)
    g2 = nx.Graph(); g2.add_edge(1,2)
    assert isinstance(graph_metrics.delta_ged(g1, g2), float)


def test_delta_ig():
    vecs = np.zeros((2,2))
    assert isinstance(graph_metrics.delta_ig(vecs, vecs, k=2), float)
