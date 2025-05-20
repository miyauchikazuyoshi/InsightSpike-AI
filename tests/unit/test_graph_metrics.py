import types, sys

# Patch sklearn components
sk_module = types.SimpleNamespace(KMeans=lambda k: types.SimpleNamespace(fit=lambda x: None, labels_=[0]*len(x)),
                                 silhouette_score=lambda x, labels=None: 0.5)
sys.modules['sklearn.cluster'] = types.SimpleNamespace(KMeans=sk_module.KMeans)
sys.modules['sklearn.metrics'] = types.SimpleNamespace(silhouette_score=sk_module.silhouette_score)
sys.modules['sklearn.metrics.pairwise'] = types.SimpleNamespace()

import networkx as nx
from insightspike import graph_metrics


def test_delta_ged():
    g1 = nx.Graph(); g1.add_edge(0,1)
    g2 = nx.Graph(); g2.add_edge(1,2)
    assert isinstance(graph_metrics.delta_ged(g1, g2), float)


def test_delta_ig():
    import numpy as np
    vecs = np.zeros((2,2))
    assert isinstance(graph_metrics.delta_ig(vecs, vecs, k=2), float)
