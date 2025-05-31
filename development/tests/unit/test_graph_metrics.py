import sys, types
import numpy as np
sys.path.insert(0, 'src')  # Add src to path for imports

# Mock sklearn components for graph_metrics more comprehensively
class DummyKMeans:
    def __init__(self, k=2, **kwargs):
        self.k = k
        self.labels_ = None
    def fit(self, x):
        self.labels_ = [0] * len(x)
        return self

# Create comprehensive sklearn mocks
sklearn_cluster = types.SimpleNamespace(KMeans=DummyKMeans)
sklearn_metrics = types.SimpleNamespace(silhouette_score=lambda x, labels=None: 0.5)
sklearn_metrics_pairwise = types.SimpleNamespace(
    paired_cosine_distances=lambda x, y: np.array([0.5]),
    paired_euclidean_distances=lambda x, y: np.array([0.5]),
    paired_manhattan_distances=lambda x, y: np.array([0.5])
)

sys.modules['sklearn'] = types.SimpleNamespace()
sys.modules['sklearn.cluster'] = sklearn_cluster
sys.modules['sklearn.metrics'] = sklearn_metrics
sys.modules['sklearn.metrics.pairwise'] = sklearn_metrics_pairwise

# Also mock sentence_transformers to avoid loading issues
sys.modules['sentence_transformers'] = types.SimpleNamespace(
    SentenceTransformer=lambda *args, **kwargs: types.SimpleNamespace(encode=lambda x: np.random.rand(len(x), 384))
)

# Import networkx directly (not mocked)
try:
    import networkx as nx
    if not hasattr(nx, 'Graph'):
        raise ImportError("NetworkX not properly imported")
except ImportError:
    # Fallback mock for NetworkX if not available
    class MockGraph:
        def __init__(self):
            self.edges = []
        def add_edge(self, a, b):
            self.edges.append((a, b))
    nx = types.SimpleNamespace(Graph=MockGraph)

from insightspike import graph_metrics


"""
Test graph_metrics module functionality
"""
import pytest
import numpy as np
from insightspike import graph_metrics

def test_delta_ged(sample_graph):
    """Test ΔGED calculation between two graphs"""
    # Create a second graph that's slightly different
    g2 = sample_graph  # Using same graph for simplicity in mocked environment
    result = graph_metrics.delta_ged(sample_graph, g2)
    assert isinstance(result, float)
    assert result >= 0

def test_delta_ig(sample_vectors):
    """Test ΔIG calculation between vector sets"""
    vecs1 = sample_vectors[:3]  # First 3 vectors
    vecs2 = sample_vectors[2:]  # Last 3 vectors (with overlap)
    result = graph_metrics.delta_ig(vecs1, vecs2)
    assert isinstance(result, float)
