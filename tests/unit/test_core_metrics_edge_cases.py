import numpy as np
import networkx as nx

from insightspike.algorithms.core.metrics import normalized_ged, entropy_ig


def test_normalized_ged_empty_graphs():
    g1 = nx.Graph()
    g2 = nx.Graph()
    out = normalized_ged(g1, g2)
    assert isinstance(out, dict)
    assert out["raw_ged"] == 0.0
    assert out["normalized_ged"] == 0.0
    # structural_improvement should be 0 when no change
    assert abs(out["structural_improvement"]) < 1e-9


def test_normalized_ged_spectral_safe():
    g1 = nx.path_graph(3)
    g2 = nx.path_graph(4)
    out = normalized_ged(g1, g2, enable_spectral=True, spectral_weight=0.5)
    assert "structural_improvement" in out
    assert -1.0 <= out["structural_improvement"] <= 1.0


def test_entropy_ig_min_nodes_guard():
    g = nx.Graph()
    g.add_node(0)
    fb = np.zeros((1, 4))
    fa = np.ones((1, 4))
    out = entropy_ig(g, fb, fa, min_nodes=2)
    assert out["ig_value"] == 0.0
    assert out["variance_reduction"] == 0.0


def test_entropy_ig_small_graph_succeeds():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from([(0, 1), (1, 2)])
    fb = np.random.randn(3, 8)
    fa = np.random.randn(3, 8)
    out = entropy_ig(g, fb, fa)
    assert isinstance(out["ig_value"], float)
    # Entropy means should be finite
    assert np.isfinite(out["entropy_before"]) and np.isfinite(out["entropy_after"]) 


def test_normalized_ged_normalization_variants():
    g1 = nx.path_graph(3)
    g2 = nx.path_graph(5)
    v_sum = normalized_ged(g1, g2, normalization="sum")["normalized_ged"]
    v_max = normalized_ged(g1, g2, normalization="max")["normalized_ged"]
    v_mean = normalized_ged(g1, g2, normalization="mean")["normalized_ged"]
    for v in (v_sum, v_max, v_mean):
        assert 0.0 <= v <= 1.0
    # Different normalization schemes should not all collapse to identical values
    assert len({round(v_sum, 6), round(v_max, 6), round(v_mean, 6)}) >= 2


def test_entropy_ig_smoothing_near_zero_is_finite():
    g = nx.path_graph(4)
    fb = np.random.randn(4, 16)
    fa = np.random.randn(4, 16)
    out = entropy_ig(g, fb, fa, smoothing=1e-16)
    assert np.isfinite(out["ig_value"]) and np.isfinite(out["variance_reduction"]) 
