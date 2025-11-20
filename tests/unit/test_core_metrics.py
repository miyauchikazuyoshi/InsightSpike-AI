import pytest

try:
    import numpy as np
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"numpy not available for core metrics tests: {exc}", allow_module_level=True)
import networkx as nx


def test_normalized_ged_identical_graphs():
    from insightspike.algorithms.core.metrics import normalized_ged

    g1 = nx.Graph(); g1.add_edges_from([(1, 2), (2, 3)])
    g2 = nx.Graph(); g2.add_edges_from([(1, 2), (2, 3)])

    out = normalized_ged(g1, g2)
    assert isinstance(out, dict)
    assert out["normalized_ged"] == 0.0
    assert -1.0 <= out["structural_improvement"] <= 1.0


def test_entropy_ig_smoke():
    from insightspike.algorithms.core.metrics import entropy_ig

    g = nx.Graph(); g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    feats_before = np.ones((4, 4), dtype=float)
    feats_after = np.vstack([
        np.ones((2, 4), dtype=float),
        np.eye(2, 4, k=0, dtype=float),
    ])

    out = entropy_ig(g, feats_before, feats_after)
    assert set(out.keys()) >= {"ig_value", "variance_reduction", "entropy_before", "entropy_after"}
    # Values are finite numbers
    for k in ("ig_value", "variance_reduction", "entropy_before", "entropy_after"):
        assert isinstance(out[k], float)


def test_normalized_ged_disjoint_graphs_range():
    from insightspike.algorithms.core.metrics import normalized_ged
    import networkx as nx

    g1 = nx.Graph(); g1.add_edges_from([(1, 2), (2, 3)])
    g2 = nx.Graph(); g2.add_edges_from([(10, 11), (11, 12), (12, 13)])
    out = normalized_ged(g1, g2)
    # normalized GED should be in [0,1]; structural improvement in [-1,1]
    assert 0.0 <= out["normalized_ged"] <= 1.0
    assert -1.0 <= out["structural_improvement"] <= 1.0


def test_entropy_ig_min_nodes_guard():
    from insightspike.algorithms.core.metrics import entropy_ig
    import numpy as np
    import networkx as nx

    g = nx.Graph(); g.add_edge(0, 1)  # 2 nodes
    feats = np.ones((2, 4), dtype=float)
    out = entropy_ig(g, feats, feats, min_nodes=3)  # requires >=3 nodes
    assert out["ig_value"] == 0.0 and out["entropy_before"] == 0.0 and out["entropy_after"] == 0.0
