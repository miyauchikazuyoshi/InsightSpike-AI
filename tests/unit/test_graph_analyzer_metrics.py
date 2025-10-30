import networkx as nx

from insightspike.features.graph_reasoning.graph_analyzer import GraphAnalyzer


def _stub_delta_ged(prev_graph, curr_graph, **kwargs):
    # pretend normalized GED of 0.4 (negative improvement in Layer3は負値扱い)
    return -0.4


def _stub_delta_ig(prev_graph, curr_graph, **kwargs):
    # ΔH_norm 相当
    return 0.25


def test_graph_analyzer_returns_normalized_metrics():
    analyzer = GraphAnalyzer()
    g_prev = nx.path_graph(3)
    g_curr = nx.path_graph(3)
    g_curr.add_edge(0, 2)  # introduce shortcut

    metrics = analyzer.calculate_metrics(
        current_graph=g_curr,
        previous_graph=g_prev,
        delta_ged_func=_stub_delta_ged,
        delta_ig_func=_stub_delta_ig,
    )

    assert "delta_ged" in metrics
    assert "delta_ged_norm" in metrics
    assert "delta_h" in metrics
    assert "delta_sp" in metrics
    assert "g0" in metrics and "gmin" in metrics

    # delta_h should mirror the stub IG value
    assert metrics["delta_h"] == 0.25
    # delta_ged_norm は正の半径（abs(delta_ged)）
    assert metrics["delta_ged_norm"] == abs(metrics["delta_ged"])
