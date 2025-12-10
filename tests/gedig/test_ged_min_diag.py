import networkx as nx

from insightspike.algorithms.gedig_core import GeDIGCore


def _make_graph(with_shortcut: bool) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from([(0, 1), (1, 2)])
    if with_shortcut:
        g.add_edge(0, 2)
    return g


def test_ged_min_proxy_enabled():
    core = GeDIGCore(enable_ged_min_diag=True, lambda_weight=0.0, enable_multihop=False, use_multihop_sp_gain=False)
    g_before = _make_graph(with_shortcut=False)
    g_after = _make_graph(with_shortcut=True)

    res = core.calculate(g_prev=g_before, g_now=g_after, features_prev=None, features_now=None)

    # Triangle vs path3: ASP decreases from 4/3 -> 1, so proxy ≈ 0.25
    assert 0.24 < res.ged_min_proxy < 0.26


def test_ged_min_proxy_disabled():
    core = GeDIGCore(enable_ged_min_diag=False, lambda_weight=0.0, enable_multihop=False, use_multihop_sp_gain=False)
    g_before = _make_graph(with_shortcut=False)
    g_after = _make_graph(with_shortcut=True)

    res = core.calculate(g_prev=g_before, g_now=g_after, features_prev=None, features_now=None)
    assert res.ged_min_proxy == 0.0
