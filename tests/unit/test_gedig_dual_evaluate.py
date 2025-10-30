import networkx as nx
from insightspike.algorithms.gedig_factory import GeDIGFactory, dual_evaluate
from insightspike.algorithms.gedig_core import GeDIGCore


def make_graph(n):
    return nx.path_graph(n)


def test_dual_evaluate_delta_increases_with_graph_change():
    """Graph structural change should yield positive delta due to legacy product formula vs ref diff formula."""
    legacy = GeDIGFactory.create({'use_refactored_gedig': False})
    ref = GeDIGFactory.create({'use_refactored_gedig': True})

    g_prev = make_graph(5)
    g_now = make_graph(8)  # add nodes/edges -> different GED path

    res, delta = dual_evaluate(legacy, ref, g_prev=g_prev, g_now=g_now, delta_threshold=0.0)
    # We expect a strictly positive delta now
    assert delta > 0, f"Expected positive divergence delta, got {delta}"
    # sanity: result object should carry rewards
    assert hasattr(res, 'hop0_reward')


def test_dual_evaluate_no_change_small_delta():
    legacy = GeDIGFactory.create({'use_refactored_gedig': False})
    ref = GeDIGFactory.create({'use_refactored_gedig': True})
    g_prev = make_graph(10)
    g_now = make_graph(10)
    res, delta = dual_evaluate(legacy, ref, g_prev=g_prev, g_now=g_now, delta_threshold=0.0)
    # With identical graphs divergence should be near zero (allow small numerical/product-vs-diff discrepancy)
    assert delta <= 0.01, f"Delta too large for identical graphs: {delta}"
