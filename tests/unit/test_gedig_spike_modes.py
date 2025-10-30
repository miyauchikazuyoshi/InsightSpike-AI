import networkx as nx
from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGPresets


def build_pair(modify=False):
    g1 = nx.Graph()
    g1.add_nodes_from(range(5))
    g1.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
    g2 = g1.copy()
    if modify:
        # add structural change
        g2.add_edge(0,4)
        g2.add_node(5)
        g2.add_edge(2,5)
    return g1, g2


def test_spike_detection_and_mode():
    g1, g2 = build_pair(modify=True)
    core = GeDIGCore(spike_detection_mode="and", tau_s=0.0, tau_i=-1.0)  # force thresholds easy
    r = core.calculate(g_prev=g1, g_now=g2)
    assert r.spike is True


def test_spike_detection_or_mode():
    g1, g2 = build_pair(modify=True)
    core = GeDIGCore(spike_detection_mode="or", tau_s=0.5, tau_i=10.0)  # OR should still allow structural
    r = core.calculate(g_prev=g1, g_now=g2)
    assert r.structural_improvement > 0 or r.ig_z_score > 0
    assert r.spike is True


def test_spike_detection_threshold_mode():
    g1, g2 = build_pair(modify=False)
    core = GeDIGCore(spike_detection_mode="threshold", spike_threshold=-0.5)
    r = core.calculate(g_prev=g1, g_now=g2)
    # unchanged graph should not cross threshold
    assert r.spike is False


def test_presets_balanced():
    params = GeDIGPresets.BALANCED
    core = GeDIGCore(**{k:v for k,v in params.items() if k not in {"lambda_weight","mu"}}, lambda_weight=params["lambda_weight"], mu=params["mu"])
    g1, g2 = build_pair(modify=True)
    r = core.calculate(g_prev=g1, g_now=g2)
    assert isinstance(r.spike, bool)
