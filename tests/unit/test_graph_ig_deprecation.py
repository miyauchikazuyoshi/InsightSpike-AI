import logging
import networkx as nx

from insightspike.algorithms.gedig_core import GeDIGCore


def test_graph_ig_path_emits_deprecation_warning(caplog):
    g1 = nx.Graph(); g1.add_edges_from([(1, 2)])
    g2 = nx.Graph(); g2.add_edges_from([(1, 2), (2, 3)])

    core = GeDIGCore(enable_multihop=False)

    caplog.clear()
    caplog.set_level(logging.WARNING)
    _ = core.calculate(g_prev=g1, g_now=g2)

    msgs = [rec.message for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("[DEPRECATION] geDIG graph-IG path is in use" in m for m in msgs), (
        "Expected deprecation warning when graph-IG path is used without linkset_info"
    )

