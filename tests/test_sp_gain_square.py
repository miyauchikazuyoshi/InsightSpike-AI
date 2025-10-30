import networkx as nx
from insightspike.algorithms.gedig_core import GeDIGCore


def build_square_graph():
    # 4-cycle: nodes 0-1-2-3-0
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return g


def build_square_with_center():
    # square + center 4 connected
    g = build_square_graph()
    g.add_node(4)
    g.add_edges_from([(4, 0), (4, 1), (4, 2), (4, 3)])
    return g


def test_sp_gain_on_square_center_add():
    g_before = build_square_graph()
    g_after = build_square_with_center()
    core = GeDIGCore()
    sp_rel = core._compute_sp_gain_norm(g_before, g_after, mode='relative')  # noqa: SLF001 (allow private for test)

    # Expected: L_before = 8/6 = 1.333..., L_after = 12/10 = 1.2 → gain_rel ≈ 0.1
    assert abs(sp_rel - 0.1) < 1e-6

