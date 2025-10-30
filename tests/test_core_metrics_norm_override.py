import networkx as nx
from insightspike.algorithms.core.metrics import normalized_ged


def test_normalized_ged_with_override_simple():
    # Before: 2 nodes, 1 edge; After: 3 nodes, 2 edges (add 1 node + 1 edge)
    g1 = nx.Graph()
    g1.add_nodes_from([0, 1])
    g1.add_edge(0, 1)

    g2 = nx.Graph()
    g2.add_nodes_from([0, 1, 2])
    g2.add_edge(0, 1)
    g2.add_edge(1, 2)

    out = normalized_ged(g1, g2, node_cost=1.0, edge_cost=1.0, norm_override=10.0)
    # Raw GED should be 2 (1 node insert + 1 edge insert). With override=10 → 0.2
    assert out['raw_ged'] == 2.0
    assert abs(out['normalized_ged'] - 0.2) < 1e-9

