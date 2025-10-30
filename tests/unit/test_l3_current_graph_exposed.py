from typing import Any


class _MiniGraph:
    """Minimal graph-like object for L3 analyze_documents context.

    Provides attributes accessed by L3 (x, edge_index, num_nodes) without requiring torch.
    """

    def __init__(self, n: int = 2):
        import numpy as np

        self.num_nodes = n
        self.x = np.zeros((n, 4), dtype=float)
        # shape (2, E) like PyG, but only length is inspected indirectly; conflict scorer handles None prev graph
        self.edge_index = np.zeros((2, 0), dtype=int)


def test_l3_sets_current_graph_after_analysis():
    from insightspike.implementations.layers.layer3_graph_reasoner import (
        L3GraphReasoner,
    )

    l3 = L3GraphReasoner(config=None)
    ctx = {"graph": _MiniGraph(3)}
    # Use non-empty documents to avoid synthetic branch
    res = l3.analyze_documents([{"text": "a"}], ctx)
    assert isinstance(res, dict)
    assert getattr(l3, "current_graph", None) is ctx["graph"], "current_graph must mirror last analyzed graph"

