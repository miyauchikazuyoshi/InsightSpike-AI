import math
import sys
import types

import networkx as nx


def _ensure_stubs(monkeypatch):
    # Minimal torch stub
    torch_stub = types.SimpleNamespace(__version__="0.0")
    setattr(torch_stub, "tensor", lambda *a, **k: None)
    setattr(torch_stub, "float32", None)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    class Data:
        def __init__(self, x=None, edge_index=None):
            self.x = x
            self.edge_index = edge_index
            self.num_nodes = len(x) if x is not None else 0

    monkeypatch.setitem(sys.modules, "torch_geometric", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_geometric.data", types.SimpleNamespace(Data=Data))


def test_layer3_query_centric_metrics(monkeypatch):
    _ensure_stubs(monkeypatch)

    # Stub MetricsSelector so Layer3 can initialise without heavy deps
    import insightspike.algorithms.metrics_selector as ms

    class StubSelector:
        def __init__(self, config=None):
            self._info = {"ged_algorithm": "stub", "ig_algorithm": "stub"}

        def get_algorithm_info(self):
            return self._info

        def delta_ged(self, *args, **kwargs):
            return -0.2

        def delta_ig(self, *args, **kwargs):
            return 0.1

    monkeypatch.setattr(ms, "MetricsSelector", StubSelector)

    from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
    from types import SimpleNamespace

    metrics_cfg = SimpleNamespace(
        query_centric=True,
        query_topk_centers=2,
        query_radius=0,
        ig_denominator="legacy",
        use_local_normalization=False,
    )

    cfg = SimpleNamespace(
        graph=SimpleNamespace(
            similarity_threshold=0.2,
            metrics=metrics_cfg,
            spike_ged_threshold=-0.5,
            spike_ig_threshold=0.2,
            spike_conflict_threshold=0.5,
        ),
        metrics=metrics_cfg,
        embedding=SimpleNamespace(dimension=4),
    )

    layer = L3GraphReasoner(config=cfg)
    import pytest
    if getattr(layer, "enabled", True) is False:
        pytest.skip("Layer3GraphReasoner running in lite/min mode")

    def dummy_graph(documents, incremental=False):
        g = nx.Graph()
        for i, _ in enumerate(documents):
            g.add_node(i)
        if g.number_of_nodes() > 1:
            g.add_edge(0, 1)
        g.num_nodes = g.number_of_nodes()
        return g

    layer.graph_builder.build_graph = dummy_graph

    prev_graph = dummy_graph([{"text": "prev1"}, {"text": "prev2"}])
    layer.previous_graph = prev_graph

    docs = [{"text": "doc1"}, {"text": "doc2"}]
    context = {
        "candidate_selection": {
            "k_star": 2,
            "l1_candidates": 2,
            "log_k_star": math.log(3.0),
        }
    }

    result = layer.analyze_documents(docs, context)
    metrics = result["metrics"]

    assert "delta_h" in metrics
    assert "delta_sp" in metrics
    assert "g0" in metrics and "gmin" in metrics
    # query-centric path should call GeDIGCore and populate normalized metrics
    assert metrics["delta_ged_norm"] >= 0.0
