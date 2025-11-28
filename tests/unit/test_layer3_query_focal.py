"""Tests for query-focal handling stub."""

from __future__ import annotations

import os


def test_query_focal_handler_returns_neutral(monkeypatch):
    from insightspike.implementations.layers.layer3.analyzer_runner import handle_query_focal

    monkeypatch.setenv("INSIGHTSPIKE_QUERY_FOCAL_METRICS", "1")
    res = handle_query_focal(reasoner=None, documents=[], context={})
    assert res["metrics"]["delta_ged"] == 0.0
    assert res["spike_detected"] is False
    assert res["reasoning_quality"] == 0.5


def test_query_focal_handler_core(monkeypatch):
    import types
    import numpy as np
    from types import SimpleNamespace
    import insightspike.implementations.layers.layer3.analyzer_runner as ar

    # Dummy GeDIGCore
    class DummyHop:
        def __init__(self):
            self.gedig = 0.4

    class DummyRes:
        def __init__(self):
            self.delta_ged_norm = 1.0
            self.delta_h_norm = 0.2
            self.delta_sp_rel = 0.1
            self.gedig_value = 0.5
            self.hop_results = {0: DummyHop()}

    class DummyCore:
        def __init__(self, *a, **k):
            pass

        def calculate(self, **kwargs):
            return DummyRes()

    # Inject dummy core modules to avoid real imports
    class DummyDC:
        def __init__(self, *a, **k):
            pass

    ar._GeDIGCore = DummyCore
    ar._DistanceCache = DummyDC  # placeholder to avoid imports
    ar._pyg_to_networkx = lambda g: g

    cfg = SimpleNamespace(
        graph=SimpleNamespace(
            sp_beta=0.2, spike_ged_threshold=-0.5, spike_ig_threshold=0.2, conflict_threshold=0.5
        ),
        metrics=SimpleNamespace(query_radius=1),
    )
    reasoner = SimpleNamespace(
        config=cfg,
        graph_builder=SimpleNamespace(build_graph=lambda docs: SimpleNamespace(num_nodes=len(docs), edge_index=None, x=np.zeros((len(docs), 1)))),
        conflict_scorer=SimpleNamespace(calculate_conflict=lambda a, b, ctx: {"total": 0.0}),
        reward_calculator=SimpleNamespace(calculate_reward=lambda m, c: {"total": 0.0, "insight_reward": 0.0, "quality_bonus": 0.0}),
        graph_analyzer=SimpleNamespace(
            detect_spike=lambda m, c, t: False,
            assess_quality=lambda m, c: 0.5,
        ),
        previous_graph=SimpleNamespace(num_nodes=2),
    )

    res = ar.handle_query_focal(reasoner, documents=[{"t": "a"}], context={})
    assert res["metrics"]["sp_engine"] == "core"
    assert res["metrics"]["delta_ged"] == -1.0


def test_query_focal_handler_cached(monkeypatch):
    import types
    import numpy as np
    from types import SimpleNamespace
    import insightspike.implementations.layers.layer3.analyzer_runner as ar

    class DummyRes:
        def __init__(self):
            self.delta_ged_norm = 1.0
            self.delta_h_norm = 0.2
            self.gedig_value = 0.5
            self.hop_results = {0: SimpleNamespace(gedig=0.4)}

    class DummyCore:
        def __init__(self, *a, **k):
            pass

        def calculate(self, **kwargs):
            return DummyRes()

    class DummyDC:
        def __init__(self, mode="cached", pair_samples=None):
            pass

        def signature(self, *a, **k):
            return "sig"

        def estimate_sp_between_graphs(self, **kwargs):
            return 0.3

    ar._GeDIGCore = DummyCore
    ar._DistanceCache = DummyDC
    ar._pyg_to_networkx = lambda g: g

    cfg = SimpleNamespace(
        graph=SimpleNamespace(
            sp_beta=0.2,
            lambda_weight=1.0,
            sp_engine="cached",
            spike_ged_threshold=-0.5,
            spike_ig_threshold=0.2,
            conflict_threshold=0.5,
        ),
        metrics=SimpleNamespace(query_radius=1),
    )
    reasoner = SimpleNamespace(
        config=cfg,
        graph_builder=SimpleNamespace(build_graph=lambda docs: SimpleNamespace(num_nodes=len(docs), edge_index=None, x=np.zeros((len(docs), 1)))),
        conflict_scorer=SimpleNamespace(calculate_conflict=lambda a, b, ctx: {"total": 0.0}),
        reward_calculator=SimpleNamespace(calculate_reward=lambda m, c: {"total": 0.0, "insight_reward": 0.0, "quality_bonus": 0.0}),
        graph_analyzer=SimpleNamespace(
            detect_spike=lambda m, c, t: False,
            assess_quality=lambda m, c: 0.5,
        ),
        previous_graph=SimpleNamespace(num_nodes=2),
    )

    res = ar.handle_query_focal(reasoner, documents=[{"t": "a"}], context={})
    assert res["metrics"]["sp_engine"] == "cached"
    assert res["metrics"]["delta_sp"] == 0.3
