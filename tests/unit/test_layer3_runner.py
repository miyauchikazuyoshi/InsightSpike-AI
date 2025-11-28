"""Tests for analyzer_runner.run_analysis basic path."""

from __future__ import annotations

from types import SimpleNamespace

import insightspike.implementations.layers.layer3.analyzer_runner as ar


class DummyGraph:
    def __init__(self, n):
        self.num_nodes = n
        self.edge_index = None
        self.x = None


def test_run_analysis_basic():
    documents = [{"text": "a"}, {"text": "b"}]

    reasoner = SimpleNamespace(
        config=SimpleNamespace(
            graph=SimpleNamespace(
                conflict_threshold=0.5,
                spike_ged_threshold=-0.5,
                spike_ig_threshold=0.2,
            ),
            embedding=SimpleNamespace(dimension=2),
        ),
        graph_builder=SimpleNamespace(build_graph=lambda docs, incremental=False: DummyGraph(len(docs))),
        conflict_scorer=SimpleNamespace(calculate_conflict=lambda a, b, ctx: {"total": 0.0}),
        reward_calculator=SimpleNamespace(
            calculate_reward=lambda metrics, conflicts: {
                "total": metrics.get("delta_ged", 0.0) + metrics.get("delta_ig", 0.0),
                "insight_reward": 0.0,
                "quality_bonus": 0.0,
            }
        ),
        graph_analyzer=SimpleNamespace(
            detect_spike=lambda metrics, conflicts, thresholds: False,
            assess_quality=lambda metrics, conflicts: 0.5,
            calculate_metrics=lambda current_graph, previous_graph, delta_ged_func, delta_ig_func: {
                "delta_ged": delta_ged_func(previous_graph, current_graph),
                "delta_ig": delta_ig_func(previous_graph, current_graph),
                "delta_ged_norm": abs(delta_ged_func(previous_graph, current_graph)),
                "delta_sp": 0.0,
                "graph_size_current": getattr(current_graph, "num_nodes", 0),
                "graph_size_previous": getattr(previous_graph, "num_nodes", 0) if previous_graph is not None else 0,
            },
        ),
        delta_ged=lambda prev, curr, **k: 0.1,
        delta_ig=lambda prev, curr, **k: 0.2,
        previous_graph=None,
        message_passing_enabled=False,
        message_passing=None,
        edge_reevaluator=None,
        _original_config={},
    )

    res = ar.run_analysis(reasoner, documents, context={})
    assert res["metrics"]["delta_ged"] == 0.1
    assert res["metrics"]["delta_ig"] == 0.2
    assert res["spike_detected"] is False
    assert res["reasoning_quality"] == 0.5


def test_run_analysis_query_focal_core(monkeypatch):
    import insightspike.implementations.layers.layer3.analyzer_runner as ar
    import numpy as np

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

    # Inject dummy core classes to avoid real imports
    ar._GeDIGCore = DummyCore
    ar._DistanceCache = None
    ar._pyg_to_networkx = lambda g: g

    monkeypatch.setenv("INSIGHTSPIKE_QUERY_FOCAL_METRICS", "1")
    reasoner = SimpleNamespace(
        config=SimpleNamespace(
            graph=SimpleNamespace(
                sp_beta=0.2, lambda_weight=1.0, spike_ged_threshold=-0.5, spike_ig_threshold=0.2, conflict_threshold=0.5
            ),
            metrics=SimpleNamespace(query_radius=1),
            embedding=SimpleNamespace(dimension=2),
        ),
        graph_builder=SimpleNamespace(build_graph=lambda docs, incremental=False: DummyGraph(len(docs))),
        conflict_scorer=SimpleNamespace(calculate_conflict=lambda a, b, ctx: {"total": 0.0}),
        reward_calculator=SimpleNamespace(
            calculate_reward=lambda metrics, conflicts: {
                "total": metrics.get("delta_ged", 0.0) + metrics.get("delta_ig", 0.0),
                "insight_reward": 0.0,
                "quality_bonus": 0.0,
            }
        ),
        graph_analyzer=SimpleNamespace(
            detect_spike=lambda metrics, conflicts, thresholds: False,
            assess_quality=lambda metrics, conflicts: 0.5,
        ),
        delta_ged=lambda prev, curr, **k: 0.1,
        delta_ig=lambda prev, curr, **k: 0.2,
        previous_graph=DummyGraph(2),
        message_passing_enabled=False,
        message_passing=None,
        edge_reevaluator=None,
        _original_config={},
    )

    res = ar.run_analysis(reasoner, documents=[{"text": "a"}], context={})
    assert res["metrics"]["sp_engine"] == "core"
    assert res["metrics"]["delta_ged"] == -1.0


def test_run_analysis_query_focal_cached(monkeypatch):
    import insightspike.implementations.layers.layer3.analyzer_runner as ar
    import numpy as np

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
        def __init__(self, *a, **k):
            pass

        def signature(self, *a, **k):
            return "sig"

        def estimate_sp_between_graphs(self, **kwargs):
            return 0.3

    ar._GeDIGCore = DummyCore
    ar._DistanceCache = DummyDC
    ar._pyg_to_networkx = lambda g: g

    monkeypatch.setenv("INSIGHTSPIKE_QUERY_FOCAL_METRICS", "1")
    monkeypatch.setenv("INSIGHTSPIKE_SP_ENGINE", "cached")
    reasoner = SimpleNamespace(
        config=SimpleNamespace(
            graph=SimpleNamespace(
                sp_beta=0.2,
                lambda_weight=1.0,
                sp_engine="cached",
                spike_ged_threshold=-0.5,
                spike_ig_threshold=0.2,
                conflict_threshold=0.5,
            ),
            metrics=SimpleNamespace(query_radius=1),
            embedding=SimpleNamespace(dimension=2),
        ),
        graph_builder=SimpleNamespace(build_graph=lambda docs, incremental=False: DummyGraph(len(docs))),
        conflict_scorer=SimpleNamespace(calculate_conflict=lambda a, b, ctx: {"total": 0.0}),
        reward_calculator=SimpleNamespace(
            calculate_reward=lambda metrics, conflicts: {
                "total": metrics.get("delta_ged", 0.0) + metrics.get("delta_ig", 0.0),
                "insight_reward": 0.0,
                "quality_bonus": 0.0,
            }
        ),
        graph_analyzer=SimpleNamespace(
            detect_spike=lambda metrics, conflicts, thresholds: False,
            assess_quality=lambda metrics, conflicts: 0.5,
        ),
        delta_ged=lambda prev, curr, **k: 0.1,
        delta_ig=lambda prev, curr, **k: 0.2,
        previous_graph=DummyGraph(2),
        message_passing_enabled=False,
        message_passing=None,
        edge_reevaluator=None,
        _original_config={},
    )

    res = ar.run_analysis(reasoner, documents=[{"text": "a"}], context={})
    assert res["metrics"]["sp_engine"] == "cached"
    assert res["metrics"]["delta_sp"] == 0.3
