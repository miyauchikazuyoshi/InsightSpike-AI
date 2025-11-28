"""Tests for the Layer3 lazy wrapper and lite stub."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _lite_env(monkeypatch):
    """Force lite/minimal mode for safety."""
    monkeypatch.setenv("INSIGHTSPIKE_LITE_MODE", "1")
    monkeypatch.setenv("INSIGHTSPIKE_MIN_IMPORT", "1")


def test_layer3_falls_back_to_stub(monkeypatch):
    """If the real class import fails, the wrapper should return the lite stub."""
    from insightspike.implementations.layers import layer3

    # Make _load_real_class raise to simulate missing deps
    monkeypatch.setattr(layer3, "_load_real_class", lambda: (_ for _ in ()).throw(ImportError("no torch")))

    inst = layer3.L3GraphReasoner(config={"dummy": True})
    from insightspike.implementations.layers.layer3.lite_stub import L3GraphReasonerLiteStub

    assert isinstance(inst, L3GraphReasonerLiteStub)
    res = inst.analyze_documents([{"text": "x"}])
    assert res["spike_detected"] is False
    assert res["metrics"]["delta_ged"] == 0.0


def test_layer3_wrapper_uses_real_when_available(monkeypatch):
    """If real class is provided, wrapper should instantiate it."""
    from insightspike.implementations.layers import layer3

    class DummyReal:
        def __init__(self, config=None):
            self.config = config
            self.created = True

    monkeypatch.setattr(layer3, "_load_real_class", lambda: DummyReal)

    inst = layer3.L3GraphReasoner(config={"ok": True})
    assert isinstance(inst, DummyReal)
    assert inst.config == {"ok": True}


def test_layer3_reexports_conflict_and_graph_builder():
    """Ensure ConflictScore/GraphBuilder are re-exported."""
    from insightspike.implementations.layers.layer3 import ConflictScore, GraphBuilder

    cs = ConflictScore(config={"graph": {"conflict_threshold": 0.5}})
    gb = GraphBuilder(config={"graph": {"similarity_threshold": 0.1}})
    assert hasattr(cs, "calculate_conflict")
    assert hasattr(gb, "build_graph")


def test_layer3_reexports_message_passing():
    """Ensure message passing stubs are re-exported and callable."""
    from insightspike.implementations.layers.layer3 import (
        EdgeReevaluator,
        MessagePassing,
    )

    mp = MessagePassing()
    er = EdgeReevaluator()
    # Stubs should be callable without raising
    assert mp.run() is None
    assert er.reevaluate() is None


def test_layer3_reexports_analysis_components():
    """Ensure GraphAnalyzer and RewardCalculator are re-exported and callable."""
    from insightspike.implementations.layers.layer3 import (
        GraphAnalyzer,
        RewardCalculator,
    )

    ga = GraphAnalyzer(config={})

    def dummy_delta_ged(a, b, **kwargs):
        return 0.1

    def dummy_delta_ig(a, b, **kwargs):
        return 0.2

    metrics = ga.calculate_metrics(current_graph=None, previous_graph=None, delta_ged_func=dummy_delta_ged, delta_ig_func=dummy_delta_ig)
    assert "delta_ged" in metrics

    rc = RewardCalculator(config={})
    reward = rc.calculate_reward({"delta_ged": 0.1, "delta_ig": 0.2}, {"total": 0.0})
    assert "total" in reward


def test_layer3_analysis_is_sourced_from_layer3_package(monkeypatch):
    """GraphAnalyzer/RewardCalculator should come from layer3.analysis."""
    import insightspike.implementations.layers.layer3.analysis as ana
    import insightspike.implementations.layers.layer3_graph_reasoner as lgr

    # Force lite/minimal path
    monkeypatch.setenv("INSIGHTSPIKE_LITE_MODE", "1")
    monkeypatch.setenv("INSIGHTSPIKE_MIN_IMPORT", "1")

    class DummyGA:
        def __init__(self, config=None):
            self.config = config

        def calculate_metrics(self, *a, **k):
            return {
                "delta_ged": 0.1,
                "delta_ig": 0.2,
                "delta_ged_norm": 0.1,
                "delta_sp": 0.0,
                "g0": 0.0,
                "gmin": 0.0,
            }

        def detect_spike(self, *a, **k):
            return False

        def assess_quality(self, *a, **k):
            return 0.0

    class DummyRC:
        def __init__(self, config=None):
            self.config = config

        def calculate_reward(self, metrics, conflicts):
            return {"total": metrics.get("delta_ged", 0) + metrics.get("delta_ig", 0)}

    # Reset lazy import flags and inject dummy components
    monkeypatch.setattr(lgr, "_COMPONENTS_LOADED", False)
    monkeypatch.setattr(lgr, "GraphAnalyzer", None)
    monkeypatch.setattr(lgr, "RewardCalculator", None)
    monkeypatch.setattr(ana, "GraphAnalyzer", DummyGA)
    monkeypatch.setattr(ana, "RewardCalculator", DummyRC)

    reasoner = lgr.L3GraphReasoner(config={"graph": {}, "embedding": {"dimension": 16}})
    assert isinstance(reasoner.graph_analyzer, DummyGA)
    assert isinstance(reasoner.reward_calculator, DummyRC)
