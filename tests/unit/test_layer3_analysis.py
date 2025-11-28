"""Tests for Layer3 analysis components (GraphAnalyzer/RewardCalculator)."""

from __future__ import annotations

import pytest


def test_graph_analyzer_defaults_with_none_prev():
    from insightspike.implementations.layers.layer3.analysis import GraphAnalyzer

    ga = GraphAnalyzer(config={})

    def _dummy_ged(prev, curr, **kwargs):
        return 0.0

    def _dummy_ig(prev, curr, **kwargs):
        return 0.0

    metrics = ga.calculate_metrics(current_graph=None, previous_graph=None, delta_ged_func=_dummy_ged, delta_ig_func=_dummy_ig)
    assert metrics["graph_size_current"] == 0
    assert metrics["graph_size_previous"] == 0
    assert metrics["delta_ged"] == 0.0
    assert metrics["delta_ig"] == 0.0


def test_reward_calculator_basic():
    from insightspike.implementations.layers.layer3.analysis import RewardCalculator

    rc = RewardCalculator(config={})
    reward = rc.calculate_reward({"delta_ged": 0.1, "delta_ig": 0.2, "graph_size_current": 1}, {"total": 0.0})
    assert "total" in reward
    assert reward["total"] != 0.0


def test_detect_spike_improvement_path():
    from insightspike.implementations.layers.layer3.analysis import GraphAnalyzer

    ga = GraphAnalyzer(config={"graph": {}})
    metrics = {"delta_ged": -0.6, "delta_ig": 0.3}
    conflicts = {"total": 0.1}
    thresholds = {"ged": -0.5, "ig": 0.2, "conflict": 0.5}
    assert ga.detect_spike(metrics, conflicts, thresholds) is True


def test_detect_spike_structural_growth_path():
    from insightspike.implementations.layers.layer3.analysis import GraphAnalyzer

    ga = GraphAnalyzer(config={"graph": {}})
    metrics = {"delta_ged": 200.0, "delta_ig": 0.0}
    conflicts = {"total": 0.0}
    thresholds = {"ged": -0.5, "ig": 0.2, "conflict": 0.5}
    # growth_factor path should trigger spike despite low IG
    assert ga.detect_spike(metrics, conflicts, thresholds) is True


def test_assess_quality_clamped_and_penalized():
    from insightspike.implementations.layers.layer3.analysis import GraphAnalyzer

    ga = GraphAnalyzer(config={"graph": {}})
    metrics = {"delta_ged": -2.0, "delta_ig": 2.0}
    conflicts = {"total": 0.8}
    quality = ga.assess_quality(metrics, conflicts)
    # (|ged| + ig)/2 = 2 -> clipped to 1, then penalty 0.8 -> 0.2 but lower bound is 0; clip upper to 1
    assert quality == pytest.approx(1.0, rel=1e-6, abs=1e-6)
