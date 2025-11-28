"""Tests for MetricsController wrapper."""

from __future__ import annotations

import types


def test_metrics_controller_uses_selector(monkeypatch):
    import importlib
    import sys
    import types as _types

    # Prepare dummy selector module for import inside controller
    dummy = types.SimpleNamespace(
        delta_ged=lambda *a, **k: 1.23,
        delta_ig=lambda *a, **k: 4.56,
        get_algorithm_info=lambda: {"ged_algorithm": "dummy", "ig_algorithm": "dummy"},
    )

    class DummySelector:
        def __init__(self, config=None):
            pass

        def delta_ged(self, *a, **k):
            return dummy.delta_ged(*a, **k)

        def delta_ig(self, *a, **k):
            return dummy.delta_ig(*a, **k)

        def get_algorithm_info(self):
            return dummy.get_algorithm_info()

    sys.modules["insightspike.implementations.algorithms.metrics_selector"] = _types.SimpleNamespace(
        MetricsSelector=DummySelector
    )

    import insightspike.implementations.layers.layer3.metrics_controller as mc_mod

    monkeypatch.setattr(mc_mod, "logger", types.SimpleNamespace(warning=lambda *a, **k: None))
    importlib.reload(mc_mod)

    ctrl = mc_mod.MetricsController(config={})
    assert ctrl.delta_ged() == 1.23
    assert ctrl.delta_ig() == 4.56
    assert ctrl.info["ged_algorithm"] == "dummy"


def test_metrics_controller_stub(monkeypatch):
    import insightspike.implementations.layers.layer3.metrics_controller as mc_mod

    # Ensure the selector import fails to trigger stub path
    if "insightspike.implementations.algorithms.metrics_selector" in __import__("sys").modules:
        __import__("sys").modules.pop("insightspike.implementations.algorithms.metrics_selector")
    mc_mod.MetricsSelector = None  # ensure attribute exists
    monkeypatch.setattr(mc_mod, "logger", types.SimpleNamespace(warning=lambda *a, **k: None))

    ctrl = mc_mod.MetricsController(config={})
    assert ctrl.delta_ged() == 0.0
    assert ctrl.delta_ig() == 0.0
    assert ctrl.info["ged_algorithm"] == "stub"
