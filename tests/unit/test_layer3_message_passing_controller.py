"""Tests for MessagePassingController behavior."""

from __future__ import annotations

import types

import pytest


@pytest.fixture(autouse=True)
def _lite_env(monkeypatch):
    monkeypatch.setenv("INSIGHTSPIKE_LITE_MODE", "1")
    monkeypatch.setenv("INSIGHTSPIKE_MIN_IMPORT", "1")


def test_controller_disabled_by_default():
    from insightspike.implementations.layers.layer3.message_passing_controller import (
        MessagePassingController,
    )

    ctrl = MessagePassingController(config={"graph": {}}, original_config={})
    ctrl.initialize()
    assert ctrl.message_passing_enabled is False
    assert ctrl.message_passing is None
    assert ctrl.edge_reevaluator is None


def test_controller_uses_basic_message_passing(monkeypatch):
    from insightspike.implementations.layers.layer3 import message_passing_controller as ctrl_mod

    # Dummy implementations to avoid heavy deps
    class DummyMP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyER:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ctrl_mod, "MessagePassing", DummyMP)
    monkeypatch.setattr(ctrl_mod, "EdgeReevaluator", DummyER)

    ctrl = ctrl_mod.MessagePassingController(
        config={"graph": {}},
        original_config={
            "graph": {
                "enable_message_passing": True,
                "message_passing": {"enable_batch_computation": False, "alpha": 0.5},
                "edge_reevaluation": {"similarity_threshold": 0.8},
            }
        },
    )
    ctrl.initialize()
    assert ctrl.message_passing_enabled is True
    assert isinstance(ctrl.message_passing, DummyMP)
    assert isinstance(ctrl.edge_reevaluator, DummyER)
    assert ctrl.message_passing.kwargs.get("alpha") == 0.5


def test_controller_uses_optimized_when_enabled(monkeypatch):
    from insightspike.implementations.layers.layer3 import message_passing_controller as ctrl_mod
    import sys

    # Dummy optimized implementation
    dummy_mod = types.ModuleType("insightspike.implementations.graph.message_passing_optimized")

    class DummyOptMP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    dummy_mod.OptimizedMessagePassing = DummyOptMP
    sys.modules["insightspike.implementations.graph.message_passing_optimized"] = dummy_mod

    # Use batch_computation=True to trigger optimized path
    ctrl = ctrl_mod.MessagePassingController(
        config={"graph": {}},
        original_config={
            "graph": {
                "enable_message_passing": True,
                "message_passing": {"enable_batch_computation": True, "max_hops": 2},
            }
        },
    )
    ctrl.initialize()
    assert ctrl.message_passing_enabled is True
    assert isinstance(ctrl.message_passing, DummyOptMP)
    assert ctrl.message_passing.kwargs.get("max_hops") == 2
