"""Executable contracts for the 2026-07 debt-repayment programme.

Tests start as strict xfails while a debt is still open.  The phase that fixes
the debt must remove the marker; an unexpected XPASS therefore cannot silently
leave the ledger stale.
"""

from types import SimpleNamespace

import pytest

from insightspike.config.models import InsightSpikeConfig


def test_create_agent_returns_an_initialized_agent(monkeypatch):
    import insightspike.quick_start as quick_start

    datastore = object()

    class DummyAgent:
        def __init__(self, config, datastore=None):
            self.config = config
            self.datastore = datastore
            self._initialized = False
            self.initialize_calls = 0

        def initialize(self):
            self.initialize_calls += 1
            self._initialized = True
            return True

    monkeypatch.setattr(
        quick_start,
        "load_config",
        lambda *args, **kwargs: InsightSpikeConfig(),
    )
    monkeypatch.setattr(quick_start, "MainAgent", DummyAgent)
    monkeypatch.setattr(
        quick_start,
        "_create_datastore_for_config",
        lambda config: datastore,
    )

    agent = quick_start.create_agent(provider="mock")

    assert agent._initialized is True
    assert agent.initialize_calls == 1
    assert agent.datastore is datastore


def test_legacy_cycle_supplies_a_valid_configuration(monkeypatch):
    from insightspike.implementations.agents import main_agent

    created = {}
    events = []

    class DummyAgent:
        def __init__(self, config, datastore=None):
            created["config"] = config

        def initialize(self):
            events.append("initialize")
            return True

        def add_document(self, text, c_value):
            events.append("add_document")
            return True

        def process_question(self, question, max_cycles=3, verbose=False):
            events.append("process_question")
            return {
                "response": "ok",
                "documents": [],
                "graph": None,
                "metrics": {},
                "success": True,
            }

    monkeypatch.setattr(main_agent, "MainAgent", DummyAgent)

    memory = SimpleNamespace(
        episodes=[SimpleNamespace(text="fact", c=0.8)],
    )
    result = main_agent.cycle(memory, "question")

    assert isinstance(created["config"], InsightSpikeConfig)
    assert events == ["initialize", "add_document", "process_question"]
    assert result["success"] is True


def test_full_mode_package_export_resolves_real_main_agent(monkeypatch):
    import insightspike
    from insightspike.implementations.agents.main_agent import (
        MainAgent as RealMainAgent,
    )

    monkeypatch.setattr(insightspike, "LITE_MODE", False)
    monkeypatch.delattr(insightspike, "MainAgent", raising=False)

    assert insightspike.__getattr__("MainAgent") is RealMainAgent


def test_legacy_cli_constructs_main_agent_with_loaded_config(monkeypatch):
    import insightspike.cli.legacy as legacy

    config = InsightSpikeConfig()
    created = {}

    class DummyAgent:
        def __init__(self, supplied_config):
            created["config"] = supplied_config

        def initialize(self):
            return True

        def get_stats(self):
            return {
                "initialized": True,
                "total_cycles": 0,
                "reasoning_history_length": 0,
                "average_quality": 0.0,
                "memory_stats": {},
            }

    monkeypatch.setattr(legacy, "get_config", lambda: config)
    monkeypatch.setattr(legacy, "MainAgent", DummyAgent)

    legacy.stats()

    assert created["config"] is config


def test_datastore_factory_accepts_config_model_in_memory_spelling():
    from insightspike.implementations.datastore.factory import DataStoreFactory
    from insightspike.implementations.datastore.memory_store import InMemoryDataStore

    store = DataStoreFactory.create("in_memory")

    assert isinstance(store, InMemoryDataStore)


def test_app_wrapper_uses_main_agent_state_persistence(monkeypatch, tmp_path):
    import insightspike.public.wrapper as wrapper_module

    class DummyAgent:
        def __init__(self):
            self.save_calls = 0
            self.load_calls = 0
            self.legacy_save_calls = 0
            self.legacy_load_calls = 0

        def save_state(self):
            self.save_calls += 1
            return True

        def load_state(self):
            self.load_calls += 1
            return True

        def save(self):
            self.legacy_save_calls += 1
            return True

        def load(self):
            self.legacy_load_calls += 1
            return True

    agent = DummyAgent()
    monkeypatch.setattr(wrapper_module, "create_agent", lambda **kwargs: agent)

    wrapper = wrapper_module.InsightAppWrapper(
        provider="mock",
        data_dir=str(tmp_path),
    )

    assert agent.load_calls == 1
    assert wrapper.save() is True
    assert agent.save_calls == 1
    assert agent.legacy_load_calls == 0
    assert agent.legacy_save_calls == 0


def test_app_wrapper_propagates_state_save_failure(monkeypatch, tmp_path):
    import insightspike.public.wrapper as wrapper_module

    agent = SimpleNamespace(
        load_state=lambda: True,
        save_state=lambda: False,
    )
    monkeypatch.setattr(wrapper_module, "create_agent", lambda **kwargs: agent)

    wrapper = wrapper_module.InsightAppWrapper(
        provider="mock",
        data_dir=str(tmp_path),
    )

    assert wrapper.save() is False
