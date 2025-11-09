import pytest

import insightspike.quick_start as quick_start
from insightspike.config import InsightSpikeConfig


@pytest.fixture
def stub_agent_env(monkeypatch):
    """Provide a lightweight MainAgent + config loader for quick_start tests."""

    def fake_load_config(*_, **__):
        return InsightSpikeConfig()

    class DummyAgent:
        def __init__(self, config):
            self.config = config
            self.initialized = True

    monkeypatch.setattr(quick_start, "load_config", fake_load_config)
    monkeypatch.setattr(quick_start, "MainAgent", DummyAgent)
    return DummyAgent


def test_create_agent_applies_nested_overrides(stub_agent_env):
    agent = quick_start.create_agent(
        provider="anthropic",
        llm__temperature=0.7,
        processing__max_cycles=5,
        model="custom-mini",
    )

    assert agent.config.llm.provider == "anthropic"
    assert agent.config.llm.temperature == 0.7
    assert agent.config.processing.max_cycles == 5
    assert agent.config.llm.model == "custom-mini"


def test_create_agent_invalid_override_raises(stub_agent_env):
    with pytest.raises(ValueError):
        quick_start.create_agent(provider="mock", unknown__field=1)


def test_local_provider_without_cuda_falls_back_to_cpu_model(monkeypatch, stub_agent_env):
    monkeypatch.setattr(quick_start, "_has_cuda_support", lambda: False)

    agent = quick_start.create_agent(provider="local")

    assert agent.config.llm.provider == "local"
    assert agent.config.llm.model == "google/flan-t5-small"
    assert agent.config.llm.device == "cpu"
