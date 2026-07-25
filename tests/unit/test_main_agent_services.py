"""Contracts for the services extracted from the MainAgent facade."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from insightspike.implementations.agents.agent_config_access import (
    AgentConfigAccess,
)
from insightspike.implementations.agents.agent_lifecycle import (
    AgentLifecycle,
)
from insightspike.implementations.agents.agent_persistence import (
    AgentPersistence,
)
from insightspike.implementations.agents.cycle_result_aggregator import (
    CycleAggregationStats,
    CycleResultAggregator,
)


@dataclass
class _FakeCycle:
    question: str
    retrieved_documents: List[Dict[str, Any]]
    graph_analysis: Dict[str, Any]
    response: str
    reasoning_quality: float
    spike_detected: bool
    error_state: Dict[str, Any]
    cycle_number: int
    success: bool = True
    query_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "quality": self.reasoning_quality,
        }


def _cycle(
    name: str,
    quality: float,
    *,
    spike: bool = False,
    success: bool = True,
) -> _FakeCycle:
    return _FakeCycle(
        question="question",
        retrieved_documents=[{"name": name}],
        graph_analysis={"source": name, "nested": {"kept": True}},
        response=name,
        reasoning_quality=quality,
        spike_detected=spike,
        error_state={},
        cycle_number=1,
        success=success,
        query_id=f"source-{name}",
    )


def test_cycle_aggregator_preserves_selection_and_metadata_contract() -> None:
    lower_spike = _cycle("spike", 0.4, spike=True)
    best = _cycle("best", 0.9, success=False)
    input_results = [lower_spike, best]
    original_graph = best.graph_analysis

    aggregation = CycleResultAggregator(_FakeCycle).aggregate(
        input_results,
        converged=True,
        include_history=True,
        stats=CycleAggregationStats(
            memory_episodes=7,
            total_processed=3,
        ),
    )

    assert aggregation.selected is best
    assert aggregation.result is not best
    assert aggregation.result.question == best.question
    assert aggregation.result.retrieved_documents is best.retrieved_documents
    assert aggregation.result.response == "best"
    assert aggregation.result.reasoning_quality == best.reasoning_quality
    assert aggregation.result.spike_detected is best.spike_detected
    assert aggregation.result.error_state is best.error_state
    assert aggregation.result.cycle_number == best.cycle_number
    assert aggregation.result.success is False
    assert aggregation.result.query_id is None
    assert aggregation.result.graph_analysis == {
        "source": "best",
        "nested": {"kept": True},
        "total_cycles": 2,
        "converged": True,
        "cycle_history": [
            lower_spike.to_dict(),
            best.to_dict(),
        ],
        "agent_stats": {
            "memory_episodes": 7,
            "total_processed": 3,
        },
    }
    assert aggregation.result.graph_analysis is not original_graph
    assert (
        aggregation.result.graph_analysis["nested"]
        is original_graph["nested"]
    )
    assert "total_cycles" not in original_graph
    assert input_results == [lower_spike, best]


def test_cycle_aggregator_keeps_first_result_on_quality_tie() -> None:
    first = _cycle("first", 0.8)
    second = _cycle("second", 0.8, spike=True)

    aggregation = CycleResultAggregator(_FakeCycle).aggregate(
        [first, second],
        converged=False,
        include_history=False,
        stats=CycleAggregationStats(0, 0),
    )

    assert aggregation.selected is first
    assert aggregation.result.graph_analysis["cycle_history"] == []


def test_cycle_aggregator_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CycleResultAggregator(_FakeCycle).aggregate(
            [],
            converged=False,
            include_history=False,
            stats=CycleAggregationStats(0, 0),
        )


def test_lifecycle_initializes_components_in_observable_order() -> None:
    events: List[str] = []

    class Provider:
        initialized = False

        def initialize(self) -> bool:
            events.append("provider")
            self.initialized = True
            return True

    class Memory:
        def load(self) -> bool:
            events.append("memory")
            return False

    class Monitor:
        def reset(self) -> None:
            events.append("monitor")

    provider = Provider()
    lifecycle = AgentLifecycle(logger=logging.getLogger(__name__))
    resolved = lifecycle.resolve_llm(
        llm=None,
        llm_factory=lambda: provider,
    )
    result = lifecycle.initialize(
        llm=resolved,
        memory=Memory(),
        datastore=None,
        lite_mode=False,
        error_monitor=Monitor(),
    )

    assert result.initialized is True
    assert result.llm is provider
    assert events == ["provider", "memory", "monitor"]


def test_lifecycle_failure_is_retryable_and_does_not_reset_monitor() -> None:
    monitor = Mock()
    provider = Mock(initialized=False)
    provider.initialize.return_value = False
    lifecycle = AgentLifecycle(logger=logging.getLogger(__name__))

    failed = lifecycle.initialize(
        llm=provider,
        memory=Mock(),
        datastore=object(),
        lite_mode=True,
        error_monitor=monitor,
    )
    provider.initialize.return_value = True
    succeeded = lifecycle.initialize(
        llm=failed.llm,
        memory=Mock(),
        datastore=object(),
        lite_mode=True,
        error_monitor=monitor,
    )

    assert failed.initialized is False
    assert succeeded.initialized is True
    assert provider.initialize.call_count == 2
    monitor.reset.assert_called_once_with()


def test_lifecycle_repeats_full_mode_memory_load_and_monitor_reset() -> None:
    lifecycle = AgentLifecycle(logger=logging.getLogger(__name__))
    provider = SimpleNamespace(initialized=True)
    memory = SimpleNamespace(load=Mock(return_value=False))
    monitor = SimpleNamespace(reset=Mock())

    for _ in range(2):
        result = lifecycle.initialize(
            llm=provider,
            memory=memory,
            datastore=None,
            lite_mode=False,
            error_monitor=monitor,
        )
        assert result.initialized is True

    assert memory.load.call_count == 2
    assert monitor.reset.call_count == 2


def test_persistence_rejection_stops_before_graph_write() -> None:
    datastore = Mock()
    graph = SimpleNamespace(previous_graph={"nodes": []})
    replace = Mock(return_value=False)

    result = AgentPersistence(
        logger=logging.getLogger(__name__)
    ).save_datastore_state(
        datastore=datastore,
        memory=SimpleNamespace(episodes=[object()]),
        graph_layer=graph,
        episode_encoder=lambda episode: {"text": "episode"},
        replace_episode_snapshot=replace,
    )

    assert result is False
    datastore.save_graph.assert_not_called()


def test_live_persistence_attempts_both_namespaces() -> None:
    calls: List[str] = []

    def replace(records, *, namespace):
        calls.append(namespace)
        return namespace == "agent_state"

    outcomes = AgentPersistence(
        logger=logging.getLogger(__name__)
    ).persist_live_episode_snapshots(
        episodes=[SimpleNamespace(text="fact")],
        episode_encoder=lambda episode: {"text": episode.text},
        replace_episode_snapshot=replace,
    )

    assert calls == ["default", "agent_state"]
    assert outcomes == {"default": False, "agent_state": True}


def test_persistence_falls_back_to_legacy_save_episodes() -> None:
    store = SimpleNamespace(
        save_episodes=Mock(return_value=True),
    )

    result = AgentPersistence(
        logger=logging.getLogger(__name__)
    ).replace_episode_snapshot(
        store,
        [{"text": "fact"}],
        namespace="agent_state",
    )

    assert result is True
    store.save_episodes.assert_called_once_with(
        [{"text": "fact"}],
        namespace="agent_state",
    )


@pytest.mark.parametrize(
    ("graph_save_result", "expected"),
    [
        (False, False),
        (None, True),
    ],
)
def test_persistence_preserves_graph_save_return_contract(
    graph_save_result,
    expected,
) -> None:
    datastore = Mock()
    datastore.save_graph.return_value = graph_save_result

    result = AgentPersistence(
        logger=logging.getLogger(__name__)
    ).save_datastore_state(
        datastore=datastore,
        memory=None,
        graph_layer=SimpleNamespace(
            previous_graph={"nodes": []}
        ),
        episode_encoder=lambda episode: {},
        replace_episode_snapshot=Mock(),
    )

    assert result is expected


def test_persistence_empty_load_clears_memory_and_rebuilds_index() -> None:
    datastore = Mock()
    datastore.load_episodes.return_value = []
    memory = SimpleNamespace(
        episodes=[object()],
        _rebuild_index=Mock(),
    )

    result = AgentPersistence(
        logger=logging.getLogger(__name__)
    ).load_datastore_state(
        datastore=datastore,
        memory=memory,
        graph_layer=None,
        episode_decoder=lambda record: record,
    )

    assert result is True
    assert memory.episodes == []
    memory._rebuild_index.assert_called_once_with()


def test_main_agent_persistence_facade_preserves_legacy_patch_points() -> None:
    from insightspike.implementations.agents.main_agent import MainAgent

    agent = MainAgent.__new__(MainAgent)
    agent.datastore = None
    agent._legacy_save_state = Mock(return_value=True)
    agent._legacy_load_state = Mock(return_value=False)

    assert agent.save_state() is True
    assert agent.load_state() is False
    agent._legacy_save_state.assert_called_once_with()
    agent._legacy_load_state.assert_called_once_with()


def test_main_agent_datastore_save_uses_snapshot_patch_point() -> None:
    from insightspike.implementations.agents.main_agent import MainAgent

    agent = MainAgent.__new__(MainAgent)
    agent.datastore = object()
    agent.l2_memory = SimpleNamespace(
        episodes=[
            {
                "text": "fact",
                "vec": [1.0, 0.0],
            }
        ]
    )
    agent.l3_graph = None
    agent._replace_episode_snapshot = Mock(return_value=True)

    assert agent.save_state() is True
    agent._replace_episode_snapshot.assert_called_once()


def test_config_access_reads_current_values_and_normalizes_thresholds() -> None:
    config = SimpleNamespace(
        graph=object(),
        metrics=SimpleNamespace(
            theta_cand=0.2,
            theta_link=0.6,
            candidate_cap=0,
            top_m="invalid",
            ig_denominator="LOCAL",
            use_local_normalization=True,
        ),
    )
    access = AgentConfigAccess(logger=logging.getLogger(__name__))

    params = access.two_threshold_params(config)

    assert access.section(config, "graph") is config.graph
    assert access.value(config, "metrics.theta_cand") == 0.2
    assert access.value(config, "missing.value", "fallback") == (
        "fallback"
    )
    assert params == {
        "theta_cand": 0.6,
        "theta_link": 0.2,
        "k_cap": 1,
        "top_m": None,
        "ig_denominator": "local",
        "use_local_normalization": True,
    }


def test_main_agent_lifecycle_facade_keeps_patch_point_and_repeat_behavior(
    monkeypatch,
) -> None:
    from insightspike.implementations.agents import main_agent

    events: List[str] = []

    class Provider:
        initialized = False

        def initialize(self) -> bool:
            events.append("provider")
            self.initialized = True
            return True

    provider = Provider()
    monkeypatch.setattr(
        main_agent,
        "get_llm_provider",
        lambda config, safe_mode=False: provider,
    )
    agent = main_agent.MainAgent.__new__(main_agent.MainAgent)
    agent._initialized = False
    agent.config = object()
    agent._lite_mode = True
    agent.l4_llm = None
    agent.l2_memory = Mock()
    agent.datastore = object()
    agent.l1_error_monitor = Mock()

    assert agent.initialize() is True
    assert agent.initialize() is True

    assert agent.l4_llm is provider
    assert events == ["provider"]
    assert agent.l1_error_monitor.reset.call_count == 2


def test_main_agent_publishes_llm_before_later_failure_and_reuses_it(
    monkeypatch,
) -> None:
    from insightspike.implementations.agents import main_agent

    factory_calls: List[object] = []

    class Provider:
        initialized = False

        def initialize(self) -> bool:
            self.initialized = True
            return True

    class FailingOnceMonitor:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("reset failed")

    provider = Provider()

    def build_provider(config, safe_mode=False):
        factory_calls.append(config)
        return provider

    monkeypatch.setattr(
        main_agent,
        "get_llm_provider",
        build_provider,
    )
    agent = main_agent.MainAgent.__new__(main_agent.MainAgent)
    agent._initialized = False
    agent.config = object()
    agent._lite_mode = True
    agent.l4_llm = None
    agent.l2_memory = Mock()
    agent.datastore = object()
    agent.l1_error_monitor = FailingOnceMonitor()

    assert agent.initialize() is False
    assert agent.l4_llm is provider
    assert agent.initialize() is True

    assert factory_calls == [agent.config]
    assert agent.l4_llm is provider
    assert agent.l1_error_monitor.calls == 2


def test_main_agent_repeat_failure_preserves_prior_initialized_state(
    monkeypatch,
) -> None:
    from insightspike.implementations.agents import main_agent

    provider = SimpleNamespace(initialized=True)
    monitor = Mock()
    monitor.reset.side_effect = [None, RuntimeError("reset failed")]
    monkeypatch.setattr(
        main_agent,
        "get_llm_provider",
        lambda config, safe_mode=False: provider,
    )
    agent = main_agent.MainAgent.__new__(main_agent.MainAgent)
    agent._initialized = False
    agent.config = object()
    agent._lite_mode = True
    agent.l4_llm = None
    agent.l2_memory = Mock()
    agent.datastore = object()
    agent.l1_error_monitor = monitor

    assert agent.initialize() is True
    assert agent.initialize() is False

    assert agent.initialized is True
    assert agent.l4_llm is provider


def test_real_provider_error_uses_canonical_lifecycle_fallback(
    monkeypatch,
) -> None:
    from insightspike.implementations.agents import main_agent
    from insightspike.implementations.layers import (
        layer4_llm_interface,
    )

    monkeypatch.setattr(
        layer4_llm_interface,
        "get_llm_provider",
        Mock(side_effect=RuntimeError("provider unavailable")),
    )
    agent = main_agent.MainAgent.__new__(main_agent.MainAgent)
    agent._initialized = False
    agent.config = object()
    agent._lite_mode = True
    agent.l4_llm = None
    agent.l2_memory = Mock()
    agent.datastore = object()
    agent.l1_error_monitor = Mock()

    assert agent.initialize() is True
    assert agent.l4_llm.generate()["text"] == (
        "[fallback-llm-response]"
    )
    assert agent.l4_llm.generate_response()["response"] == (
        "[fallback-llm-response]"
    )
    assert agent.l4_llm.generate_response_detailed()["response"] == (
        "[fallback-llm-response]"
    )


def test_main_agent_normalized_config_refreshes_and_honors_override() -> None:
    from insightspike.implementations.agents.main_agent import MainAgent

    config = {
        "graph": {
            "similarity_threshold": 0.31,
            "hop_limit": 2,
        },
        "gedig": {"mode": "full"},
    }
    agent = MainAgent.__new__(MainAgent)
    agent.config = config
    initial = agent._get_config_access().normalized(config)
    agent._normalized_config = initial
    agent._generated_normalized_config = initial

    config["graph"]["similarity_threshold"] = 0.73
    refreshed = agent._nc()

    assert refreshed is not initial
    assert refreshed.similarity_threshold == pytest.approx(0.73)
    assert agent._get_config_snapshot()["similarity_threshold"] == (
        pytest.approx(0.73)
    )

    override = replace(refreshed, gedig_mode="ab")
    agent._normalized_config = override
    config["graph"]["similarity_threshold"] = 0.91

    overlaid = agent._nc()
    assert overlaid.gedig_mode == "ab"
    assert overlaid.similarity_threshold == pytest.approx(0.91)
    assert agent._get_config_snapshot()["similarity_threshold"] == (
        pytest.approx(0.91)
    )


def test_main_agent_compile_facade_and_public_result_identity() -> None:
    from insightspike.implementations.agents import (
        CycleResult as ExportedCycleResult,
    )
    from insightspike.implementations.agents.main_agent import (
        CycleResult,
        MainAgent,
    )

    agent = MainAgent.__new__(MainAgent)
    agent.l2_memory = SimpleNamespace(
        get_memory_stats=lambda: {"total_episodes": 4}
    )
    agent.reasoning_history = [{"previous": True}]
    agent.enable_learning = False
    first = CycleResult(
        question="q",
        retrieved_documents=[],
        graph_analysis={},
        response="first",
        reasoning_quality=0.2,
        spike_detected=False,
        error_state={},
        cycle_number=1,
    )
    best = CycleResult(
        question="q",
        retrieved_documents=[],
        graph_analysis={},
        response="best",
        reasoning_quality=0.9,
        spike_detected=False,
        error_state={},
        cycle_number=2,
    )

    result = agent._compile_results(
        [first, best],
        converged=True,
        verbose=False,
        question="q",
    )

    assert ExportedCycleResult is CycleResult
    assert isinstance(result, CycleResult)
    assert result.response == "best"
    assert result.graph_analysis["agent_stats"] == {
        "memory_episodes": 4,
        "total_processed": 1,
    }


def test_main_agent_learning_hook_runs_before_result_materialization() -> None:
    from insightspike.implementations.agents.main_agent import (
        CycleResult,
        MainAgent,
    )

    events: List[str] = []

    class PatternLogger:
        def __init__(self):
            self.patterns = []
            self.context = None

        def log_pattern(self, **kwargs):
            events.append("log")
            self.context = kwargs["context"]
            kwargs["result"].response = "mutated-by-hook"
            self.patterns.append(kwargs)

    class Optimizer:
        def decay_exploration(self):
            events.append("decay")

    agent = MainAgent.__new__(MainAgent)
    agent.l2_memory = SimpleNamespace(
        get_memory_stats=lambda: {"total_episodes": 2}
    )
    agent.reasoning_history = []
    agent.enable_learning = True
    agent.pattern_logger = PatternLogger()
    agent.strategy_optimizer = Optimizer()
    agent._get_config_snapshot = lambda: {"snapshot": True}
    selected = CycleResult(
        question="q",
        retrieved_documents=[],
        graph_analysis={"original": True},
        response="before-hook",
        reasoning_quality=0.9,
        spike_detected=False,
        error_state={},
        cycle_number=1,
    )

    result = agent._compile_results(
        [selected],
        converged=False,
        verbose=False,
        question="q",
    )

    assert events == ["log", "decay"]
    assert result.response == "mutated-by-hook"
    assert agent.pattern_logger.context["graph_analysis"][
        "total_cycles"
    ] == 1


def test_main_agent_empty_compile_preserves_legacy_error_shape() -> None:
    from insightspike.implementations.agents.main_agent import MainAgent

    agent = MainAgent.__new__(MainAgent)
    result = agent._compile_results(
        [],
        converged=False,
        verbose=False,
        question="ignored",
    )

    assert result.question == ""
    assert result.success is False
    assert result.cycle_number == 0
