import sys
import types


class _StubHistory:
    def __init__(self, initial_query: str):
        self.states = [
            {"confidence": 0.1, "transformation_magnitude": 0.0, "insights": []}
        ]

    def add_state(self, state):
        self.states.append({"confidence": 0.2, "transformation_magnitude": 0.1, "insights": []})

    def get_current_state(self):
        return types.SimpleNamespace(
            text="q", embedding=None, stage="initial", confidence=0.1, absorbed_concepts=[], insights=[],
        )


class _StubTransformer:
    def __init__(self, *a, **k):
        pass

    def place_query_on_graph(self, question, G):
        return types.SimpleNamespace(
            text=question,
            embedding=None,
            stage="initial",
            confidence=0.1,
            absorbed_concepts=[],
            insights=[],
            connected_nodes=[],
            edge_weights={},
        )

    def transform_query(self, state, G, docs):
        return types.SimpleNamespace(
            text=state.text,
            embedding=None,
            stage="exploring",
            confidence=0.2,
            absorbed_concepts=[],
            insights=[],
            connected_nodes=[],
            edge_weights={},
            transformation_magnitude=0.1,
        )


def _install_stub_module():
    mod = types.ModuleType("insightspike.features.query_transformation")
    mod.QueryTransformer = _StubTransformer
    mod.QueryTransformationHistory = _StubHistory
    sys.modules[mod.__name__] = mod


def test_query_transform_smoke(monkeypatch):
    _install_stub_module()
    from insightspike.implementations.agents.configurable_agent import (
        AgentConfig,
        AgentMode,
        ConfigurableAgent,
    )

    cfg = AgentConfig.from_mode(AgentMode.QUERY_TRANSFORM)
    agent = ConfigurableAgent(cfg)
    result = agent.process_question("What is energy?")
    # Should produce a result without exceptions; history may be attached by best_result path
    assert result is not None
