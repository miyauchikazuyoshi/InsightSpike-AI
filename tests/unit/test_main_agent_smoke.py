"""Smoke tests for MainAgent with heavy deps stubbed out."""

from __future__ import annotations

def test_main_agent_initializes_with_stubs(monkeypatch):
    monkeypatch.setenv("INSIGHTSPIKE_LITE_MODE", "1")
    monkeypatch.setenv("INSIGHTSPIKE_MIN_IMPORT", "1")
    # Avoid gating by env; rely on stubs instead
    monkeypatch.setenv("INSIGHTSPIKE_IMPORT_MAX_LAYER", "3")

    import insightspike.implementations.agents.main_agent as ma
    from insightspike.config.models import InsightSpikeConfig

    class DummyMem:
        def __init__(self, dim=0, config=None):
            self.dim = dim
            self.config = config

    class DummyL3:
        def __init__(self, config=None):
            self.config = config

    class DummyGMS:
        def __init__(self, config=None):
            self.config = config

    monkeypatch.setattr(ma, "Memory", DummyMem)
    monkeypatch.setattr(ma, "GRAPH_REASONER_AVAILABLE", True)
    monkeypatch.setattr(ma, "L3GraphReasoner", DummyL3)
    monkeypatch.setattr(ma, "GraphMemorySearch", DummyGMS)

    cfg = InsightSpikeConfig(
        embedding={"dimension": 2, "model_name": "dummy"},
        graph={"hop_limit": 1, "neighbor_threshold": 0.1, "path_decay": 0.5},
        memory={"episodic_memory_capacity": 1, "max_retrieved_docs": 1},
    )

    agent = ma.MainAgent(cfg, datastore=None)
    assert isinstance(agent.l2_memory, DummyMem)
    assert agent.l2_memory.dim == 2
    assert isinstance(agent.l3_graph, DummyL3)
    assert isinstance(agent.graph_memory_search, DummyGMS)
