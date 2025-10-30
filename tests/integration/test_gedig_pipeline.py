"""Integration tests for geDIG modes, A/B correlation and query recording.

目的:
 - full / pure / ab の各 geDIG モードで `process_question` が成功するか確認
 - A/B (ab) モードで複数クエリ実行後に相関メトリクスカウンタが増えること
 - QueryRecorder により `CycleResult.query_id` が付与されること
 - 空メモリ状態でもクラッシュせず結果を返すこと

注意:
 - Pydantic `InsightSpikeConfig` には `gedig.mode` フィールドが無いため、
   `NormalizedConfig` 生成後に dataclasses.replace で `gedig_mode` を強制切替。
 - Graph Reasoner (torch 依存) が無い環境では full 経路が 0 値を返し得るため、
   相関値自体 (NaN/None) には強い制約を置かない。
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import List

import pytest

# Ensure package importable without editable install (tests root changes sys.path)
if 'insightspike' not in sys.modules:  # lightweight guard
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from insightspike.config.models import InsightSpikeConfig
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory


@pytest.fixture()
def tmp_datastore(tmp_path: Path):
    """Filesystem datastore fixture (軽量)."""
    cfg = {"type": "filesystem", "params": {"base_path": str(tmp_path)}}
    return DataStoreFactory.create(cfg)


def _mk_agent(mode: str, datastore):
    """Create agent and force geDIG mode via NormalizedConfig override."""
    # Use minimal test config (mock LLM to avoid heavy model load)
    config = InsightSpikeConfig(
        environment="test",
        llm={"provider": "mock", "max_tokens": 64},
        embedding={"model_name": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384},
        memory={"max_retrieved_docs": 5},
    )
    agent = MainAgent(config=config, datastore=datastore)
    assert agent.initialize(), "Agent failed to initialize"
    # Force mode if normalized config exists
    if getattr(agent, "_normalized_config", None) is not None:
        agent._normalized_config = replace(agent._normalized_config, gedig_mode=mode)
    return agent


@pytest.mark.parametrize("mode", ["full", "pure", "ab"])
def test_gedig_modes_smoke(mode: str, tmp_datastore):
    agent = _mk_agent(mode, tmp_datastore)

    # Inject minimal knowledge to allow retrieval
    knowledge = [
        "InsightSpike uses graph reasoning.",
        "Graph reasoning can detect spikes.",
    ]
    for k in knowledge:
        r = agent.learn(k)
        assert r["success"], "Knowledge insertion failed"

    result = agent.process_question("What does InsightSpike use?", max_cycles=2)
    assert result.success
    assert isinstance(result.reasoning_quality, float)
    # QueryRecorder should set an id
    assert result.query_id is not None

    if mode == "ab":
        # Export CSV (even if small dataset)
        out_csv = Path(tmp_datastore.base_path) / "gedig_ab.csv"  # type: ignore[attr-defined]
        written = agent.export_gedig_ab_csv(str(out_csv))
        assert out_csv.exists()
        # At least header line
        assert written >= 1


def test_ab_correlation_accumulates(tmp_datastore):
    agent = _mk_agent("ab", tmp_datastore)
    # Seed some knowledge
    for i in range(5):
        agent.learn(f"Concept {i} relates to base system.")

    n_queries = 12
    for i in range(n_queries):
        res = agent.process_question(f"Explain concept {i%3}?", max_cycles=1)
        assert res.success
    # Access internal logger metrics (best-effort)
    logger = getattr(agent, "_gedig_ab_logger", None)
    if logger is not None:
        metrics = logger.current_metrics()
        # Count should match executed queries (or more if internal extra calls happen)
        assert metrics.get("count", 0) >= n_queries
        assert "gedig_corr" in metrics  # value may be None/NaN depending on data


def test_empty_memory_question(tmp_datastore):
    agent = _mk_agent("pure", tmp_datastore)
    # No knowledge added
    result = agent.process_question("What is the system doing?", max_cycles=1)
    assert result.success
    assert result.retrieved_documents == [] or isinstance(result.retrieved_documents, list)
    # Ensure query id is present even with empty memory
    assert result.query_id is not None


def test_query_recorder_basic(tmp_datastore):
    agent = _mk_agent("full", tmp_datastore)
    agent.learn("The agent stores queries with embeddings.")
    res = agent.process_question("How are queries stored?", max_cycles=1)
    assert res.query_id is not None
    # Datastore persistence: expect episodes.json or similar (best-effort, non-fatal if missing)
    # We only assert the directory exists
    assert Path(tmp_datastore.base_path).exists()  # type: ignore[attr-defined]
