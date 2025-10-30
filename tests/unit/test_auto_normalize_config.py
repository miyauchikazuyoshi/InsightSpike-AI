import pytest

from insightspike.implementations.agents.main_agent import MainAgent


def test_auto_normalize_dict_config_basic():
    # Minimal dict (intentionally sparse) relying on auto-normalize defaults
    cfg = {
        'llm': {'provider': 'mock', 'max_tokens': 32},
        # memory intentionally partial to force default injection in normalizer
    }
    agent = MainAgent(config=cfg)
    assert agent.is_pydantic_config, "Config should be normalized to pydantic model"
    assert agent.was_auto_normalized() is True, "Should report auto-normalization occurred"
    # Normalized config should exist
    nc = agent._nc()
    assert nc is not None
    # embedding dimension default propagated
    assert nc.embedding_dim in (384, 512, 768)  # accept common dims (defensive)


def test_auto_normalize_keeps_pydantic():
    from insightspike.config.models import InsightSpikeConfig
    # Force mock provider to avoid local transformers dependency during unit test
    base = InsightSpikeConfig(llm={"provider": "mock", "model": "mock", "max_tokens": 32})
    agent = MainAgent(config=base)
    assert agent.is_pydantic_config
    assert agent.was_auto_normalized() is False  # original already pydantic


def test_process_question_with_auto_normalized():
    cfg = {'llm': {'provider': 'mock', 'max_tokens': 32}}
    agent = MainAgent(config=cfg)
    agent.initialize()
    res = agent.process_question("What is geDIG?", max_cycles=1)
    assert res.success
    # A minimal smoke: reasoning_quality bounded
    assert 0.0 <= res.reasoning_quality <= 1.0
