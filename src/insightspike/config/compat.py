"""
Compatibility helpers for legacy/dict configs.

Replaces legacy.compat_config functions with thin wrappers around the
Pydantic-based normalizer. Intended to remove direct imports of
`insightspike.legacy.*` while keeping call sites stable.
"""

from __future__ import annotations

from typing import Any

from .models import InsightSpikeConfig
from .normalizer import ConfigNormalizer


def detect_config_type(config: Any) -> str:
    """Return 'pydantic' for InsightSpikeConfig, else 'dict'."""
    return "pydantic" if isinstance(config, InsightSpikeConfig) else "dict"


def is_pydantic_config(config: Any) -> bool:
    """True if config is an InsightSpikeConfig instance."""
    return isinstance(config, InsightSpikeConfig)


def normalize(config: Any) -> InsightSpikeConfig:
    """Normalize input config (dict or Pydantic) into InsightSpikeConfig."""
    return ConfigNormalizer.normalize(config)


__all__ = ["detect_config_type", "is_pydantic_config", "normalize"]

