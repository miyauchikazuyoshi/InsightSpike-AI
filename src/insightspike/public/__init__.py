"""Public API surface for InsightSpike-AI.

This module exposes stable, user-facing entry points. Keep imports light and
free of internal implementation details.
"""

from __future__ import annotations

# Re-export quick start helpers as the primary public interface
from ..quick_start import create_agent, quick_demo  # noqa: F401

# Light wrappers around internal utilities (kept minimal and stable)
from ..config import load_config as _load_config  # noqa: F401
from ..config.summary import summarize_config as _summarize_config  # noqa: F401
from ..implementations.datastore.factory import DataStoreFactory as _DSFactory  # noqa: F401

def load_config(*args, **kwargs):
    """Public wrapper for configuration loading.

    Mirrors `insightspike.config.load_config` while keeping public import path stable.
    """
    return _load_config(*args, **kwargs)


def get_config_summary(config=None):
    """Return a lightweight diagnostic summary for a config.

    If `config` is None, loads the default config and summarizes it.
    """
    if config is None:
        try:
            cfg = _load_config()
        except Exception:
            cfg = None
        return _summarize_config(cfg) if cfg is not None else {}
    return _summarize_config(config)


def create_datastore(kind: str = "filesystem", **kwargs):
    """Create a simple, safe DataStore instance.

    Allowed kinds: "filesystem", "memory". Additional kwargs are passed to the
    underlying implementation (e.g., root path for filesystem).
    """
    allowed = {"filesystem", "memory"}
    k = (kind or "filesystem").lower()
    if k not in allowed:
        raise ValueError(f"Unsupported datastore kind: {kind}. Allowed: {sorted(allowed)}")
    return _DSFactory.create(k, **kwargs)


from .wrapper import InsightAppWrapper  # noqa: F401

__all__ = [
    "create_agent",
    "quick_demo",
    "load_config",
    "get_config_summary",
    "create_datastore",
    "InsightAppWrapper",
]
