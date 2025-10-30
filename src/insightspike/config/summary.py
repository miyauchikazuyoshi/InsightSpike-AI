"""Config summary utilities.

Provides helper to extract a lightweight diagnostic summary including
auto-applied defaults (e.g. memory vector index parameters) without
requiring every caller to know dynamic attribute names.
"""
from __future__ import annotations

from typing import Any, Dict

def summarize_memory_config(mem_cfg: Any) -> Dict[str, Any]:
    """Return a dict summary of memory config including defaults_applied.

    Accepts either dataclass-style MemoryConfig (layer2) or pydantic model.
    Missing dynamic attributes are simply omitted.
    """
    summary: Dict[str, Any] = {
        "max_episodes": getattr(mem_cfg, "max_episodes", None),
        "embedding_dim": getattr(mem_cfg, "embedding_dim", None),
    }
    # Optional dynamic attributes
    for attr in ("faiss_index_type", "metric", "defaults_applied"):
        if hasattr(mem_cfg, attr):
            summary[attr] = getattr(mem_cfg, attr)
    return summary

def summarize_config(config: Any) -> Dict[str, Any]:
    """High-level summary for an InsightSpikeConfig or legacy wrapper.

    Only includes currently relevant remediation diagnostics.
    """
    mem = None
    if hasattr(config, "memory"):
        mem = summarize_memory_config(getattr(config, "memory"))
    elif hasattr(config, "config") and hasattr(config.config, "memory"):
        mem = summarize_memory_config(getattr(config.config, "memory"))
    return {
        "memory": mem,
    }

__all__ = ["summarize_memory_config", "summarize_config"]