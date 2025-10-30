"""
Graph utilities for InsightSpike.

Avoid importing heavy optional submodules at package import time so that
`insightspike.graph` remains importable in lightweight environments and during
test collection. Submodules can still be imported directly, e.g.
`from insightspike.graph.message_passing import MessagePassing`.
"""

__all__ = ["GraphConstructor"]

# Optional exposure of `GraphConstructor` when construction module is available.
try:  # pragma: no cover - best-effort exposure
    from .construction import GraphConstructor  # type: ignore
except Exception:
    # Keep package importable even if optional deps (e.g., sentence-transformers)
    # are missing; tests that need GraphConstructor should import the module
    # directly and manage their own optional dependencies.
    pass
