"""Layer3 Graph Reasoner package (scaffold for refactor).

This keeps backward compatibility by lazily delegating to the existing
`layer3_graph_reasoner.L3GraphReasoner`. If heavy dependencies are missing,
it falls back to a lightweight stub that reports `enabled=False`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _load_real_class():
    """Import the existing Layer3 implementation lazily."""
    from ..layer3_graph_reasoner import L3GraphReasoner as _Real  # local import

    return _Real


class L3GraphReasoner:
    """Lazy proxy that instantiates the existing Layer3GraphReasoner.

    This wrapper avoids importing heavy dependencies at module import time and
    provides a stub fallback when imports fail (e.g., missing torch/PyG).
    """

    def __new__(cls, *args, **kwargs):
        try:
            real_cls = _load_real_class()
            return real_cls(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Layer3GraphReasoner import failed; using lite stub. reason=%s", exc
            )
            from .lite_stub import L3GraphReasonerLiteStub  # local import

            return L3GraphReasonerLiteStub(*args, **kwargs)


__all__ = [
    "L3GraphReasoner",
    # Re-exports for callers migrating to the new package
    "ConflictScore",
    "GraphBuilder",
    "GraphAnalyzer",
    "RewardCalculator",
    "MessagePassing",
    "EdgeReevaluator",
    "MessagePassingController",
    "MetricsController",
    "build_simple_gnn",
]

# Local imports placed after __all__ to avoid circulars during module init
from .conflict import ConflictScore  # noqa: E402
from .graph_builder import GraphBuilder  # noqa: E402
from .analysis import GraphAnalyzer, RewardCalculator  # noqa: E402
from .message_passing import MessagePassing, EdgeReevaluator  # noqa: E402
from .message_passing_controller import MessagePassingController  # noqa: E402
from .metrics_controller import MetricsController  # noqa: E402
from .gnn import build_simple_gnn  # noqa: E402
