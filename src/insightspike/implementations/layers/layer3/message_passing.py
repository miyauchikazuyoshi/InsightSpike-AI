"""Message passing wrappers/stubs for Layer3 refactor."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from ...graph.message_passing import MessagePassing as _RealMP  # type: ignore
    from ...graph.edge_reevaluator import EdgeReevaluator as _RealER  # type: ignore
    MessagePassing = _RealMP  # type: ignore
    EdgeReevaluator = _RealER  # type: ignore
except Exception as exc:  # pragma: no cover - fallback path
    logger.warning("Layer3 message_passing: fallback stubs in use (%s)", exc)

    class MessagePassing:  # type: ignore
        """No-op stub for environments without graph message passing."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *args, **kwargs):
            return None

    class EdgeReevaluator:  # type: ignore
        """No-op stub for edge reevaluation."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def reevaluate(self, *args, **kwargs):
            return None


__all__ = ["MessagePassing", "EdgeReevaluator"]
