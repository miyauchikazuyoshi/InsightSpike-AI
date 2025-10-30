"""geDIG Fallback Tracker (extracted from MainAgent)

Single-purpose, dependency-light tracker to record fallback events during
geDIG computation (pure/full/ab paths). Extraction enables isolated unit tests
without importing heavy MainAgent dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class GeDIGFallbackEvent:
    kind: str
    error: str
    count: int


@dataclass
class GeDIGFallbackTracker:
    events: List[GeDIGFallbackEvent] = field(default_factory=list)
    total: int = 0

    def record(self, kind: str, exc: Exception) -> None:
        try:
            evt = GeDIGFallbackEvent(kind=kind, error=repr(exc), count=self.total + 1)
            self.events.append(evt)
            self.total += 1
            logger.warning(
                "[gedig_fallback] kind=%s total=%s err=%s", kind, self.total, exc
            )
        except Exception:  # pragma: no cover
            pass

    def summary(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'kinds': {k: sum(1 for e in self.events if e.kind == k) for k in {e.kind for e in self.events}},
        }

__all__ = ["GeDIGFallbackTracker", "GeDIGFallbackEvent"]
