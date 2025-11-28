"""Lightweight stub for Layer3 graph reasoner.

Used when the real implementation cannot be imported (e.g., missing torch/PyG).
Provides a minimal shape-compatible API to keep callers from crashing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class L3GraphReasonerLiteStub:
    """No-op stub that reports disabled state."""

    enabled: bool = False

    def __init__(self, *args, **kwargs) -> None:
        self.current_graph = None

    def initialize(self) -> bool:
        return True

    def analyze_documents(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "graph": None,
            "metrics": {"delta_ged": 0.0, "delta_ig": 0.0},
            "conflicts": {"total": 0.0},
            "reward": {
                "insight_reward": 0.0,
                "quality_bonus": 0.0,
                "total": 0.0,
            },
            "reasoning_quality": 0.5,
            "spike_detected": False,
        }


__all__ = ["L3GraphReasonerLiteStub"]
