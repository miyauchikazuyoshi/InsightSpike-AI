from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Decision:
    mode: str  # 'maintain' | 'expand_hop' | 'expand_cands' | 'stop'
    params: Dict[str, Any]


class DecisionController:
    """Lightweight decision module for MainAgent.

    Current behavior is conservative: return 'maintain' unless clear signals indicate expansion.
    This class is intentionally simple to avoid behavior changes while scaffolding the interface.
    """

    def __init__(self, *, hop_cap: int = 3) -> None:
        self.hop_cap = int(max(0, hop_cap))

    def decide(self, metrics: Dict[str, Any], state: Dict[str, Any] | None = None) -> Decision:
        try:
            best_h = int(metrics.get('best_h', 0))
            delta_ig = float(metrics.get('delta_ig', 0.0))
            delta_ged = float(metrics.get('delta_ged', 0.0))
            # Simple heuristic (placeholder): if IG is positive and best_h>0, suggest expand_hop by 1
            if best_h > 0 and delta_ig > 0.0:
                next_h = min(best_h + 1, self.hop_cap)
                return Decision(mode='expand_hop', params={'next_h': next_h})
        except Exception:
            pass
        return Decision(mode='maintain', params={})

