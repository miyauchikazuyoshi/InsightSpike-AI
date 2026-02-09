"""geDIG type definitions.

This module contains all dataclasses and enums used by geDIG.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, Set


class ProcessingMode(Enum):
    """Processing mode for geDIG calculation."""
    WAKE = "wake"
    SLEEP = "sleep"


class SpikeDetectionMode(Enum):
    """Spike detection mode."""
    THRESHOLD = "threshold"
    AND = "and"
    OR = "or"


@dataclass
class HopResult:
    """Result for a single hop in multi-hop geDIG calculation."""
    hop: int
    # Normalized GED (cost-based, positive), kept for inspection
    ged: float
    # Shannon-entropy based IG (variance reduction)
    ig: float
    # Per-hop geDIG value (cost - lambda*IG)
    gedig: float
    # Structural cost used for this hop (positive is worse); equals
    # base normalized GED at hop==0, optionally adjusted by SP gain for hop>0.
    struct_cost: float
    node_count: int
    edge_count: int
    sp: float = 0.0
    h_component: float = 0.0
    ged_raw: float = 0.0
    ged_den: float = 1.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    ig_delta: float = 0.0
    ig_den: float = 1.0
    variance_reduction: float = 0.0
    betti_1: int = 0

    @property
    def struct_term(self) -> float:
        """Deprecated alias returning negative structural improvement (legacy behaviour)."""
        return -self.struct_cost


@dataclass
class LinksetMetrics:
    """Metrics computed from linkset (candidate pool) for paper-aligned geDIG."""
    delta_ged_norm: float
    delta_h_norm: float
    delta_sp_rel: float
    gedig_value: float
    raw_ged: float
    ged_norm_den: float
    ig_norm_den: float
    entropy_before: float
    entropy_after: float
    ig_delta: float
    before_size: int
    after_size: int
    query_similarity: float
    # Diagnostics: positive-weight counts and top weights (before/after)
    pos_w_before: int = 0
    pos_w_after: int = 0
    topw_before: list[float] | None = None
    topw_after: list[float] | None = None


@dataclass
class GeDIGResult:
    """Complete result from geDIG calculation."""
    gedig_value: float
    ged_value: float
    ig_value: float
    raw_ged: float = 0.0
    ged_norm_den: float = 1.0
    ig_raw: float = 0.0
    ig_norm_den: float = 1.0
    ig_z_score: float = 0.0
    delta_ged_norm: float = 0.0
    delta_sp_rel: float = 0.0
    delta_h_norm: float = 0.0
    structural_cost: float = 0.0
    structural_improvement: float = 0.0
    information_integration: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    ig_delta: float = 0.0
    variance_reduction: float = 0.0
    hop0_reward: float = 0.0
    aggregate_reward: float = 0.0
    reward: float = 0.0
    hop_results: Optional[Dict[int, HopResult]] = None
    computation_time: float = 0.0
    focal_nodes: Optional[Set[str]] = None
    version: str = "refactor_phaseA"
    spike: bool = False
    linkset_metrics: Optional[LinksetMetrics] = None
    ged_min_proxy: float = 0.0
    betti_1_before: int = 0
    betti_1_after: int = 0
    delta_betti_1: int = 0

    @property
    def has_spike(self) -> bool:
        """Backward compatibility alias for spike attribute."""
        return self.spike

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


__all__ = [
    "ProcessingMode",
    "SpikeDetectionMode",
    "HopResult",
    "LinksetMetrics",
    "GeDIGResult",
]
