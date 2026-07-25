"""Protocol definitions for the unified geDIG framework.

Uses structural typing (Protocol) so existing classes can satisfy
these interfaces without modification — critical for gradual migration.

Three experiments map to these protocols:
  - Maze:        NetworkX spatial graph → GraphSnapshot
  - RAG/BRIGHT:  NetworkX hetero graph  → GraphSnapshot
  - Transformer: PyTorch attention tensor → GraphSnapshot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Set, runtime_checkable


# ─── Result Types ────────────────────────────────────────────────

@dataclass
class FEvalResult:
    """Universal F-eval output.

    F = ΔEPC - λ(ΔH + γΔB)

    All three experiments produce this same structure.
    """

    f_value: float          # F score (lower = more stable change)
    delta_epc: float        # Structural change cost (GED / edge density)
    delta_h: float          # Entropy change (ordering signal)
    delta_b: float          # Structure potential change (SP or β₁)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AGDGResult:
    """Deprecated compatibility result for historical edge labels.

    These fields describe a percentile/threshold partition of edge scores,
    not the Attention Gate / Decision Gate event pair. New code uses
    :class:`EdgePartitionResult`.
    """

    n_ag: int = 0
    n_dg: int = 0
    threshold: float = 0.0
    ag_edges: list = field(default_factory=list)
    dg_edges: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgePartitionResult:
    """Exhaustive partition of edge IDs by score band.

    Percentile partitioners use only the lower and higher bands. A
    two-threshold partitioner also records the deadband between its
    thresholds instead of silently dropping those edges.
    """

    threshold: float = 0.0
    low_score_edges: list = field(default_factory=list)
    high_score_edges: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Kept after ``metadata`` to preserve the original positional field order.
    middle_score_edges: list = field(default_factory=list)

    @property
    def n_low_score(self) -> int:
        return len(self.low_score_edges)

    @property
    def n_high_score(self) -> int:
        return len(self.high_score_edges)

    @property
    def n_middle_score(self) -> int:
        return len(self.middle_score_edges)


@dataclass(frozen=True)
class TwoStageGateDecision:
    """One Attention-Gate / Decision-Gate event decision."""

    attention_gate_fired: bool
    decision_gate_fired: bool
    hop0_score: float
    best_multihop_score: float
    best_hop: int

    def as_legacy_dict(self) -> Dict[str, Any]:
        """Return the historical maze-adapter mapping."""

        return {
            "ag_fire": self.attention_gate_fired,
            "dg_fire": self.decision_gate_fired,
            "best_hop": self.best_hop,
            "g0": self.hop0_score,
            "gmin_mh": self.best_multihop_score,
        }


# ─── Graph Snapshot Protocol ─────────────────────────────────────

@runtime_checkable
class GraphSnapshot(Protocol):
    """Backend-agnostic graph state for before/after comparison.

    Implementations:
      - NxGraphSnapshot: wraps nx.Graph (maze, RAG)
      - TorchGraphSnapshot: wraps attention tensor (transformer)
    """

    def node_count(self) -> int:
        """Number of nodes in the graph."""
        ...

    def edge_count(self) -> int:
        """Number of edges (or soft edge sum for continuous graphs)."""
        ...

    def edge_set(self) -> Set:
        """Set of edges for discrete graphs. Empty for continuous."""
        ...


# ─── Component Protocols ─────────────────────────────────────────

@runtime_checkable
class EPCComputer(Protocol):
    """Computes EPC (Edit Path Cost / Graph Edit Distance) between states.

    Maze:        normalized_ged() — set-difference of edges/nodes
    RAG:         _local_ged() — edge set symmetric difference
    Transformer: soft edge density change via sigmoid
    """

    def compute(self, before: GraphSnapshot, after: GraphSnapshot) -> float:
        ...


@runtime_checkable
class EntropyComputer(Protocol):
    """Computes entropy change between states.

    Maze:        entropy_ig() — histogram-based Shannon entropy of features
    RAG:         _local_entropy() — feature distribution entropy
    Transformer: normalized Shannon entropy of attention weights
    """

    def compute(self, before: GraphSnapshot, after: GraphSnapshot) -> float:
        ...


@runtime_checkable
class StructurePotentialComputer(Protocol):
    """Computes structure potential change (SP or β₁).

    The third component of F-eval is pluggable. Relative shortest-path gain
    is the established/default form; β₁ (Betti number = cycle count) is an
    in-progress topological generalization selected explicitly by adapters.

    Maze:        compute_sp_gain_norm() — sampled pair shortest-path gain
    RAG:         _local_sp_gain() or β₁ = E - V + C
    Transformer: matrix-power reachability OR Laplacian eigenvalue β₁
    """

    def compute(self, before: GraphSnapshot, after: GraphSnapshot) -> float:
        ...


# ─── Message Passing Protocol ────────────────────────────────────

@runtime_checkable
class MessagePasser(Protocol):
    """Generic message passing / propagation.

    Maze:        Q-learning: propagated = reward + γ·max(neighbor)
    RAG:         attention-weighted: (1-α)·self + α·flow_weighted_avg
    Transformer: self-attention (built into model, not used here)
    """

    def propagate(
        self,
        graph: Any,
        node_values: Dict[Any, float],
        n_iterations: int = 1,
        **kwargs: Any,
    ) -> Dict[Any, float]:
        ...
