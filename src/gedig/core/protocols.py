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
    """AG/DG edge classification output.

    AG = Assertion Graph (confirmed, low uncertainty)
    DG = Differential Graph (uncertain, information gap)

    All experiments classify edges into these two categories.
    """

    n_ag: int = 0
    n_dg: int = 0
    threshold: float = 0.0
    ag_edges: list = field(default_factory=list)
    dg_edges: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    The "third component" of F-eval. Originally defined as shortest-path
    efficiency, redefined as β₁ (Betti number = cycle count) because
    SP's essence is "structural short-circuiting via holes" = β₁ itself.

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
