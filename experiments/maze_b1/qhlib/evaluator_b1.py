"""β₁-based geDIG evaluator for maze experiments.

Replaces the SP (shortest path) computation with Betti number β₁.

F = ΔGED - λ(ΔH + γ·Δβ₁)

where:
  ΔGED = normalized graph edit distance (before → after)
  ΔH   = entropy change (information gain)
  Δβ₁  = Betti-1 change = ΔE - ΔV + ΔC
         (edges - vertices + connected components)

β₁ measures the number of independent cycles in the graph.
SP was a proxy for "structural shortcuts from cycles" — β₁ IS that quantity.

Key simplification:
  SP computation: 350+ lines (DistanceCache, all_pairs_shortest_path, fixed pairs)
  β₁ computation: 3 lines (E - V + C)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Use the same Node type as the original maze code
Node = Any


@dataclass
class EvalResult:
    """Result of multi-hop geDIG evaluation with β₁."""
    hop_series: List[Dict[str, Any]]
    g0: float
    gmin: float
    best_hop: int
    delta_ged: float
    delta_ig: float
    delta_b1: float              # was: delta_sp
    gmin_mh: float
    delta_ged_min_mh: float
    delta_ig_min_mh: float
    delta_b1_min_mh: float       # was: delta_sp_min_mh
    chosen_edges_by_hop: List[Tuple[Node, Node]]
    # Legacy SP field for comparison (always 0.0 unless explicitly computed)
    delta_sp: float = 0.0
    delta_sp_min_mh: float = 0.0


def compute_betti_1(g: nx.Graph) -> int:
    """Compute β₁ = E - V + C (first Betti number).

    For a graph:
      E = number of edges
      V = number of vertices
      C = number of connected components
      β₁ = E - V + C = number of independent cycles
    """
    if g.number_of_nodes() == 0:
        return 0
    n_components = nx.number_connected_components(g)
    return g.number_of_edges() - g.number_of_nodes() + n_components


def delta_betti_1(g_before: nx.Graph, g_after: nx.Graph) -> float:
    """Compute Δβ₁ = β₁(after) - β₁(before), normalized to [0, 1].

    Positive Δβ₁ = more cycles added (structural redundancy increased).
    Negative Δβ₁ = cycles removed (structure simplified).

    Normalization: divide by max(β₁_before, β₁_after, 1) to keep in [-1, 1].
    """
    b1_before = compute_betti_1(g_before)
    b1_after = compute_betti_1(g_after)
    delta = b1_after - b1_before
    denom = max(b1_before, b1_after, 1)
    return delta / denom


def _norm_ged_simple(
    g_before: nx.Graph,
    g_after: nx.Graph,
    denom: float,
) -> float:
    """Simplified normalized GED based on edge/node difference.

    Uses the same principle as the original: count structural changes
    normalized by maximum possible changes.
    """
    if denom <= 0:
        return 0.0

    nodes_before = set(g_before.nodes())
    nodes_after = set(g_after.nodes())
    edges_before = set(g_before.edges())
    edges_after = set(g_after.edges())

    # Count operations
    added_nodes = len(nodes_after - nodes_before)
    removed_nodes = len(nodes_before - nodes_after)
    added_edges = len(edges_after - edges_before)
    removed_edges = len(edges_before - edges_after)

    raw_ged = added_nodes + removed_nodes + added_edges + removed_edges
    return min(1.0, raw_ged / denom)


def evaluate_multihop_b1(
    *,
    # Core parameters
    lambda_weight: float = 1.0,
    gamma: float = 0.5,          # was: sp_beta
    # Graphs
    prev_graph: nx.Graph,
    stage_graph: nx.Graph,
    g_before_for_expansion: nx.Graph,
    # Node sets
    anchors_core: Set[Node],
    anchors_top_before: Set[Node],
    anchors_top_after: Set[Node],
    # Candidate edges
    ecand: List[Tuple[Node, Node, Dict[str, Any]]],
    # Base IG (entropy-based information gain)
    base_ig: float,
    # Normalization denominator for GED
    denom_cmax_base: float,
    # Hop control
    max_hops: int,
    # AG/DG thresholds
    theta_ag: Optional[float] = None,
    theta_dg: Optional[float] = None,
    # Subgraph expansion function (from original maze code)
    get_subgraph_nodes_fn=None,
) -> EvalResult:
    """Compute per-hop g(h) using β₁ instead of SP.

    g(h) = ΔGED(h) - λ · (ΔH + γ · Δβ₁(h))

    This replaces the 350+ line SP computation with β₁.
    """
    # Build before subgraph at hop 0
    if get_subgraph_nodes_fn:
        nodes_b0 = get_subgraph_nodes_fn(g_before_for_expansion, anchors_core, 0)
        nodes_a0 = get_subgraph_nodes_fn(stage_graph, anchors_core, 0)
    else:
        nodes_b0 = set(anchors_core) | set(anchors_top_before)
        nodes_a0 = set(anchors_core) | set(anchors_top_after)

    sub_b0 = g_before_for_expansion.subgraph(nodes_b0).copy()
    sub_a0 = stage_graph.subgraph(nodes_a0).copy()

    # Hop 0: compute GED, IG, β₁
    ged0 = _norm_ged_simple(sub_b0, sub_a0, denom_cmax_base)
    b1_0 = delta_betti_1(sub_b0, sub_a0)
    ig0 = base_ig + gamma * b1_0
    g0 = float(ged0 - lambda_weight * ig0)

    records_h = [(0, g0, ged0, ig0, b1_0)]
    hop_series = [{"hop": 0, "g": g0, "ged": ged0, "ig": ig0, "h": base_ig, "b1": b1_0}]

    g_best = g0
    h_best = 0

    # AG gate: if g0 < theta_ag, skip multi-hop
    if theta_ag is not None and g0 < theta_ag:
        return EvalResult(
            hop_series=hop_series,
            g0=g0, gmin=g0, best_hop=0,
            delta_ged=ged0, delta_ig=ig0, delta_b1=b1_0,
            gmin_mh=g0,
            delta_ged_min_mh=ged0, delta_ig_min_mh=ig0, delta_b1_min_mh=b1_0,
            chosen_edges_by_hop=[],
        )

    # Multi-hop evaluation
    h_graph = stage_graph.copy()
    chosen_edges: List[Tuple[Node, Node]] = []
    remaining_cand = list(ecand)

    for h in range(1, max_hops + 1):
        if not remaining_cand:
            break

        # Greedy: find the candidate edge that minimizes g(h)
        best_edge = None
        best_g_h = float("inf")
        best_ged_h = 0.0
        best_ig_h = 0.0
        best_b1_h = 0.0

        for ci, (u, v, edata) in enumerate(remaining_cand):
            # Tentatively add edge
            h_graph_try = h_graph.copy()
            h_graph_try.add_edge(u, v, **edata)

            # Get subgraph at this hop
            eff_hop = h
            if get_subgraph_nodes_fn:
                nodes_bh = get_subgraph_nodes_fn(g_before_for_expansion, anchors_core, eff_hop)
                nodes_ah = get_subgraph_nodes_fn(h_graph_try, anchors_core, eff_hop)
            else:
                nodes_bh = set(g_before_for_expansion.nodes())
                nodes_ah = set(h_graph_try.nodes())

            sub_bh = g_before_for_expansion.subgraph(nodes_bh).copy()
            sub_ah = h_graph_try.subgraph(nodes_ah).copy()

            # Compute metrics
            ged_h = _norm_ged_simple(sub_bh, sub_ah, denom_cmax_base)
            b1_h = delta_betti_1(sub_bh, sub_ah)
            ig_h = base_ig + gamma * b1_h
            g_h = float(ged_h - lambda_weight * ig_h)

            if g_h < best_g_h:
                best_g_h = g_h
                best_edge = ci
                best_ged_h = ged_h
                best_ig_h = ig_h
                best_b1_h = b1_h

        if best_edge is None:
            break

        # Accept best edge
        u, v, edata = remaining_cand.pop(best_edge)
        h_graph.add_edge(u, v, **edata)
        chosen_edges.append((u, v))

        records_h.append((h, best_g_h, best_ged_h, best_ig_h, best_b1_h))
        hop_series.append({
            "hop": h, "g": best_g_h, "ged": best_ged_h,
            "ig": best_ig_h, "h": base_ig, "b1": best_b1_h,
        })

        if best_g_h < g_best:
            g_best = best_g_h
            h_best = h

        # DG gate: if g(h) > theta_dg, stop
        if theta_dg is not None and best_g_h > theta_dg:
            break

    # Extract results at g_min
    min_rec = records_h[h_best]

    return EvalResult(
        hop_series=hop_series,
        g0=g0,
        gmin=g_best,
        best_hop=h_best,
        delta_ged=records_h[0][2],
        delta_ig=records_h[0][3],
        delta_b1=records_h[0][4],
        gmin_mh=min_rec[1],
        delta_ged_min_mh=min_rec[2],
        delta_ig_min_mh=min_rec[3],
        delta_b1_min_mh=min_rec[4],
        chosen_edges_by_hop=chosen_edges,
    )
