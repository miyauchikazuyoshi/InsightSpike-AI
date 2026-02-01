"""Multi-hop geDIG computation.

This module provides the multi-hop processing logic for geDIG calculations.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .types import HopResult, GeDIGResult, LinksetMetrics
from .graph_utils import (
    extract_k_hop_subgraph,
    trim_terminal_edges,
    filter_features,
    compute_sp_gain_norm,
)

logger = logging.getLogger(__name__)


def calculate_multihop(
    g1: nx.Graph,
    g2: nx.Graph,
    features_before: np.ndarray,
    features_after: np.ndarray,
    focal_nodes: Set[str],
    start_time: float,
    *,
    # GED parameters
    node_cost: float = 1.0,
    edge_cost: float = 1.0,
    ged_norm_scheme: str = 'edges_after',
    candidate_count: int = 1,
    # IG parameters
    ig_source_mode: str = 'graph',
    ig_hop_apply: str = 'all',
    ig_mode: str = 'raw',
    ig_nonneg: bool = False,
    ig_norm_strategy: str = 'before',
    ig_delta_mode: str = 'after_before',
    smoothing: float = 1e-10,
    min_nodes: int = 2,
    # Multi-hop parameters
    max_hops: int = 3,
    adaptive_hops: bool = True,
    lambda_weight: float = 1.0,
    # SP gain parameters
    use_multihop_sp_gain: bool = True,
    sp_beta: float = 0.2,
    sp_scope_mode: str = 'auto',
    sp_hop_expand: int = 0,
    sp_boundary_mode: str = 'induced',
    sp_eval_mode: str = 'connected',
    sp_node_cap: int = 200,
    sp_pair_samples: int = 400,
    # Optional inputs
    norm_override: Optional[float] = None,
    query_vector: Optional[List[float]] = None,
    fixed_den: Optional[float] = None,
    k_star: Optional[int] = None,
    linkset_metrics: Optional[LinksetMetrics] = None,
    # Feature weights
    feature_weights: Optional[np.ndarray] = None,
    # Structural similarity evaluator
    ss_evaluator: Optional[Any] = None,
    # Callbacks for GED and IG computation
    ged_calculator: Optional[Any] = None,
    ig_calculator: Optional[Any] = None,
) -> GeDIGResult:
    """Calculate multi-hop geDIG metrics.

    Args:
        g1: Graph before change.
        g2: Graph after change.
        features_before: Feature matrix before change.
        features_after: Feature matrix after change.
        focal_nodes: Set of focal nodes for hop expansion.
        start_time: Start time for computation timing.
        node_cost: Node cost for GED calculation.
        edge_cost: Edge cost for GED calculation.
        ged_norm_scheme: GED normalization scheme.
        candidate_count: Number of candidates for normalization.
        ig_source_mode: IG source mode ('graph', 'linkset', etc.).
        ig_hop_apply: Which hops to apply IG ('all', 'hop0').
        ig_mode: IG mode ('raw', 'norm', 'normalized').
        ig_nonneg: Whether to clamp IG to non-negative.
        ig_norm_strategy: IG normalization strategy.
        ig_delta_mode: IG delta mode.
        smoothing: Smoothing constant.
        min_nodes: Minimum nodes for computation.
        max_hops: Maximum number of hops.
        adaptive_hops: Whether to use adaptive hop termination.
        lambda_weight: Lambda weight for IG term.
        use_multihop_sp_gain: Whether to use SP gain.
        sp_beta: SP gain weight.
        sp_scope_mode: SP scope mode.
        sp_hop_expand: SP hop expansion.
        sp_boundary_mode: SP boundary mode.
        sp_eval_mode: SP evaluation mode.
        sp_node_cap: SP node cap for performance.
        sp_pair_samples: SP pair samples.
        norm_override: Optional normalization override.
        query_vector: Optional query vector.
        fixed_den: Optional fixed denominator for IG.
        k_star: Optional k_star value.
        linkset_metrics: Optional precomputed linkset metrics.
        feature_weights: Optional feature weights.
        ss_evaluator: Optional structural similarity evaluator.
        ged_calculator: Callback for GED calculation.
        ig_calculator: Callback for IG calculation.

    Returns:
        GeDIGResult with multi-hop computation results.
    """
    hop_results: Dict[int, HopResult] = {}

    for hop in range(max_hops + 1):
        sub_g1, nodes1 = extract_k_hop_subgraph(g1, focal_nodes, hop)
        sub_g2, nodes2 = extract_k_hop_subgraph(g2, focal_nodes, hop)
        if len(sub_g1) == 0 and len(sub_g2) == 0:
            continue

        # GED normalization denominator scheme selection
        if str(ged_norm_scheme).lower() in ('candidate', 'candidate_base', 'link', 'links', 'linkset'):
            # Cmax ≈ c_node + |S_link|·c_edge (fixed candidate base)
            base_k = int(max(1, candidate_count))
            denom = node_cost + edge_cost * base_k
        else:
            denom = node_cost + edge_cost * max(sub_g2.number_of_edges(), 0)
            if denom <= 0.0:
                denom = node_cost + edge_cost

        # Calculate GED
        if ged_calculator is not None:
            ged_result = ged_calculator(sub_g1, sub_g2, norm_override=denom)
        else:
            ged_result = _default_ged(sub_g1, sub_g2, node_cost, edge_cost, denom)

        sub_before = filter_features(features_before, nodes1, g1)
        sub_after = filter_features(features_after, nodes2, g2)

        # IG source selection
        if str(ig_source_mode).lower() in ('linkset', 'paper', 'strict') and linkset_metrics is not None:
            # Paper-compliant: use candidate distribution-based ΔH
            delta_h_norm = float(linkset_metrics.delta_h_norm)
            ig_result = {
                'ig_value': delta_h_norm,
                'entropy_before': float(linkset_metrics.entropy_before),
                'entropy_after': float(linkset_metrics.entropy_after),
                'delta_entropy': float(linkset_metrics.ig_delta),
                'normalization_den': float(linkset_metrics.ig_norm_den),
            }
        else:
            if ig_calculator is not None:
                ig_result = ig_calculator(
                    sub_g2,
                    sub_before,
                    sub_after,
                    query_vector=query_vector,
                    fixed_den=fixed_den,
                    k_star=candidate_count,
                )
            else:
                ig_result = _default_ig(sub_g2, sub_before, sub_after, smoothing, min_nodes, fixed_den)
            delta_h_norm = float(ig_result['ig_value'])

        delta_ged_norm = float(ged_result['normalized_ged'])
        delta_sp_rel = 0.0
        sp_multiplier = 0.0

        if hop > 0 and use_multihop_sp_gain:
            delta_sp_rel, sp_multiplier = _compute_hop_sp_gain(
                g1, g2, focal_nodes, hop,
                sp_hop_expand=sp_hop_expand,
                sp_scope_mode=sp_scope_mode,
                sp_boundary_mode=sp_boundary_mode,
                sp_eval_mode=sp_eval_mode,
                sp_beta=sp_beta,
                sp_node_cap=sp_node_cap,
                sp_pair_samples=sp_pair_samples,
            )

        # IG application scope (hop0 only or all hops)
        if str(ig_source_mode).lower() in ('linkset', 'paper', 'strict') and str(ig_hop_apply).lower() == 'hop0' and hop > 0:
            # hop>0 uses only SP
            combined_ig = 0.0 + sp_multiplier * delta_sp_rel
        else:
            combined_ig = delta_h_norm + sp_multiplier * delta_sp_rel

        # Structural similarity bonus for analogy detection
        analogy_bonus = 0.0
        if ss_evaluator is not None:
            try:
                center_node = list(focal_nodes)[0] if focal_nodes else None
                analogy_bonus = ss_evaluator.compute_analogy_bonus(
                    sub_g1, sub_g2,
                    center1=center_node,
                    center2=center_node,
                )
                if analogy_bonus > 0:
                    combined_ig += analogy_bonus
                    logger.debug(
                        "[ANALOGY] hop=%d bonus=%.4f combined_ig=%.4f",
                        hop, analogy_bonus, combined_ig
                    )
            except Exception as e:
                logger.warning("Structural similarity evaluation failed: %s", e)

        ig_for_lambda = combined_ig
        if str(ig_mode).lower() in ('norm', 'normalized'):
            ig_for_lambda = float(np.tanh(max(0.0, ig_for_lambda)))
        if ig_nonneg:
            ig_for_lambda = max(0.0, ig_for_lambda)
        lambda_term = lambda_weight * ig_for_lambda
        hop_gedig = float(delta_ged_norm - lambda_term)

        hop_results[hop] = HopResult(
            hop=hop,
            ged=delta_ged_norm,
            ig=combined_ig,
            gedig=hop_gedig,
            struct_cost=delta_ged_norm,
            node_count=len(sub_g2),
            edge_count=sub_g2.number_of_edges(),
            sp=delta_sp_rel,
            h_component=delta_h_norm,
            ged_raw=float(ged_result.get('raw_ged', 0.0)),
            ged_den=float(ged_result.get('normalization_den', denom)),
            entropy_before=float(ig_result.get('entropy_before', 0.0)),
            entropy_after=float(ig_result.get('entropy_after', 0.0)),
            ig_delta=float(ig_result.get('delta_entropy', 0.0)),
            ig_den=float(ig_result.get('normalization_den', fixed_den if fixed_den is not None else 1.0)),
            variance_reduction=float(ig_result.get('variance_reduction', 0.0)),
        )

        if adaptive_hops and hop > 0 and abs(hop_gedig) < 0.01:
            break

    if not hop_results:
        return GeDIGResult(
            gedig_value=0.0,
            ged_value=0.0,
            ig_value=0.0,
            raw_ged=0.0,
            ged_norm_den=1.0,
            ig_raw=0.0,
            ig_norm_den=1.0,
            delta_ged_norm=0.0,
            delta_sp_rel=0.0,
            delta_h_norm=0.0,
            structural_cost=0.0,
            structural_improvement=0.0,
            information_integration=0.0,
            entropy_before=0.0,
            entropy_after=0.0,
            ig_delta=0.0,
            variance_reduction=0.0,
            hop_results={},
            computation_time=time.time() - start_time,
            focal_nodes=focal_nodes,
            version="onegauge_v1_multihop",
        )

    hop0 = hop_results.get(0, next(iter(hop_results.values())))
    best_hop = min(hop_results.keys(), key=lambda h: hop_results[h].gedig)
    best_result = hop_results[best_hop]

    return GeDIGResult(
        gedig_value=best_result.gedig,
        ged_value=hop0.ged,
        ig_value=best_result.ig,
        raw_ged=hop0.ged_raw,
        ged_norm_den=hop0.ged_den,
        ig_raw=best_result.ig,
        ig_norm_den=hop0.ig_den,
        delta_ged_norm=hop0.ged,
        delta_sp_rel=best_result.sp,
        delta_h_norm=hop0.h_component,
        structural_cost=hop0.struct_cost,
        structural_improvement=-hop0.ged,
        information_integration=best_result.ig,
        entropy_before=hop0.entropy_before,
        entropy_after=hop0.entropy_after,
        ig_delta=hop0.ig_delta,
        variance_reduction=hop0.variance_reduction,
        hop_results=hop_results,
        focal_nodes=focal_nodes,
        computation_time=time.time() - start_time,
        version="onegauge_v1_multihop"
    )


def _compute_hop_sp_gain(
    g1: nx.Graph,
    g2: nx.Graph,
    focal_nodes: Set[str],
    hop: int,
    *,
    sp_hop_expand: int = 0,
    sp_scope_mode: str = 'auto',
    sp_boundary_mode: str = 'induced',
    sp_eval_mode: str = 'connected',
    sp_beta: float = 0.2,
    sp_node_cap: int = 200,
    sp_pair_samples: int = 400,
) -> Tuple[float, float]:
    """Compute SP gain for a specific hop.

    Returns:
        Tuple of (delta_sp_rel, sp_multiplier).
    """
    eff_hop = hop + int(max(0, sp_hop_expand))
    sp_g1, nodes_sp1 = extract_k_hop_subgraph(g1, focal_nodes, eff_hop)
    sp_g2, nodes_sp2 = extract_k_hop_subgraph(g2, focal_nodes, eff_hop)

    if str(sp_scope_mode).lower() in ('union', 'merge', 'superset'):
        all_nodes = set(nodes_sp1) | set(nodes_sp2)
        if all_nodes:
            sp_g1 = g1.subgraph(all_nodes).copy()
            sp_g2 = g2.subgraph(all_nodes).copy()

    if str(sp_boundary_mode).lower() in ('trim', 'terminal', 'nodes'):
        sp_g1 = trim_terminal_edges(sp_g1, focal_nodes, eff_hop)
        sp_g2 = trim_terminal_edges(sp_g2, focal_nodes, eff_hop)

    delta_sp_rel = 0.0

    if sp_eval_mode in ('fixed_before_pairs', 'fixed_pairs', 'fixed'):
        # Fixed-before-pairs: measure La on the same pair set as before
        try:
            dist1 = dict(nx.all_pairs_shortest_path_length(sp_g1))
            pairs = []
            total1 = 0.0
            for u, dmap in dist1.items():
                for v, d in dmap.items():
                    if v == u:
                        continue
                    if v <= u:
                        continue
                    total1 += float(d)
                    pairs.append((u, v, float(d)))
            if pairs:
                Lb = total1 / len(pairs)
                dist2 = dict(nx.all_pairs_shortest_path_length(sp_g2))
                total2 = 0.0
                count2 = 0
                for u, v, _ in pairs:
                    dm = dist2.get(u, {})
                    if v in dm:
                        total2 += float(dm[v])
                        count2 += 1
                if count2 > 0 and Lb > 0.0:
                    La = total2 / count2
                    gain = Lb - La  # signed gain
                    # relative signed change clamped to [-1, 1] for robustness
                    delta_sp_rel = max(-1.0, min(1.0, gain / Lb))
        except Exception:
            delta_sp_rel = 0.0
    else:
        delta_sp_rel = float(compute_sp_gain_norm(sp_g1, sp_g2, 'relative', sp_node_cap, sp_pair_samples))

    return delta_sp_rel, sp_beta


def _default_ged(
    g1: nx.Graph,
    g2: nx.Graph,
    node_cost: float,
    edge_cost: float,
    denom: float,
) -> Dict[str, float]:
    """Default GED calculation when no callback is provided."""
    nodes1 = set(g1.nodes())
    nodes2 = set(g2.nodes())
    edges1 = set(g1.edges())
    edges2 = set(g2.edges())

    node_diff = len(nodes2 - nodes1) + len(nodes1 - nodes2)
    edge_diff = len(edges2 - edges1) + len(edges1 - edges2)
    raw_ged = node_cost * node_diff + edge_cost * edge_diff
    normalized_ged = raw_ged / denom if denom > 0 else 0.0

    return {
        'raw_ged': raw_ged,
        'normalized_ged': normalized_ged,
        'normalization_den': denom,
        'structural_cost': normalized_ged,
        'structural_improvement': -normalized_ged,
    }


def _default_ig(
    graph: nx.Graph,
    features_before: np.ndarray,
    features_after: np.ndarray,
    smoothing: float,
    min_nodes: int,
    fixed_den: Optional[float],
) -> Dict[str, float]:
    """Default IG calculation when no callback is provided."""
    # Simple entropy-based IG
    def _entropy(features: np.ndarray) -> float:
        if features.size == 0:
            return 0.0
        if features.ndim == 1:
            features = features.reshape(1, -1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, smoothing)
        normalized = features / norms
        if len(normalized) < 2:
            return 0.0
        sims = np.dot(normalized, normalized.T)
        probs = (sims + 1) / 2
        probs = probs.flatten()
        probs = probs / (probs.sum() + smoothing)
        return float(-np.sum(probs * np.log(probs + smoothing)))

    h_before = _entropy(features_before)
    h_after = _entropy(features_after)
    delta_h = h_after - h_before

    den = fixed_den if fixed_den is not None else max(abs(h_before), 1.0)
    ig_value = delta_h / den if den > 0 else 0.0

    return {
        'ig_value': ig_value,
        'entropy_before': h_before,
        'entropy_after': h_after,
        'delta_entropy': delta_h,
        'normalization_den': den,
        'variance_reduction': 0.0,
    }


__all__ = ["calculate_multihop"]
