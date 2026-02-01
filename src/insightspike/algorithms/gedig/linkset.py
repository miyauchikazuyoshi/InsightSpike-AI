"""Linkset metrics computation for geDIG.

This module provides functions for computing linkset-based geDIG metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from .types import LinksetMetrics


def compute_linkset_metrics(
    g_before: nx.Graph,
    g_after: nx.Graph,
    linkset_info: Optional[Dict[str, Any]],
    *,
    entropy_tau: float = 1.0,
    sp_beta: float = 0.0,
    use_multihop_sp_gain: bool = False,
    ig_mode: str = "raw",
    ig_nonneg: bool = False,
    lambda_weight: float = 1.0,
    use_legacy_formula: bool = False,
    query_vector: Optional[Sequence[float]] = None,
    ig_fixed_den: Optional[float] = None,
) -> LinksetMetrics:
    """Compute linkset-based geDIG metrics.

    Args:
        g_before: Graph before change.
        g_after: Graph after change.
        linkset_info: Dictionary containing linkset information:
            - s_link: List of link items
            - candidate_pool: List of candidate items
            - decision: Decision info with 'index' and 'similarity'
            - query_entry: Query entry info
            - base_mode: Base mode ('link', 'mem', 'pool')
        entropy_tau: Temperature for entropy computation.
        sp_beta: Shortest path beta coefficient.
        use_multihop_sp_gain: Whether to use multihop SP gain.
        ig_mode: IG mode ('raw', 'norm', 'normalized').
        ig_nonneg: Whether to clamp IG to non-negative.
        lambda_weight: Lambda weight for IG term.
        use_legacy_formula: Whether to use legacy formula.
        query_vector: Optional query vector.
        ig_fixed_den: Optional fixed denominator for IG.

    Returns:
        LinksetMetrics with computed values.
    """
    linkset_info = linkset_info or {}
    s_link = linkset_info.get('s_link') or []
    candidate_pool = linkset_info.get('candidate_pool') or []
    decision = linkset_info.get('decision') or {}
    chosen_index = decision.get('index')
    query_entry = linkset_info.get('query_entry')
    base_mode = str(linkset_info.get('base_mode', 'link') or 'link').lower()
    if query_entry is not None:
        query_entry = dict(query_entry)

    # Build before/after sets based on base_mode and unique candidate indices
    before_map: Dict[str, Dict[str, Any]] = {}
    chosen_entry: Optional[Dict[str, Any]] = None

    def _add_before_items(items: List[Dict[str, Any]]):
        nonlocal chosen_entry
        for item in items:
            idx = item.get('index')
            if not idx:
                continue
            key = str(idx)
            snap = dict(item)
            before_map.setdefault(key, snap)
            if chosen_entry is None and idx == chosen_index:
                chosen_entry = snap

    if base_mode in ('mem', 'pool'):
        if base_mode == 'mem':
            base_items = [dict(it) for it in candidate_pool if (it.get('origin') == 'mem')]
        else:
            base_items = [dict(it) for it in candidate_pool]
        if base_items:
            _add_before_items(base_items)
        else:
            _add_before_items([dict(it) for it in s_link])
    else:
        _add_before_items([dict(it) for it in s_link])

    if chosen_entry is None:
        for item in candidate_pool:
            idx = item.get('index')
            if not idx:
                continue
            if idx == chosen_index:
                chosen_entry = dict(item)
                break

    if chosen_entry is None and candidate_pool:
        chosen_entry = dict(candidate_pool[0])

    if chosen_entry is None:
        chosen_entry = {'index': chosen_index, 'similarity': 1.0}
    else:
        chosen_entry = dict(chosen_entry)

    idx = chosen_entry.get('index')
    if idx:
        before_map.setdefault(str(idx), dict(chosen_entry))

    if not before_map and idx:
        before_map[str(idx)] = dict(chosen_entry)

    if query_entry is None:
        sim = decision.get('similarity')
        sim = float(sim) if isinstance(sim, (int, float)) else 1.0
        query_entry = {
            'index': 'query',
            'origin': 'query',
            'similarity': sim if sim > 0 else 1.0,
            'distance': 0.0,
            'weighted_distance': 0.0,
        }
    else:
        query_entry.setdefault('index', 'query')
        query_entry.setdefault('origin', 'query')
        if not query_entry.get('similarity'):
            sim = decision.get('similarity')
            query_entry['similarity'] = float(sim) if isinstance(sim, (int, float)) and sim > 0 else 1.0

    after_map = dict(before_map)
    after_map[str(query_entry.get('index', 'query'))] = dict(query_entry)

    before_list = list(before_map.values())
    after_list = list(after_map.values())

    raw_ged = max(0, len(after_map) - len(before_map))
    denom = 1.0 + len(after_list)
    delta_ged_norm = raw_ged / denom if denom > 0 else 0.0

    def _weights(items: List[Dict[str, Any]]) -> List[float]:
        ws = [item.get('similarity', 0.0) or 0.0 for item in items]
        return [float(w) for w in ws if w > 0.0]

    def _entropy_from_weights(weights: List[float], tau: float = 1.0) -> float:
        if not weights:
            return 0.0
        if tau <= 0 or not math.isfinite(tau):
            tau = 1.0
        if abs(tau - 1.0) < 1e-9:
            total = sum(weights)
            if total <= 0:
                return 0.0
            probabilities = [w / total for w in weights]
        else:
            powered = [math.pow(w, 1.0 / tau) for w in weights if w > 0.0]
            total = sum(powered)
            if total <= 0:
                return 0.0
            probabilities = [w / total for w in powered]
        if not probabilities:
            return 0.0
        return -sum(p * math.log(p + 1e-12) for p in probabilities)

    ws_before = _weights(before_list)
    ws_after = _weights(after_list)
    H_before = _entropy_from_weights(ws_before, tau=entropy_tau)
    H_after = _entropy_from_weights(ws_after, tau=entropy_tau)

    K = max(0, len(after_list))
    norm_den = math.log(K) if K >= 2 else 1e-6
    delta_h_norm = (H_after - H_before) / norm_den if norm_den > 0 else 0.0
    delta_sp_rel = 0.0
    combined_ig = delta_h_norm + sp_beta * delta_sp_rel if use_multihop_sp_gain else delta_h_norm

    ig_for_lambda = combined_ig
    if str(ig_mode).lower() in ('norm', 'normalized'):
        ig_for_lambda = float(np.tanh(max(0.0, ig_for_lambda)))
    if ig_nonneg:
        ig_for_lambda = max(0.0, ig_for_lambda)
    lambda_term = lambda_weight * ig_for_lambda

    if use_legacy_formula:
        struct_impr = float(delta_h_norm)
        g_value = float(struct_impr * (1.0 - lambda_weight * 0.0))
    else:
        struct_cost_eff = float(delta_ged_norm)
        g_value = float(struct_cost_eff - lambda_term)

    topw_b = sorted(ws_before, reverse=True)[:5] if ws_before else []
    topw_a = sorted(ws_after, reverse=True)[:5] if ws_after else []

    return LinksetMetrics(
        delta_ged_norm=float(delta_ged_norm),
        delta_h_norm=float(delta_h_norm),
        delta_sp_rel=float(delta_sp_rel),
        gedig_value=float(g_value),
        raw_ged=float(raw_ged),
        ged_norm_den=float(denom if denom > 0 else 1.0),
        ig_norm_den=float(norm_den if norm_den > 0 else 1.0),
        entropy_before=float(H_before),
        entropy_after=float(H_after),
        ig_delta=float(delta_h_norm),
        before_size=len(before_list),
        after_size=len(after_list),
        query_similarity=float(query_entry.get('similarity', 1.0)),
        pos_w_before=int(len(ws_before)),
        pos_w_after=int(len(ws_after)),
        topw_before=topw_b,
        topw_after=topw_a,
    )


__all__ = ["compute_linkset_metrics"]
