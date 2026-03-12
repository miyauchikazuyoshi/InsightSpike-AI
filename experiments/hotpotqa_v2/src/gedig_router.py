"""geDIG-based adaptive routing for BRIGHT pipeline.

Computes geDIG value from episode graph topology + feature divergence
to decide which pipeline tier to apply for each query.

geDIG formula (lightweight, self-contained):
  geDIG = Δ_GED_norm - λ · (Δ_H_norm + β_sp · Δ_SP_rel)

Where:
  Δ_GED_norm = normalized graph edit distance (edge/node change ratio)
  Δ_H_norm   = normalized Shannon entropy change of node features
  Δ_SP_rel   = relative shortest-path improvement
  λ          = lambda weight (balance structural cost vs information gain)
  β_sp       = shortest-path weight

geDIG interpretation:
  geDIG < τ_dg → High information integration → DG mode (skip CoT)
  τ_dg ≤ geDIG ≤ τ_ag → Balanced → Moderate CoT
  geDIG > τ_ag → Low integration → AG mode (aggressive CoT + re-retrieval)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Routing decision from geDIG analysis."""
    tier: int               # 1=DG, 2=moderate, 3=AG
    gedig_value: float      # Raw geDIG score
    delta_betti_0: int      # Connected component change
    ig_value: float         # Information gain (Δ_H + β·Δ_SP)
    ged_value: float        # Normalized GED
    delta_sp_rel: float     # Shortest path relative gain
    entropy_before: float
    entropy_after: float
    computation_time_ms: float = 0.0


def _shannon_entropy_features(features: np.ndarray, n_bins: int = 32) -> float:
    """Compute Shannon entropy of feature distribution.

    Discretizes each dimension into bins and computes joint entropy
    over a random subset of dimensions for efficiency.
    """
    if features.shape[0] < 2:
        return 0.0

    n_nodes, n_dim = features.shape

    # Use a subset of dimensions for efficiency (max 64)
    rng = np.random.RandomState(42)
    if n_dim > 64:
        dim_idx = rng.choice(n_dim, 64, replace=False)
        features = features[:, dim_idx]
        n_dim = 64

    # Compute per-dimension entropy and average
    total_entropy = 0.0
    for d in range(n_dim):
        col = features[:, d]
        col_range = col.max() - col.min()
        if col_range < 1e-10:
            continue
        # Bin the values
        bins = np.linspace(col.min() - 1e-10, col.max() + 1e-10, n_bins + 1)
        hist, _ = np.histogram(col, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        total_entropy -= np.sum(probs * np.log(probs))

    return total_entropy / max(n_dim, 1)


def _normalized_ged(g_before: nx.Graph, g_after: nx.Graph,
                    focal_nodes: set[str]) -> float:
    """Compute normalized Graph Edit Distance.

    GED = (|added_nodes| + |added_edges| + |removed_edges|) / max(|E_before|, 1)
    Only counts structural changes relative to the before-graph scale.
    """
    nodes_before = set(g_before.nodes())
    nodes_after = set(g_after.nodes())
    edges_before = set(g_before.edges())
    edges_after = set(g_after.edges())

    added_nodes = len(nodes_after - nodes_before)
    added_edges = len(edges_after - edges_before)
    removed_edges = len(edges_before - edges_after)

    denominator = max(len(edges_before), 1)
    return (added_nodes + added_edges + removed_edges) / denominator


def _sp_gain(g_before: nx.Graph, g_after: nx.Graph,
             sample_pairs: int = 200) -> float:
    """Compute relative shortest-path improvement.

    Samples random node pairs from before-graph and measures how much
    their shortest paths decrease in the after-graph.
    Returns value in [-1, 1]: positive = paths shortened, negative = lengthened.
    """
    common_nodes = list(set(g_before.nodes()) & set(g_after.nodes()))
    if len(common_nodes) < 2:
        return 0.0

    rng = np.random.RandomState(42)
    n = len(common_nodes)
    n_pairs = min(sample_pairs, n * (n - 1) // 2)

    # Sample pairs
    pairs = set()
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        i, j = rng.randint(0, n, size=2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
        attempts += 1

    if not pairs:
        return 0.0

    total_gain = 0.0
    valid_pairs = 0

    for i, j in pairs:
        u, v = common_nodes[i], common_nodes[j]
        try:
            sp_before = nx.shortest_path_length(g_before, u, v)
        except nx.NetworkXNoPath:
            sp_before = None
        try:
            sp_after = nx.shortest_path_length(g_after, u, v)
        except nx.NetworkXNoPath:
            sp_after = None

        if sp_before is not None and sp_after is not None:
            # Relative improvement: positive = shorter path after
            if sp_before > 0:
                gain = (sp_before - sp_after) / sp_before
                total_gain += gain
                valid_pairs += 1
        elif sp_before is None and sp_after is not None:
            # Previously unreachable, now reachable → big gain
            total_gain += 1.0
            valid_pairs += 1
        elif sp_before is not None and sp_after is None:
            # Previously reachable, now unreachable → big loss (shouldn't happen)
            total_gain -= 1.0
            valid_pairs += 1

    return total_gain / max(valid_pairs, 1)


class GeDIGRouter:
    """geDIG-based routing for BRIGHT pipeline.

    Lightweight self-contained geDIG computation (no external dependencies).

    Parameters
    ----------
    lambda_weight : float
        λ in geDIG formula.
    sp_beta : float
        Weight for shortest-path component.
    tau_dg : float
        Threshold for DG mode (geDIG < tau_dg → tier 1).
    tau_ag : float
        Threshold for AG mode (geDIG > tau_ag → tier 3).
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        max_hops: int = 2,       # kept for CLI compat; unused internally
        sp_beta: float = 0.5,
        tau_dg: float = -0.1,
        tau_ag: float = 0.3,
        feature_weights: list[float] | None = None,  # kept for CLI compat
    ):
        self.lambda_weight = lambda_weight
        self.max_hops = max_hops
        self.sp_beta = sp_beta
        self.tau_dg = tau_dg
        self.tau_ag = tau_ag
        self.feature_weights = feature_weights

    def compute_routing(self, episode_graph_result: Any) -> RoutingDecision:
        """Compute routing decision from episode graph.

        Uses lightweight self-contained geDIG:
          geDIG = Δ_GED_norm - λ · (Δ_H_norm + β_sp · Δ_SP)
        """
        t0 = time.time()
        egr = episode_graph_result

        # Edge case: empty graph
        if egr.g_before.number_of_nodes() < 2:
            return RoutingDecision(
                tier=3, gedig_value=1.0, delta_betti_0=0,
                ig_value=0.0, ged_value=0.0, delta_sp_rel=0.0,
                entropy_before=0.0, entropy_after=0.0,
            )

        try:
            # --- Topology: Betti-0 (connected components) ---
            cc_before = nx.number_connected_components(egr.g_before)
            cc_after = nx.number_connected_components(egr.g_after)
            delta_betti_0 = cc_after - cc_before

            # --- GED: normalized graph edit distance ---
            ged_norm = _normalized_ged(
                egr.g_before, egr.g_after, egr.focal_nodes
            )

            # --- IG: Shannon entropy change on features ---
            h_before = _shannon_entropy_features(egr.features_before)
            h_after = _shannon_entropy_features(egr.features_after)

            # Normalize: Δ_H / max(H_before, 0.01) → [-1, +∞)
            # Positive = entropy increased (new info), Negative = entropy decreased (info condensed)
            delta_h = h_after - h_before
            delta_h_norm = delta_h / max(h_before, 0.01)

            # --- SP: shortest-path gain ---
            delta_sp = _sp_gain(
                egr.g_before, egr.g_after, sample_pairs=200
            )

            # --- geDIG formula ---
            ig_combined = delta_h_norm + self.sp_beta * delta_sp
            gedig_value = ged_norm - self.lambda_weight * ig_combined

            # --- Routing decision ---
            tier = self._decide_tier(gedig_value, delta_betti_0, ig_combined)

            elapsed_ms = (time.time() - t0) * 1000

            return RoutingDecision(
                tier=tier,
                gedig_value=round(gedig_value, 4),
                delta_betti_0=delta_betti_0,
                ig_value=round(ig_combined, 4),
                ged_value=round(ged_norm, 4),
                delta_sp_rel=round(delta_sp, 4),
                entropy_before=round(h_before, 4),
                entropy_after=round(h_after, 4),
                computation_time_ms=round(elapsed_ms, 1),
            )

        except Exception as e:
            logger.warning("geDIG computation failed: %s. Defaulting to tier 3.", e)
            return RoutingDecision(
                tier=3, gedig_value=1.0, delta_betti_0=0,
                ig_value=0.0, ged_value=0.0, delta_sp_rel=0.0,
                entropy_before=0.0, entropy_after=0.0,
            )

    def _decide_tier(
        self,
        gedig_value: float,
        delta_betti_0: int,
        ig_value: float,
    ) -> int:
        """Map geDIG value to routing tier.

        Primary: geDIG thresholds only.
        Δβ₀ is used for secondary disambiguation within middle zone,
        but NEVER overrides to T1 (DG) — only primary threshold can skip CoT.
        """
        if gedig_value < self.tau_dg:
            # Strong information integration → DG mode (skip CoT)
            return 1

        if gedig_value > self.tau_ag:
            # Low integration → AG mode (aggressive CoT + re-retrieval)
            return 3

        # Middle zone: default to moderate CoT
        # Use Δβ₀ only to push toward AG (never toward DG)
        if delta_betti_0 > 1:
            # Query creates new disconnected components → needs more exploration
            return 3

        return 2

    def to_dict(self) -> dict:
        """Serialize router config for logging."""
        return {
            "lambda_weight": self.lambda_weight,
            "max_hops": self.max_hops,
            "sp_beta": self.sp_beta,
            "tau_dg": self.tau_dg,
            "tau_ag": self.tau_ag,
            "feature_weights": self.feature_weights,
        }
