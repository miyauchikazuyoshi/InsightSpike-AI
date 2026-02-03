"""geDIG metrics computation for attention matrices.

This module provides utilities for computing geDIG F-score from Transformer
attention patterns. Used by Transformer experiments for consistent calculation.

Formula:
    F = (ΔEPC - λ·γ·ΔSP) - λ·ΔH

Where:
    - ΔEPC: Edge density (|E| / L²)
    - ΔSP: Path efficiency (1 / avg_shortest_path)
    - ΔH: Normalized entropy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import networkx as nx
import numpy as np
from scipy.stats import entropy


@dataclass
class AttentionGeDIGConfig:
    """Configuration for attention-based geDIG computation."""
    lambda_param: float = 1.0
    gamma: float = 0.5
    threshold: float = 0.01
    use_percentile: bool = True
    percentile: float = 0.9
    undirected_sp: bool = True


@dataclass
class AttentionGeDIGResult:
    """Result of geDIG computation from attention matrix."""
    F: float
    E_eff: float
    delta_epc: float
    delta_sp: float
    delta_h: float
    num_edges: int
    density: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "F": self.F,
            "E_eff": self.E_eff,
            "delta_epc": self.delta_epc,
            "delta_sp": self.delta_sp,
            "delta_h": self.delta_h,
            "num_edges": self.num_edges,
            "density": self.density,
        }


class AttentionGeDIGCalculator:
    """Compute geDIG F-score from attention matrices.

    This calculator is designed for Transformer attention analysis:
    - Input: attention matrix (L x L) and valid token mask
    - Output: geDIG metrics including F-score

    Example:
        >>> calc = AttentionGeDIGCalculator()
        >>> result = calc.compute(attn_matrix, valid_mask)
        >>> print(f"F-score: {result.F}")
    """

    def __init__(self, config: Optional[AttentionGeDIGConfig] = None):
        """Initialize calculator.

        Args:
            config: Configuration for geDIG computation. Uses defaults if None.
        """
        self.config = config or AttentionGeDIGConfig()

    def compute(
        self,
        attn: np.ndarray,
        valid_mask: np.ndarray,
    ) -> AttentionGeDIGResult:
        """Compute geDIG metrics from attention matrix.

        Args:
            attn: Attention matrix of shape (L, L).
            valid_mask: Boolean mask of valid tokens, shape (L,).

        Returns:
            AttentionGeDIGResult with F-score and component metrics.
        """
        idx = np.where(valid_mask)[0]
        if len(idx) == 0:
            return AttentionGeDIGResult(
                F=0.0, E_eff=0.0, delta_epc=0.0,
                delta_sp=0.0, delta_h=0.0, num_edges=0, density=0.0
            )

        # Extract valid tokens only
        attn = attn[np.ix_(idx, idx)]
        L = attn.shape[0]

        # Build graph and compute metrics
        G = self._build_graph(attn)
        max_edges = L * L
        delta_epc = G.number_of_edges() / max_edges if max_edges > 0 else 0.0
        delta_sp = self._compute_path_efficiency(G)
        delta_h = self._compute_entropy(attn)

        # Compute F-score
        cfg = self.config
        E_eff = delta_epc - cfg.lambda_param * cfg.gamma * delta_sp
        F = E_eff - cfg.lambda_param * delta_h

        return AttentionGeDIGResult(
            F=float(F),
            E_eff=float(E_eff),
            delta_epc=float(delta_epc),
            delta_sp=float(delta_sp),
            delta_h=float(delta_h),
            num_edges=int(G.number_of_edges()),
            density=float(nx.density(G)) if G.number_of_nodes() > 0 else 0.0,
        )

    def _build_graph(self, attn: np.ndarray) -> nx.DiGraph:
        """Build directed graph from attention matrix."""
        G = nx.DiGraph()
        L = attn.shape[0]
        G.add_nodes_from(range(L))

        cfg = self.config
        if cfg.use_percentile:
            thresh = float(np.quantile(attn, cfg.percentile))
        else:
            thresh = float(cfg.threshold)

        for i in range(L):
            for j in range(L):
                if attn[i, j] > thresh:
                    G.add_edge(i, j, weight=float(attn[i, j]))
        return G

    def _compute_path_efficiency(self, G: nx.DiGraph) -> float:
        """Compute path efficiency (1 / avg_shortest_path)."""
        if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
            return 0.0
        try:
            if self.config.undirected_sp:
                G2 = G.to_undirected()
                if nx.is_connected(G2):
                    avg_path = nx.average_shortest_path_length(G2)
                    return 1.0 / avg_path if avg_path > 0 else 0.0
                comp = max(nx.connected_components(G2), key=len)
                sub = G2.subgraph(comp).copy()
            else:
                if nx.is_weakly_connected(G):
                    avg_path = nx.average_shortest_path_length(G)
                    return 1.0 / avg_path if avg_path > 0 else 0.0
                comp = max(nx.weakly_connected_components(G), key=len)
                sub = G.subgraph(comp).copy()
            if sub.number_of_nodes() < 2:
                return 0.0
            avg_path = nx.average_shortest_path_length(sub)
            return 1.0 / avg_path if avg_path > 0 else 0.0
        except Exception:
            return 0.0

    def _compute_entropy(self, attn: np.ndarray) -> float:
        """Compute normalized entropy of attention distribution."""
        flat = attn.flatten()
        flat = flat[flat > 1e-10]
        if flat.size == 0:
            return 0.0
        flat = flat / flat.sum()
        H = entropy(flat)
        max_H = np.log(flat.size)
        return float(H / max_H) if max_H > 0 else 0.0


# Convenience function for simple usage
def compute_attention_gedig(
    attn: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lambda_param: float = 1.0,
    gamma: float = 0.5,
    percentile: float = 0.9,
) -> Dict[str, float]:
    """Compute geDIG F-score from attention matrix (convenience function).

    Args:
        attn: Attention matrix of shape (L, L).
        valid_mask: Boolean mask of valid tokens, shape (L,).
        lambda_param: Lambda parameter for IG term weight.
        gamma: Gamma parameter for SP term weight.
        percentile: Percentile threshold for edge creation.

    Returns:
        Dictionary with F-score and component metrics.
    """
    config = AttentionGeDIGConfig(
        lambda_param=lambda_param,
        gamma=gamma,
        use_percentile=True,
        percentile=percentile,
    )
    calc = AttentionGeDIGCalculator(config)
    result = calc.compute(attn, valid_mask)
    return result.to_dict()


__all__ = [
    "AttentionGeDIGConfig",
    "AttentionGeDIGResult",
    "AttentionGeDIGCalculator",
    "compute_attention_gedig",
]
