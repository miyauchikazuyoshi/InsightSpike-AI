"""Pure geDIG core: geDIG ≈ GED - k*IG (networkxのみ依存).
API: PureGeDIGCalculator.calculate(g_prev, g_now) -> PureGeDIGResult.
GED: ノード/エッジ差分単純正規化 / IG: 次数分布エントロピー差 / geDIG: ged - k*ig.
adaptive_k: 呼出回数で k_min→k_max へ指数収束。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
import math
import networkx as nx


@dataclass
class PureGeDIGResult:
    gedig: float
    ged: float
    ig: float
    k: float
    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    entropy_before: float
    entropy_after: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:  # convenience
        return asdict(self)


class PureGeDIGCalculator:
    """Minimal calculator (GED差分正規化, IG=entropy増加, geDIG=ged - k*ig)."""

    def __init__(self, k: float = 0.5, adaptive_k: bool = False, k_min: float = 0.3, k_max: float = 0.7, decay: float = 0.02):
        self.base_k = k
        self.adaptive_k = adaptive_k
        self.k_min = k_min
        self.k_max = k_max
        self.decay = decay
        self._calls = 0

    # ------------------ public ------------------ #
    def calculate(self, g_prev: nx.Graph, g_now: nx.Graph) -> PureGeDIGResult:
        self._calls += 1
        k = self._current_k()

        ged = self._approx_ged(g_prev, g_now)
        ig_gain, ent_before, ent_after = self._ig_gain(g_prev, g_now)
        gedig = ged - k * ig_gain

        return PureGeDIGResult(
            gedig=gedig,
            ged=ged,
            ig=ig_gain,
            k=k,
            nodes_before=g_prev.number_of_nodes(),
            nodes_after=g_now.number_of_nodes(),
            edges_before=g_prev.number_of_edges(),
            edges_after=g_now.number_of_edges(),
            entropy_before=ent_before,
            entropy_after=ent_after,
            metadata={
                "calls": self._calls,
                "adaptive": self.adaptive_k,
            },
        )

    # ------------------ internal: GED ------------------ #
    def _approx_ged(self, g1: nx.Graph, g2: nx.Graph) -> float:
        # 単純差分: ノードとエッジ増減の総量を正規化
        dn = abs(g2.number_of_nodes() - g1.number_of_nodes())
        de = abs(g2.number_of_edges() - g1.number_of_edges())
        denom = max(1, g1.number_of_nodes() + g1.number_of_edges())
        return (dn + de) / denom

    # ------------------ internal: IG ------------------ #
    def _entropy(self, g: nx.Graph) -> float:
        n = g.number_of_nodes()
        if n == 0:
            return 0.0
        degs = [g.degree(n_) for n_ in g.nodes()]
        total = sum(degs)
        if total == 0:
            return 0.0
        ent = 0.0
        for d in degs:
            if d <= 0:
                continue
            p = d / total
            ent -= p * math.log(p + 1e-12)
        return ent

    def _ig_gain(self, g_prev: nx.Graph, g_now: nx.Graph):
        ent_prev = self._entropy(g_prev)
        ent_now = self._entropy(g_now)
        # 情報統合を“構造多様性増加”として扱う (増加が正)
        return ent_now - ent_prev, ent_prev, ent_now

    # ------------------ internal: adaptive k ------------------ #
    def _current_k(self) -> float:
        if not self.adaptive_k:
            return self.base_k
        # 初期は k を低め (情報統合許容)、徐々に base_k に近づき、さらに k_max へしきい値化
        # シンプル: k = clamp(k_min + (k_max - k_min)*(1 - exp(-decay*calls)))
        raw = self.k_min + (self.k_max - self.k_min) * (1 - math.exp(-self.decay * self._calls))
        return raw


__all__ = ["PureGeDIGCalculator", "PureGeDIGResult"]
