"""geDIG evaluation wrapper using main code (insightspike.algorithms.gedig_core).

Falls back to a lightweight signature-based delta when heavy deps are unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import networkx as nx
import numpy as np


_IMPORT_ERROR: Exception | None = None

os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")

try:  # pragma: no cover
    from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGResult  # type: ignore
    from insightspike.algorithms.linkset_adapter import build_linkset_info  # type: ignore
    HAVE_CORE = True
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    GeDIGCore = GeDIGResult = None  # type: ignore
    HAVE_CORE = False


@dataclass
class _LiteHopResult:
    gedig: float


@dataclass
class _LiteGeDIGResult:
    gedig_value: float
    hop_results: Dict[int, _LiteHopResult]


@dataclass
class GeDIGGateState:
    g0: float
    gmin: float
    ag: bool
    dg: bool
    result: object


def _lite_calculate(graph_before: nx.Graph, graph_after: nx.Graph, lambda_weight: float) -> _LiteGeDIGResult:
    def sig(g: nx.Graph) -> Dict[str, float]:
        if g.number_of_nodes() == 0:
            return {"nodes": 0.0, "edges": 0.0, "density": 0.0, "clustering": 0.0}
        und = g.to_undirected()
        clustering = nx.average_clustering(und) if und.number_of_nodes() > 1 else 0.0
        return {
            "nodes": float(g.number_of_nodes()),
            "edges": float(g.number_of_edges()),
            "density": float(nx.density(g)),
            "clustering": float(clustering),
        }

    before = sig(graph_before)
    after = sig(graph_after)
    d_ged = sum(abs(after[k] - before.get(k, 0.0)) for k in after)
    d_ig = max(0.0, np.var(list(before.values())) - np.var(list(after.values())))
    gedig_value = float(d_ged - lambda_weight * d_ig)
    return _LiteGeDIGResult(gedig_value=gedig_value, hop_results={0: _LiteHopResult(gedig=gedig_value)})


class GeDIGController:
    def __init__(
        self,
        lambda_weight: float,
        use_multihop: bool,
        max_hops: int,
        decay_factor: float,
        sp_beta: float,
        ig_mode: str,
        spike_mode: str,
        theta_ag: float,
        theta_dg: float,
    ) -> None:
        self.theta_ag = theta_ag
        self.theta_dg = theta_dg
        self.lambda_weight = lambda_weight
        self.backend = "core" if HAVE_CORE else "lite"
        if HAVE_CORE:
            self.core = GeDIGCore(
                enable_multihop=use_multihop,
                max_hops=max_hops,
                decay_factor=decay_factor,
                sp_beta=sp_beta,
                lambda_weight=lambda_weight,
                ig_mode=ig_mode,
                spike_detection_mode=spike_mode,
            )
        else:
            self.core = None  # type: ignore

    def evaluate(
        self,
        graph_before: nx.Graph,
        graph_after: nx.Graph,
        features_before: np.ndarray,
        features_after: np.ndarray,
        focal_nodes: Optional[Dict[str, str]] = None,
    ) -> GeDIGGateState:
        if HAVE_CORE and self.core is not None:
            # Linkset-first: when no candidate pool is available, pass a minimal
            # linkset_info so Core does not fall back to graph-IG (IG≈0 path).
            _ls = build_linkset_info(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link")
            result = self.core.calculate(
                g_prev=graph_before,
                g_now=graph_after,
                features_prev=features_before,
                features_now=features_after,
                focal_nodes=focal_nodes,
                linkset_info=_ls,
            )
            hop_results = result.hop_results or {}
            hop_values = [hop.gedig for hop in hop_results.values()] if hop_results else []
            g0 = float(getattr(result, "gedig_value", 0.0))
            if not hop_values and hop_results:
                hop0 = hop_results.get(0)
                if hop0 is not None:
                    g0 = float(hop0.gedig)
            gmin = float(min(hop_values)) if hop_values else g0
            ag = g0 > self.theta_ag
            dg = min(g0, gmin) <= self.theta_dg
            return GeDIGGateState(g0=g0, gmin=gmin, ag=ag, dg=dg, result=result)

        # fallback
        lite = _lite_calculate(graph_before, graph_after, self.lambda_weight)
        g0 = lite.gedig_value
        gmin = g0
        ag = g0 > self.theta_ag
        dg = g0 <= self.theta_dg
        return GeDIGGateState(g0=g0, gmin=gmin, ag=ag, dg=dg, result=lite)
