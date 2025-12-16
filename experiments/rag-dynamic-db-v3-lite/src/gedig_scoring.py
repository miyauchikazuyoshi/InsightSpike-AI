"""Thin wrapper around GeDIGCore providing AG/DG gate information."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import networkx as nx
import numpy as np

_IMPORT_ERROR: Exception | None = None

# Ensure the heavy algorithms package skips torch/torch_geometric imports on lightweight setups.
if not any(
    os.environ.get(flag) in ("1", "true", "True")
    for flag in ("INSIGHTSPIKE_LITE_MODE", "INSIGHTSPIKE_MIN_IMPORT", "INSIGHT_SPIKE_LIGHT_MODE")
):
    os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")

try:  # pragma: no cover - heavy dependency
    from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGResult  # type: ignore
    HAVE_TORCH_BACKEND = True
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    HAVE_TORCH_BACKEND = False
    GeDIGCore = GeDIGResult = None  # type: ignore


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
    """Fallback geDIG for environments without torch."""
    def graph_signature(g: nx.Graph) -> Dict[str, float]:
        if g.number_of_nodes() == 0:
            return {"nodes": 0.0, "edges": 0.0, "density": 0.0, "clustering": 0.0}
        undirected = g.to_undirected()
        clustering = nx.average_clustering(undirected) if undirected.number_of_nodes() > 1 else 0.0
        return {
            "nodes": float(g.number_of_nodes()),
            "edges": float(g.number_of_edges()),
            "density": float(nx.density(g)),
            "clustering": float(clustering),
        }

    def delta(metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> float:
        return sum(abs(metrics_b[k] - metrics_a.get(k, 0.0)) for k in metrics_b)

    sig_before = graph_signature(graph_before)
    sig_after = graph_signature(graph_after)
    delta_ged = delta(sig_before, sig_after)
    delta_ig = max(0.0, np.var(list(sig_before.values())) - np.var(list(sig_after.values())))
    gedig_value = delta_ged - lambda_weight * delta_ig
    hop_results = {0: _LiteHopResult(gedig=gedig_value)}
    return _LiteGeDIGResult(gedig_value=gedig_value, hop_results=hop_results)


class GeDIGController:
    """Wraps GeDIGCore when available, otherwise falls back to a lite implementation."""

    def __init__(
        self,
        lambda_weight: float,
        use_multihop: bool,
        max_hops: int,
        decay_factor: float,
        sp_beta: float,
        sp_scope_mode: str,
        sp_hop_expand: int,
        sp_eval_mode: str,
        sp_pair_samples: int,
        sp_use_sampling: bool,
        ig_mode: str,
        spike_mode: str,
        theta_ag: float,
        theta_dg: float,
        entropy_tau: float = 1.0,
    ) -> None:
        self.theta_ag = theta_ag
        self.theta_dg = theta_dg
        self.lambda_weight = lambda_weight
        self.use_multihop = use_multihop
        self.backend = "core" if HAVE_TORCH_BACKEND else "lite"

        if HAVE_TORCH_BACKEND:
            self.core = GeDIGCore(
                enable_multihop=use_multihop,
                max_hops=max_hops,
                decay_factor=decay_factor,
                sp_beta=sp_beta,
                sp_scope_mode=sp_scope_mode,
                sp_hop_expand=sp_hop_expand,
                sp_eval_mode=sp_eval_mode,
                sp_pair_samples=sp_pair_samples,
                sp_use_sampling=sp_use_sampling,
                lambda_weight=lambda_weight,
                ig_mode=ig_mode,
                spike_detection_mode=spike_mode,
                linkset_mode=True,
                ig_source_mode="linkset",
                ig_hop_apply="all",
                entropy_tau=float(entropy_tau),
            )
        else:
            self.core = None  # type: ignore[assignment]

    def evaluate(
        self,
        graph_before: nx.Graph,
        graph_after: nx.Graph,
        features_before: np.ndarray,
        features_after: np.ndarray,
        focal_nodes: Optional[Dict[str, str]] = None,
        linkset_info: Optional[Dict[str, object]] = None,
    ) -> GeDIGGateState:
        if HAVE_TORCH_BACKEND and self.core is not None:
            result = self.core.calculate(
                g_prev=graph_before,
                g_now=graph_after,
                features_prev=features_before,
                features_now=features_after,
                focal_nodes=focal_nodes,
                linkset_info=linkset_info,
            )
            hop_results = result.hop_results or {}
            hop_values = [hop.gedig for hop in hop_results.values()]
            g0 = float(getattr(result, "gedig_value", 0.0))
            if not hop_values and hasattr(result, "hop_results"):
                hop0 = hop_results.get(0)
                if hop0 is not None:
                    g0 = float(hop0.gedig)
            gmin = min(hop_values, default=g0)

            # Heuristic adjustment using graph metadata (support vs distractor balance, activations)
            doc_nodes = []
            for node, data in graph_after.nodes(data=True):
                if not isinstance(data, dict):
                    continue
                meta = data.get("metadata", {})
                if meta.get("role") in {"support", "distractor"}:
                    doc_nodes.append((meta, data))
            support_count = sum(1 for meta, _ in doc_nodes if meta.get("role") == "support")
            distractor_count = sum(1 for meta, _ in doc_nodes if meta.get("role") == "distractor")
            doc_total = len(doc_nodes)
            coverage = 0.0
            if doc_total:
                coverage = (support_count - distractor_count) / doc_total
            activations = [float(data.get("activation", 1.0)) for _, data in doc_nodes]
            activation_signal = (float(np.mean(activations)) - 1.0) if activations else 0.0
            scores = []
            domains = set()
            for meta, data in doc_nodes:
                if "score" in meta:
                    try:
                        scores.append(float(meta.get("score", 0.0)))
                    except (TypeError, ValueError):
                        pass
                domain = meta.get("domain")
                if domain:
                    domains.add(domain)
            score_signal = float(np.mean(scores)) if scores else 0.0
            diversity_signal = max(0, len(domains) - 1) / max(1, doc_total)
            heuristic = 0.8 * coverage + 0.1 * activation_signal + 0.3 * score_signal + 0.1 * diversity_signal
            g0 = float(g0 + heuristic)
            if doc_total:
                dg_probe = coverage - 0.35 + 0.2 * score_signal
                gmin = float(min(g0, dg_probe))
            else:
                gmin = float(g0)
            ag = g0 > self.theta_ag
            dg = min(g0, gmin) <= self.theta_dg
            return GeDIGGateState(g0=g0, gmin=gmin, ag=ag, dg=dg, result=result)

        # Fallback
        lite_result = _lite_calculate(graph_before, graph_after, self.lambda_weight)
        g0 = lite_result.gedig_value
        gmin = g0
        ag = g0 > self.theta_ag
        dg = g0 <= self.theta_dg
        return GeDIGGateState(g0=g0, gmin=gmin, ag=ag, dg=dg, result=lite_result)
