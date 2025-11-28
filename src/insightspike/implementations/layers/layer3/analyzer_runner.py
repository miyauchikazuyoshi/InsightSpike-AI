"""Top-level analyze_documents orchestration for Layer3 (refactor target).

This runner encapsulates the main analyze_documents flow so that
layer3_graph_reasoner.py can be slimmed down.
"""

from __future__ import annotations

import logging
import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Set

from .analysis import GraphAnalyzer, RewardCalculator
from .message_passing_controller import MessagePassingController

logger = logging.getLogger(__name__)


def run_analysis(
    reasoner: Any,
    documents: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    allow_query_focal: bool = True,
) -> Dict[str, Any]:
    """Extracted analyze_documents core logic."""
    context = context or {}

    # If query-focal is requested, delegate to dedicated handler (non-breaking)
    if allow_query_focal and _query_focal_enabled(reasoner, context):
        return handle_query_focal(reasoner, documents, context)

    # Build current graph (with incremental + synthetic fallback)
    if context.get("graph") is not None:
        current_graph = context["graph"]
    elif not documents:
        synthetic_embedding = np.random.normal(0, 0.1, (1, 384))
        current_graph = _synthetic_graph(synthetic_embedding)
        if reasoner.previous_graph is None:
            reasoner.previous_graph = _synthetic_graph(synthetic_embedding)
    else:
        if reasoner.previous_graph is not None and documents:
            current_graph = reasoner.graph_builder.build_graph(documents, incremental=True)
            _ensure_graph_is_real(current_graph, documents, reasoner)
            if current_graph.num_nodes < getattr(reasoner.previous_graph, "num_nodes", 0):
                current_graph = reasoner.previous_graph
        else:
            current_graph = reasoner.graph_builder.build_graph(documents)
            _ensure_graph_is_real(current_graph, documents, reasoner)

    query_vector = context.get("query_vector")

    # Message passing
    if getattr(reasoner, "message_passing_enabled", False) and query_vector is not None:
        mp_controller = MessagePassingController(
            config=reasoner.config, original_config=reasoner._original_config
        )
        mp_controller.message_passing_enabled = reasoner.message_passing_enabled
        mp_controller.message_passing = reasoner.message_passing
        mp_controller.edge_reevaluator = reasoner.edge_reevaluator
        current_graph = mp_controller.apply(current_graph, query_vector)

    previous_graph = context.get("previous_graph", reasoner.previous_graph)

    # Metrics
    graph_analyzer = getattr(reasoner, "graph_analyzer", GraphAnalyzer(reasoner.config))
    metrics = graph_analyzer.calculate_metrics(
        current_graph=current_graph,
        previous_graph=previous_graph,
        delta_ged_func=reasoner.delta_ged,
        delta_ig_func=reasoner.delta_ig,
    )

    # Conflicts
    conflicts = reasoner.conflict_scorer.calculate_conflict(previous_graph, current_graph, context)

    # Rewards
    reward_calc = getattr(reasoner, "reward_calculator", RewardCalculator(reasoner.config))
    reward = reward_calc.calculate_reward(metrics, conflicts)

    # Spike detection / quality
    thresholds = {
        "ged": getattr(reasoner.config.graph, "spike_ged_threshold", -0.5),
        "ig": getattr(reasoner.config.graph, "spike_ig_threshold", 0.2),
        "conflict": getattr(reasoner.config.graph, "conflict_threshold", 0.5),
    }
    spike_detected = graph_analyzer.detect_spike(metrics, conflicts, thresholds)
    reasoning_quality = graph_analyzer.assess_quality(metrics, conflicts)

    reasoner.previous_graph = current_graph

    return {
        "graph": current_graph,
        "metrics": metrics,
        "conflicts": conflicts,
        "reward": reward,
        "spike_detected": spike_detected,
        "reasoning_quality": reasoning_quality,
    }


def _synthetic_graph(embedding):
    from torch_geometric.data import Data as _Data

    g = _Data(
        x=torch.tensor(embedding, dtype=torch.float),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
    )
    g.num_nodes = 1
    return g


def _ensure_graph_is_real(current_graph, documents, reasoner):
    """If graph_builder returns a mock without num_nodes, create a minimal PyG graph."""
    try:
        if not isinstance(getattr(current_graph, "num_nodes", None), int):
            import numpy as _np
            import torch as _torch
            from torch_geometric.data import Data as _Data

            emb = _np.random.randn(len(documents), reasoner.config.embedding.dimension)
            edge_index = (
                _torch.tensor([[0], [0]], dtype=_torch.long)
                if len(documents) == 1
                else _torch.tensor([[0, 1], [1, 0]], dtype=_torch.long)
            )
            current_graph = _Data(x=_torch.tensor(emb, dtype=_torch.float), edge_index=edge_index)
            current_graph.num_nodes = current_graph.x.size(0)
    except Exception:
        pass
    return current_graph


def _query_focal_enabled(reasoner, context: Dict[str, Any], allow: bool = True) -> bool:
    """Detect whether query-focal metrics are requested (delegate to legacy flow)."""
    if not allow:
        return False

    def _get_bool(env_key: str, default: bool) -> bool:
        val = os.getenv(env_key)
        if val is None:
            return default
        return val.strip().lower() in ("1", "true", "yes", "on")

    def _cfg_attr(obj: Any, path: str, default: Any) -> Any:
        cur = obj
        for part in path.split("."):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = getattr(cur, part, None)
        return default if cur is None else cur

    return _get_bool("INSIGHTSPIKE_QUERY_FOCAL_METRICS", False) or bool(
        _cfg_attr(reasoner.config, "graph.query_focal_metrics", False)
    )


def handle_query_focal(
    reasoner: Any,
    documents: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Handle query-focal metrics (centers-based GeDIG evaluation).

    Falls back to a neutral result on any error to avoid breaking callers.
    """
    context = context or {}
    try:
        current_graph = context.get("graph")
        if current_graph is None:
            # Build graph similar to main runner (no incremental shortcut here)
            if not documents:
                synthetic_embedding = np.random.normal(0, 0.1, (1, 384))
                current_graph = _synthetic_graph(synthetic_embedding)
            else:
                current_graph = reasoner.graph_builder.build_graph(documents)
                current_graph = _ensure_graph_is_real(current_graph, documents, reasoner)

        previous_graph = context.get("previous_graph", getattr(reasoner, "previous_graph", None))
        if previous_graph is None:
            # No baseline; return neutral
            return _neutral_result(current_graph)

        centers = _determine_centers(current_graph)
        cfg = getattr(reasoner, "config", None)
        sp_engine = _select_sp_engine(cfg)
        use_cached_sp = sp_engine in ("cached", "cached_incr")

        core = _get_gedig_core(use_cached_sp, cfg)

        k_star = None
        l1_candidates = None
        ig_fixed_den = None
        selection_summary = context.get("candidate_selection") or {}
        if selection_summary:
            try:
                k_val = int(selection_summary.get("k_star") or 0)
                if k_val >= 1:
                    k_star = k_val
                    l1_candidates = int(selection_summary.get("l1_candidates", k_val) or k_val)
                    ig_fixed_den = selection_summary.get("log_k_star")
            except Exception:
                pass

        metrics = {}
        if not use_cached_sp:
            res = core.calculate(
                g_prev=previous_graph,
                g_now=current_graph,
                focal_nodes=set(centers),
                k_star=k_star,
                l1_candidates=l1_candidates,
                ig_fixed_den=ig_fixed_den,
            )
            hop0 = res.hop_results.get(0) if getattr(res, "hop_results", None) else None
            sp_beta = _get_float(cfg, "graph.sp_beta", os.getenv("INSIGHTSPIKE_SP_BETA", 0.2))
            _h_norm = float(getattr(res, "delta_h_norm", 0.0))
            _sp_rel = float(getattr(res, "delta_sp_rel", 0.0))
            _ig_norm = float(_h_norm + sp_beta * _sp_rel)
            metrics = {
                "delta_ged": -float(getattr(res, "delta_ged_norm", 0.0)),
                "delta_ged_norm": float(getattr(res, "delta_ged_norm", 0.0)),
                "delta_h": _h_norm,
                "delta_sp": _sp_rel,
                "delta_ig": _ig_norm,
                "g0": float(hop0.gedig if hop0 else getattr(res, "gedig_value", 0.0)),
                "gmin": float(getattr(res, "gedig_value", 0.0)),
                "graph_size_current": int(getattr(current_graph, "num_nodes", 0)),
                "graph_size_previous": int(getattr(previous_graph, "num_nodes", 0)),
                "candidate_selection": selection_summary,
                "sp_engine": "core",
            }
        else:
            res0 = core.calculate(
                g_prev=previous_graph,
                g_now=current_graph,
                focal_nodes=set(centers),
                k_star=k_star,
                l1_candidates=l1_candidates,
                ig_fixed_den=ig_fixed_den,
            )
            sp_rel = _estimate_sp_cached(previous_graph, current_graph, centers, cfg)
            lambda_w = _get_float(cfg, "graph.lambda_weight", os.getenv("INSIGHTSPIKE_GEDIG_LAMBDA", 1.0))
            sp_beta = _get_float(cfg, "graph.sp_beta", os.getenv("INSIGHTSPIKE_SP_BETA", 0.2))
            delta_ged_norm = float(getattr(res0, "delta_ged_norm", 0.0))
            h_norm = float(getattr(res0, "delta_h_norm", 0.0))
            ig_norm = h_norm + sp_beta * sp_rel
            g_cached = delta_ged_norm - lambda_w * ig_norm
            g0_val = (
                float(res0.hop_results.get(0).gedig)
                if getattr(res0, "hop_results", None) and 0 in res0.hop_results
                else float(getattr(res0, "gedig_value", 0.0))
            )
            metrics = {
                "delta_ged": -delta_ged_norm,
                "delta_ged_norm": delta_ged_norm,
                "delta_ig": ig_norm,
                "delta_h": h_norm,
                "delta_sp": sp_rel,
                "g0": g0_val,
                "gmin": g_cached,
                "graph_size_current": int(getattr(current_graph, "num_nodes", 0)),
                "graph_size_previous": int(getattr(previous_graph, "num_nodes", 0)),
                "candidate_selection": selection_summary,
                "sp_engine": "cached",
            }

        conflicts = getattr(reasoner, "conflict_scorer", None)
        conflicts = conflicts.calculate_conflict(previous_graph, current_graph, context) if conflicts else {"total": 0.0}

        reward_calc = getattr(reasoner, "reward_calculator", None) or RewardCalculator(getattr(reasoner, "config", {}))
        reward = reward_calc.calculate_reward(metrics, conflicts)

        ga = getattr(reasoner, "graph_analyzer", None) or GraphAnalyzer(getattr(reasoner, "config", {}))
        thresholds = {
            "ged": _get_float(cfg, "graph.spike_ged_threshold", -0.5),
            "ig": _get_float(cfg, "graph.spike_ig_threshold", 0.2),
            "conflict": _get_float(cfg, "graph.conflict_threshold", 0.5),
        }
        spike_detected = ga.detect_spike(metrics, conflicts, thresholds)
        reasoning_quality = ga.assess_quality(metrics, conflicts)

        reasoner.previous_graph = current_graph

        return {
            "graph": current_graph,
            "metrics": metrics,
            "conflicts": conflicts,
            "reward": reward,
            "spike_detected": spike_detected,
            "reasoning_quality": reasoning_quality,
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("Query-focal handler failed, returning neutral result: %s", exc)
        return _neutral_result(context.get("graph") if context else None)


def _determine_centers(current_graph) -> List[int]:
    try:
        n = int(getattr(current_graph, "num_nodes", 0))
        topk = max(1, min(3, n))
        return list(range(topk))
    except Exception:
        return []


def _select_sp_engine(cfg) -> str:
    try:
        return str(os.getenv("INSIGHTSPIKE_SP_ENGINE", str(getattr(cfg.graph, "sp_engine", "core") or "core"))).lower()
    except Exception:
        return "core"


def _get_float(cfg, path: str, default):
    try:
        cur = cfg
        for part in path.split("."):
            cur = getattr(cur, part, None)
        if cur is None:
            return float(default)
        return float(cur)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def _get_core_classes(require_cache: bool = True):
    global _GeDIGCore, _DistanceCache, _pyg_to_networkx
    need_import = _GeDIGCore is None or (require_cache and (_DistanceCache is None or _pyg_to_networkx is None))
    if need_import:
        from ...algorithms.gedig_core import GeDIGCore as _GC  # type: ignore

        _GeDIGCore = _GC
        if require_cache:
            from ...algorithms.sp_distcache import DistanceCache as _DC  # type: ignore
            from ...metrics.pyg_compatible_metrics import pyg_to_networkx as _p2n  # type: ignore

            _DistanceCache = _DC
            _pyg_to_networkx = _p2n
    return _GeDIGCore, _DistanceCache, _pyg_to_networkx


def _get_gedig_core(use_cached_sp: bool, cfg) -> Any:
    GeDIGCore, _, _ = _get_core_classes(require_cache=use_cached_sp)
    # if cached, we still construct core but let caller decide usage
    return GeDIGCore(
        enable_multihop=not use_cached_sp,
        max_hops=max(0, int(getattr(getattr(cfg, "metrics", None) or getattr(getattr(cfg, "graph", None), "metrics", None) or {}, "query_radius", 1) or 1)),
        use_local_normalization=bool(getattr(getattr(cfg, "metrics", None) or getattr(getattr(cfg, "graph", None), "metrics", None) or {}, "use_local_normalization", False)),
    )


def _estimate_sp_cached(g_before, g_after, centers: List[int], cfg) -> float:
    try:
        _, DistanceCache, pyg_to_networkx = _get_core_classes(require_cache=True)
        cache = DistanceCache(mode="cached")
        sig = cache.signature(pyg_to_networkx(g_before), set(centers), max(0, _get_int(cfg, "metrics.query_radius", 1)), "union", "trim")
        return float(cache.estimate_sp_between_graphs(sig=sig, g_before=g_before, g_after=g_after))
    except Exception:
        return 0.0


def _get_int(cfg, path: str, default: int) -> int:
    try:
        cur = cfg
        for part in path.split("."):
            cur = getattr(cur, part, None)
        if cur is None:
            return int(default)
        return int(cur)
    except Exception:
        try:
            return int(default)
        except Exception:
            return 0


def _neutral_result(graph):
    return {
        "graph": graph,
        "metrics": {"delta_ged": 0.0, "delta_ig": 0.0, "delta_ged_norm": 0.0, "delta_sp": 0.0},
        "conflicts": {"total": 0.0},
        "reward": {"insight_reward": 0.0, "quality_bonus": 0.0, "total": 0.0},
        "spike_detected": False,
        "reasoning_quality": 0.5,
    }


__all__ = ["run_analysis", "handle_query_focal"]
_GeDIGCore = None
_DistanceCache = None
_pyg_to_networkx = None
