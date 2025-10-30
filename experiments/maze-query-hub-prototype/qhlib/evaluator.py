from __future__ import annotations

"""
Evaluator (stub for Phase 2)

This module will host the evaluation routines for geDIG/IG/SP including:
  - 2-phase evaluation (graph_pre → eval_after)
  - hop series computation (g(h), ΔGED, IG, H, ΔSP)
  - best hop selection and gmin aggregation

For now, this file provides only the interface and minimal helpers to be
incrementally adopted from the legacy runner.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.algorithms.core.metrics import normalized_ged as _norm_ged
from insightspike.algorithms.sp_distcache import DistanceCache
try:
    from .sp_pairsets import SPPairsetService, SignatureBuilder, PairRecord, Pairset, _id_to_node
except Exception:  # pragma: no cover
    SPPairsetService = object  # type: ignore
    SignatureBuilder = object  # type: ignore
    def _id_to_node(s: str):  # type: ignore
        a,b,c = (int(p) for p in s.split(',')); return (a,b,c)

Node = Tuple[int, int, int]


@dataclass
class EvalResult:
    hop_series: List[Dict[str, Any]]
    g0: float
    gmin: float
    best_hop: int
    delta_ged: float
    delta_ig: float
    delta_sp: float
    gmin_mh: float
    delta_ged_min_mh: float
    delta_ig_min_mh: float
    delta_sp_min_mh: float
    chosen_edges_by_hop: List[Tuple[Node, Node]]
    # Diagnostics: SP evaluation cost proxies
    sssp_calls_du: int = 0
    sssp_calls_dv: int = 0
    dv_leaf_skips: int = 0
    cycle_verifies: int = 0


def evaluate_multihop(
    *,
    core: GeDIGCore,
    prev_graph: nx.Graph,
    stage_graph: nx.Graph,
    g_before_for_expansion: nx.Graph,
    anchors_core: Set[Node],
    anchors_top_before: Set[Node],
    anchors_top_after: Set[Node],
    ecand: List[Tuple[Node, Node, Dict[str, Any]]],
    base_ig: float,
    denom_cmax_base: float,
    max_hops: int,
    ged_hop0_const: bool = False,
    ig_recompute: bool = False,
    pre_linkset_info: Optional[Dict[str, Any]] = None,
    query_vec: Optional[List[float]] = None,
    ig_fixed_den: Optional[float] = None,
    theta_ag: Optional[float] = None,
    theta_dg: Optional[float] = None,
    eval_all_hops: bool = False,
    sp_early_stop: bool = False,
    # SP caching/approx options
    sp_cache: bool = False,
    sp_cache_mode: str = "core",
    sp_pair_samples: int = 400,
    sp_verify_threshold: float = 0.05,
    # Optional DS-backed pairset service (before reuse / after save)
    pairset_service: Optional[SPPairsetService] = None,
    signature_builder: Optional[SignatureBuilder] = None,
) -> EvalResult:
    """Compute per-hop g(h) with greedy selection and return hop series and minima.

    - Uses fixed-pair SP (_compute_sp_gain_norm) with union/trim taken from core settings
    - GED normalized with denom_cmax_base（Cmax は実験設定の正規化スキームに依存:
      例: link/candidate ベースで c_node + |S_link|·c_edge 等）。
    - IG is base_ig; optionally recomputed per hop via linkset metrics when ig_recompute True
    """

    def _union_khop_nodes(graph_obj: nx.Graph, anchors_a: Set[Node], anchors_b: Set[Node], max_h: int) -> List[Set[Node]]:
        """Return list of node-sets for union of k-hop neighborhoods from two anchor sets for k=0..max_h.

        Caches expansion incrementally to avoid repeated BFS per hop.
        """
        if max_h < 0:
            return []
        # initialize
        a_sets: List[Set[Node]] = [set(anchors_a)]
        b_sets: List[Set[Node]] = [set(anchors_b)]
        a_frontier = set(anchors_a)
        b_frontier = set(anchors_b)
        for _ in range(max_h):
            # expand one hop for A
            a_next: Set[Node] = set()
            for n in a_frontier:
                try:
                    for m in graph_obj.neighbors(n):
                        if m not in a_sets[-1]:
                            a_next.add(m)
                except Exception:
                    continue
            a_sets.append(set(a_sets[-1]) | a_next)
            a_frontier = a_next
            # expand one hop for B
            b_next: Set[Node] = set()
            for n in b_frontier:
                try:
                    for m in graph_obj.neighbors(n):
                        if m not in b_sets[-1]:
                            b_next.add(m)
                except Exception:
                    continue
            b_sets.append(set(b_sets[-1]) | b_next)
            b_frontier = b_next
        # union per hop
        return [ (a_sets[h] | b_sets[h]) for h in range(max_h + 1) ]

    def _union_khop_subgraph(graph_obj: nx.Graph, anchors_a: Set[Node], anchors_b: Set[Node], hop: int) -> nx.Graph:
        # Fallback helper when building one-off subgraphs
        nodes_by_h = _union_khop_nodes(graph_obj, anchors_a, anchors_b, max(0, hop))
        nodes = nodes_by_h[max(0, hop)] if nodes_by_h else (set(anchors_a) | set(anchors_b))
        return graph_obj.subgraph(nodes).copy()

    def _sp_gain_fixed_pairs(sub_before: nx.Graph, sub_after: nx.Graph) -> float:
        # Delegate to core's SP implementation（厳密）
        scope = str(core.sp_scope_mode).lower()
        bound = str(core.sp_boundary_mode).lower()
        g1 = sub_before
        g2 = sub_after
        if scope in ("union", "merge", "superset"):
            all_nodes = set(g1.nodes()) | set(g2.nodes())
            g1 = sub_before.subgraph(all_nodes).copy()
            g2 = sub_after.subgraph(all_nodes).copy()
        if bound in ("trim", "terminal", "nodes"):
            g1 = core._trim_terminal_edges(g1, anchors_core, 1)
            g2 = core._trim_terminal_edges(g2, anchors_core, 1)
        try:
            return float(core._compute_sp_gain_norm(g1, g2, mode=core.sp_norm_mode))
        except Exception:
            return 0.0

    records_h: List[Tuple[int, float, float, float, float]] = []  # (h, g, ged, ig, sp)
    h_graph = stage_graph.copy()
    used_edges: Set[Tuple[Node, Node]] = set((min(u, v), max(u, v)) for u, v in h_graph.edges())
    chosen_edges_by_hop: List[Tuple[Node, Node]] = []
    g_best: Optional[float] = None
    h_best: int = 0

    # Precompute union k-hop node sets up to H_eval = max_hops + sp_hop_expand
    try:
        sp_expand = int(getattr(core, 'sp_hop_expand', 0))
    except Exception:
        sp_expand = 0
    H_eval = max(0, int(max_hops)) + max(0, sp_expand)
    before_nodes_by_h = _union_khop_nodes(g_before_for_expansion, anchors_core, anchors_top_before, H_eval)
    after_nodes_by_h = _union_khop_nodes(h_graph, anchors_core, anchors_top_after, H_eval)

    # h=0 evaluation
    nodes_b0 = before_nodes_by_h[0] if before_nodes_by_h else (set(anchors_core) | set(anchors_top_before))
    nodes_a0 = after_nodes_by_h[0] if after_nodes_by_h else (set(anchors_core) | set(anchors_top_after))
    sub_b0 = g_before_for_expansion.subgraph(nodes_b0).copy()
    sub_a0 = h_graph.subgraph(nodes_a0).copy()
    res0 = _norm_ged(sub_b0, sub_a0, node_cost=core.node_cost, edge_cost=core.edge_cost,
                     normalization=core.normalization, efficiency_weight=core.efficiency_weight,
                     enable_spectral=core.enable_spectral, spectral_weight=core.spectral_weight,
                     norm_override=denom_cmax_base) if denom_cmax_base > 0 else {"normalized_ged": 0.0, "raw_ged": 0.0, "normalization_den": float(denom_cmax_base or 0.0)}
    ged0 = float(min(1.0, max(0.0, res0.get("normalized_ged", 0.0))))
    ged0 = float(min(1.0, max(0.0, ged0)))
    raw_ged0 = float(res0.get("raw_ged", 0.0))
    added_edge_ops = 0  # EPC増分: 採用した追加エッジ数（prev_graph基準）
    sp0 = 0.0
    ig0 = base_ig + core.sp_beta * sp0
    g0v = float(ged0 - core.lambda_weight * ig0)
    records_h.append((0, g0v, ged0, ig0, sp0))
    g_best = g0v
    h_best = 0

    # AG gate: if g0 < theta_ag, skip multi-hop evaluation (strictly less)
    if theta_ag is not None:
        try:
            if float(g0v) < float(theta_ag):
                delta_ged = records_h[0][2]
                delta_ig = records_h[0][3]
                delta_sp = records_h[0][4]
                hop_series = [
                    {"hop": int(0), "g": float(g0v), "ged": float(ged0), "ig": float(ig0), "h": float(ig0), "sp": float(sp0)}
                ]
                return EvalResult(
                    hop_series=hop_series,
                    g0=float(g0v),
                    gmin=float(g0v),
                    best_hop=0,
                    delta_ged=float(delta_ged),
                    delta_ig=float(delta_ig),
                    delta_sp=float(delta_sp),
                    gmin_mh=float(g0v),
                    delta_ged_min_mh=float(delta_ged),
                    delta_ig_min_mh=float(delta_ig),
                    delta_sp_min_mh=float(delta_sp),
                    chosen_edges_by_hop=[],
                )
        except Exception:
            pass

    # Prepare SP cached-incremental helpers (per-eff-hop state)
    distcache = DistanceCache(mode="cached", pair_samples=int(max(1, sp_pair_samples)))
    pairs_by_eff: Dict[int, Any] = {}
    la_by_eff: Dict[int, List[float]] = {}
    lb_by_eff: Dict[int, float] = {}

    def _ensure_pairs_state(eff: int) -> Tuple[nx.Graph, float, List[Tuple[object, object, float]], str]:
        nodes_b = before_nodes_by_h[eff] if eff < len(before_nodes_by_h) else before_nodes_by_h[-1]
        sp_b = g_before_for_expansion.subgraph(nodes_b).copy()
        sig = distcache.signature(sp_b, anchors_core, eff, str(core.sp_scope_mode), str(core.sp_boundary_mode))
        if eff not in pairs_by_eff:
            # Try DS-backed pairset first
            ps_loaded = None
            if pairset_service is not None and signature_builder is not None:
                try:
                    sig2, meta2 = signature_builder.for_subgraph(sp_b, anchors_core, eff, str(core.sp_scope_mode), str(core.sp_boundary_mode))
                    # Prefer signature_builder's signature for DS purposes, but keep distcache sig for in-proc caching
                    ps_loaded = pairset_service.load(sig2)
                except Exception:
                    ps_loaded = None
            if ps_loaded is not None and ps_loaded.pairs:
                pairs = [( _id_to_node(pr.u_id), _id_to_node(pr.v_id), float(pr.d_before) ) for pr in ps_loaded.pairs]
                class _PS:  # minimal shim for local usage
                    def __init__(self, pr, lb):
                        self.pairs = pr; self.lb_avg = lb
                pairset_local = _PS(pairs, ps_loaded.lb_avg)
                pairs_by_eff[eff] = pairset_local  # type: ignore
                lb_by_eff[eff] = float(ps_loaded.lb_avg)
                la_by_eff[eff] = [d for (_, _, d) in pairs]
            else:
                pairset = distcache.get_fixed_pairs(sig, sp_b)
                pairs_by_eff[eff] = pairset
                lb_by_eff[eff] = float(pairset.lb_avg)
                la_by_eff[eff] = [d for (_, _, d) in pairset.pairs]
                # Save to DS for future reuse
                if pairset_service is not None and signature_builder is not None:
                    try:
                        sig2, meta2 = signature_builder.for_subgraph(sp_b, anchors_core, eff, str(core.sp_scope_mode), str(core.sp_boundary_mode))
                        pr = [PairRecord(u_id=f"{u[0]},{u[1]},{u[2]}", v_id=f"{v[0]},{v[1]},{v[2]}", d_before=float(d)) for (u, v, d) in pairset.pairs]
                        ps = Pairset(
                            signature=sig2,
                            lb_avg=float(pairset.lb_avg),
                            pairs=pr,
                            node_count=int(sp_b.number_of_nodes()),
                            edge_count=int(sp_b.number_of_edges()),
                            scope=str(core.sp_scope_mode),
                            boundary=str(core.sp_boundary_mode),
                            eff_hop=int(eff),
                            meta=meta2,
                        )
                        pairset_service.upsert(ps)
                    except Exception:
                        pass
        return sp_b, float(lb_by_eff[eff]), list(pairs_by_eff[eff].pairs), sig

    def _current_sp(eff: int) -> float:
        lb = float(lb_by_eff.get(eff, 0.0))
        la = la_by_eff.get(eff)
        if lb <= 0.0 or la is None or not la:
            return 0.0
        la_avg = sum(la) / float(len(la))
        return max(0.0, min(1.0, max(0.0, (lb - la_avg) / lb)))

    # Diagnostics counters
    sssp_calls_du_ct = 0
    sssp_calls_dv_ct = 0
    dv_leaf_skips_ct = 0
    cycle_verifies_ct = 0

    def _estimate_candidate_delta(eff: int, sp_b: nx.Graph, sig: str, e_u: Node, e_v: Node) -> Tuple[float, bool]:
        nonlocal sssp_calls_du_ct, sssp_calls_dv_ct, dv_leaf_skips_ct
        lb = float(lb_by_eff.get(eff, 0.0))
        base_sp = _current_sp(eff)
        la_cur = la_by_eff.get(eff) or []
        pairset = pairs_by_eff.get(eff)
        if lb <= 0.0 or pairset is None or not pairset.pairs or not la_cur:
            return (0.0, False)
        # Fast path: if e_v is not in before-subgraph (new leaf), SP(fixed-before)は変化しない
        if not sp_b.has_node(e_v):
            dv_leaf_skips_ct += 1
            return (0.0, False)
        du = distcache.get_sssp(sig, e_u, sp_b); sssp_calls_du_ct += 1
        dv = distcache.get_sssp(sig, e_v, sp_b); sssp_calls_dv_ct += 1
        # cycle closure if endpoints already connected
        cycle = (du.get(e_v) is not None)
        total = 0.0
        count = 0
        for idx, (a, b, dab) in enumerate(pairset.pairs):
            cur = la_cur[idx]
            alt = cur
            au = du.get(a)
            vb = dv.get(b)
            if au is not None and vb is not None:
                alt = min(alt, float(au + 1 + vb))
            av = dv.get(a)
            ub = du.get(b)
            if av is not None and ub is not None:
                alt = min(alt, float(av + 1 + ub))
            total += alt
            count += 1
        if count == 0:
            return (0.0, cycle)
        la_avg_new = total / float(count)
        sp_new = max(0.0, min(1.0, max(0.0, (lb - la_avg_new) / lb)))
        return (float(sp_new - base_sp), cycle)

    def _apply_best_edge(eff: int, sp_b: nx.Graph, sig: str, e_u: Node, e_v: Node) -> float:
        nonlocal sssp_calls_du_ct, sssp_calls_dv_ct
        lb = float(lb_by_eff.get(eff, 0.0))
        la_cur = la_by_eff.get(eff) or []
        pairset = pairs_by_eff.get(eff)
        if lb <= 0.0 or pairset is None or not pairset.pairs or not la_cur:
            return 0.0
        # New leaf fast path: SPは固定前ペアのみ対象のため変化しない
        if not sp_b.has_node(e_v):
            return _current_sp(eff)
        du = distcache.get_sssp(sig, e_u, sp_b); sssp_calls_du_ct += 1
        dv = distcache.get_sssp(sig, e_v, sp_b); sssp_calls_dv_ct += 1
        new_la: List[float] = []
        for idx, (a, b, dab) in enumerate(pairset.pairs):
            cur = la_cur[idx]
            alt = cur
            au = du.get(a)
            vb = dv.get(b)
            if au is not None and vb is not None:
                alt = min(alt, float(au + 1 + vb))
            av = dv.get(a)
            ub = du.get(b)
            if av is not None and ub is not None:
                alt = min(alt, float(av + 1 + ub))
            new_la.append(alt)
        la_by_eff[eff] = new_la
        return _current_sp(eff)

    # greedy hops
    for h in range(1, max(0, int(max_hops)) + 1):
        best_delta = 0.0
        best_item: Optional[Tuple[Node, Node, Dict[str, Any]]] = None
        eff_hop = h + int(max(0, int(getattr(core, 'sp_hop_expand', 0))))
        # evaluate δSP for each candidate
        for e_u, e_v, meta in ecand:
            key = (min(e_u, e_v), max(e_u, e_v))
            if key in used_edges:
                continue
            if sp_cache and str(sp_cache_mode).lower() == 'cached_incr':
                sp_b, lb, pairs, sig = _ensure_pairs_state(max(1, eff_hop))
                d_sp, _cycle = _estimate_candidate_delta(max(1, eff_hop), sp_b, sig, e_u, e_v)
                de = max(0.0, float(d_sp))
            else:
                g_try = h_graph.copy()
                if not g_try.has_node(e_u): g_try.add_node(e_u)
                if not g_try.has_node(e_v): g_try.add_node(e_v)
                if not g_try.has_edge(e_u, e_v): g_try.add_edge(e_u, e_v)
                sub_b = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, max(1, eff_hop))
                sub_a = _union_khop_subgraph(g_try, anchors_core, anchors_top_after, max(1, eff_hop))
                de = _sp_gain_fixed_pairs(sub_b, sub_a)
            if de > best_delta:
                best_delta = de; best_item = (e_u, e_v, meta)

        # adopt best edge if positive gain
        if best_item is not None and best_delta > 0.0:
            e_u, e_v, meta = best_item
            if not h_graph.has_edge(e_u, e_v):
                h_graph.add_edge(e_u, e_v)
            used_edges.add((min(e_u, e_v), max(e_u, e_v)))
            chosen_edges_by_hop.append((e_u, e_v))
            # EPC: update added_edge_ops if the edge did not exist in prev_graph
            try:
                if not prev_graph.has_edge(e_u, e_v):
                    added_edge_ops += 1
            except Exception:
                added_edge_ops += 1
        else:
            # No improving candidate at this hop
            # Early stop only when explicitly allowed (sp_early_stop) and not in full-eval mode
            if sp_early_stop and not eval_all_hops:
                break

        # compute g(h)
        eff_hop_eval = h + int(max(0, int(getattr(core, 'sp_hop_expand', 0))))
        he = max(1, eff_hop_eval)
        # Build subgraphs from cached node sets
        nodes_b = before_nodes_by_h[he] if he < len(before_nodes_by_h) else before_nodes_by_h[-1]
        nodes_a = after_nodes_by_h[he] if he < len(after_nodes_by_h) else after_nodes_by_h[-1]
        sub_b = g_before_for_expansion.subgraph(nodes_b).copy()
        sub_a = h_graph.subgraph(nodes_a).copy()
        # EPC増分（式(12)）: ΔGED(h) = (raw_ged0 + added_edge_ops * edge_cost) / Cmax
        if not ged_hop0_const and denom_cmax_base > 0:
            ged_h = float((raw_ged0 + added_edge_ops * float(getattr(core, 'edge_cost', 1.0))) / float(denom_cmax_base))
        else:
            ged_h = float(ged0)
        ged_h = float(min(1.0, max(0.0, ged_h)))
        # SP gain
        if sp_cache and str(sp_cache_mode).lower() == 'cached_incr' and best_item is not None:
            sp_b, lb, pairs, sig = _ensure_pairs_state(he)
            prev_sp = _current_sp(he)
            # apply chosen edge incrementally
            sp_fast = _apply_best_edge(he, sp_b, sig, best_item[0], best_item[1])
            # verify on suspected cycle and large gain
            add_nodes = max(0, sub_a.number_of_nodes() - sub_b.number_of_nodes())
            add_edges = max(0, sub_a.number_of_edges() - sub_b.number_of_edges())
            suspected_cycle = add_edges > add_nodes
            d_sp_gain = float(max(0.0, sp_fast - prev_sp))
            if suspected_cycle and (d_sp_gain >= float(sp_verify_threshold)):
                cycle_verifies_ct += 1
                sp_h = _sp_gain_fixed_pairs(sub_b, sub_a)
            else:
                sp_h = sp_fast
        else:
            sp_h = _sp_gain_fixed_pairs(sub_b, sub_a)
        ig_h_val = base_ig
        if ig_recompute and pre_linkset_info is not None:
            try:
                ls_h = core._compute_linkset_metrics(prev_graph, h_graph, pre_linkset_info, query_vector=query_vec, ig_fixed_den=ig_fixed_den)
                ig_h_val = float(ls_h.delta_h_norm)
            except Exception:
                ig_h_val = base_ig
        ig_h = ig_h_val + core.sp_beta * sp_h
        g_h = float(ged_h - core.lambda_weight * ig_h)
        records_h.append((h, g_h, ged_h, ig_h, sp_h))
        if g_best is None or g_h < g_best:
            g_best = g_h; h_best = h
        # Early stop on DG threshold if enabled and not in diagnostic full-eval mode
        try:
            if (not eval_all_hops) and (theta_dg is not None) and (float(g_h) < float(theta_dg)):
                break
        except Exception:
            pass

    # summarize
    delta_ged = records_h[0][2]
    delta_ig = records_h[0][3]
    delta_sp = records_h[0][4]
    g0 = records_h[0][1]
    gmin = g_best if g_best is not None else g0
    # Note: for now we do not include raw_ged/den per-hop in the output due to the inline tuple structure above.
    # We will reconstruct raw/den for hop0 and the final hop (best) using res0/res_h if needed in future.
    hop_series = [
        {"hop": int(h), "g": float(g), "ged": float(ged), "ig": float(ig), "h": float(delta_ig), "sp": float(sp)}
        for (h, g, ged, ig, sp) in records_h
    ]
    # mh-only minima
    gvals_mh = [(h, g, ged, ig, sp) for (h, g, ged, ig, sp) in records_h if h >= 1]
    if gvals_mh:
        h_mh, gmin_mh_val, ged_mh_val, ig_mh_val, sp_mh_val = min(gvals_mh, key=lambda t: t[1])
    else:
        h_mh, gmin_mh_val, ged_mh_val, ig_mh_val, sp_mh_val = (0, g0, delta_ged, delta_ig, delta_sp)

    # Persist SPafter pairset for next-step reuse (best-hop neighborhood)
    try:
        if pairset_service is not None and signature_builder is not None:
            he_best = max(1, int(h_best + int(max(0, int(getattr(core, 'sp_hop_expand', 0))))))
            nodes_b = before_nodes_by_h[he_best] if he_best < len(before_nodes_by_h) else before_nodes_by_h[-1]
            nodes_a = after_nodes_by_h[he_best] if he_best < len(after_nodes_by_h) else after_nodes_by_h[-1]
            sub_a_best = h_graph.subgraph(nodes_a).copy()
            sig_after, meta_after = signature_builder.for_subgraph(sub_a_best, anchors_core, he_best, str(core.sp_scope_mode), str(core.sp_boundary_mode))
            # sample pairs on after graph (reuse distcache sampler)
            ps_after = distcache.get_fixed_pairs(sig_after, sub_a_best)
            pr = [PairRecord(u_id=f"{u[0]},{u[1]},{u[2]}", v_id=f"{v[0]},{v[1]},{v[2]}", d_before=float(d)) for (u, v, d) in ps_after.pairs]
            pairset_service.upsert(Pairset(
                signature=sig_after,
                lb_avg=float(ps_after.lb_avg),
                pairs=pr,
                node_count=int(sub_a_best.number_of_nodes()),
                edge_count=int(sub_a_best.number_of_edges()),
                scope=str(core.sp_scope_mode),
                boundary=str(core.sp_boundary_mode),
                eff_hop=int(he_best),
                meta=meta_after,
            ))
    except Exception:
        pass

    return EvalResult(
        hop_series=hop_series,
        g0=g0,
        gmin=float(gmin),
        best_hop=int(h_best),
        delta_ged=float(delta_ged),
        delta_ig=float(delta_ig),
        delta_sp=float(delta_sp),
        gmin_mh=float(gmin_mh_val),
        delta_ged_min_mh=float(ged_mh_val),
        delta_ig_min_mh=float(ig_mh_val),
        delta_sp_min_mh=float(sp_mh_val),
        chosen_edges_by_hop=chosen_edges_by_hop,
        sssp_calls_du=int(sssp_calls_du_ct),
        sssp_calls_dv=int(sssp_calls_dv_ct),
        dv_leaf_skips=int(dv_leaf_skips_ct),
        cycle_verifies=int(cycle_verifies_ct),
    )
