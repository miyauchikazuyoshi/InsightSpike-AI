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
import numpy as np

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
    # Optional carry-over state for ALL-PAIRS-EXACT (per eff-hop APSP matrices)
    apsp_carry: Optional[Dict[int, Dict[str, Any]]] = None


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
    sp_allpairs: bool = False,
    sp_allpairs_exact: bool = False,
    # Optional DS-backed pairset service (before reuse / after save)
    pairset_service: Optional[SPPairsetService] = None,
    signature_builder: Optional[SignatureBuilder] = None,
    # Exact ALL-PAIRS incremental helpers
    apsp_carry: Optional[Dict[int, Dict[str, Any]]] = None,
    sp_exact_stable_nodes: bool = False,
    # Treat ΔSP as signed (no clamp to [0,1]) when True
    sp_signed: bool = False,
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

    def _collect_features(g: nx.Graph, default_dim: int = 8) -> np.ndarray:
        """Collect per-node features aligned to graph node order.

        Uses node['abs_vector'] or node['vector']; pads/truncates to default_dim.
        """
        feats: List[List[float]] = []
        for n in g.nodes():
            d = g.nodes[n]
            vec = d.get("abs_vector") or d.get("vector")
            if vec is None:
                feats.append([0.0] * default_dim)
                continue
            try:
                arr = list(vec)
            except Exception:
                arr = []
            if len(arr) < default_dim:
                arr = arr + [0.0] * (default_dim - len(arr))
            elif len(arr) > default_dim:
                arr = arr[:default_dim]
            feats.append([float(x) for x in arr])
        return np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, default_dim), dtype=np.float32)

    def _derive_query_similarity(sub_g: nx.Graph, query_vector: Optional[List[float]]) -> float:
        """Estimate a hop-dependent similarity weight for query_entry using subgraph features.

        Produces a value in (0, 1], based on average cosine similarity between
        query_vector and node features in the current after-subgraph.
        """
        if query_vector is None:
            return 1.0
        try:
            fa = _collect_features(sub_g)
            if fa.size == 0:
                return 1.0
            q = np.asarray(list(query_vector), dtype=np.float32)
            if q.ndim == 0:
                q = q.reshape(1)
            # pad/trunc to match
            dim = fa.shape[1]
            if q.size < dim:
                q = np.concatenate([q, np.zeros(dim - q.size, dtype=np.float32)], axis=0)
            elif q.size > dim:
                q = q[:dim]
            # cosine sim ∈ [-1,1] → map to [0,1]
            qn = q / (np.linalg.norm(q) + 1e-9)
            fn = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-9)
            sims = (fn @ qn)
            sims01 = (sims + 1.0) * 0.5
            val = float(np.clip(np.mean(sims01), 0.0, 1.0))
            return max(1e-6, val)
        except Exception:
            return 1.0

    def _sp_gain_fixed_pairs(sub_before: nx.Graph, sub_after: nx.Graph, eff_hop: int = 1) -> float:
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
            g1 = core._trim_terminal_edges(g1, anchors_core, max(1, int(eff_hop)))
            g2 = core._trim_terminal_edges(g2, anchors_core, max(1, int(eff_hop)))
        try:
            return float(core._compute_sp_gain_norm(g1, g2, mode=core.sp_norm_mode))
        except Exception:
            return 0.0

    def _avg_sp_length(g: nx.Graph) -> float:
        try:
            n = g.number_of_nodes()
            if n < 2:
                return 0.0
            total = 0.0
            cnt = 0
            for u, dmap in nx.all_pairs_shortest_path_length(g):
                for v, d in dmap.items():
                    if v == u:
                        continue
                    if v <= u:
                        continue
                    total += float(d)
                    cnt += 1
            return (total / cnt) if cnt > 0 else 0.0
        except Exception:
            return 0.0

    def _sp_gain_allpairs(sub_before: nx.Graph, sub_after: nx.Graph, eff_h: int) -> float:
        scope = str(core.sp_scope_mode).lower()
        bound = str(core.sp_boundary_mode).lower()
        g1 = sub_before
        g2 = sub_after
        if scope in ("union", "merge", "superset"):
            all_nodes = set(g1.nodes()) | set(g2.nodes())
            g1 = sub_before.subgraph(all_nodes).copy()
            g2 = sub_after.subgraph(all_nodes).copy()
        if bound in ("trim", "terminal", "nodes"):
            g1 = core._trim_terminal_edges(g1, anchors_core, max(1, int(eff_h)))
            g2 = core._trim_terminal_edges(g2, anchors_core, max(1, int(eff_h)))
        Lb = _avg_sp_length(g1)
        La = _avg_sp_length(g2)
        if Lb <= 0.0:
            return 0.0
        rel = (Lb - La) / Lb
        return float(rel) if sp_signed else float(max(0.0, rel))

    # Exact ALL-PAIRS helpers (evaluation subgraph only)
    def _nodes_and_index(g: nx.Graph) -> Tuple[List[Node], Dict[Node, int]]:
        nodes = list(g.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        return nodes, idx

    def _apsp_sum_and_mat(g: nx.Graph) -> Tuple[float, int, List[List[int]]]:
        nodes, idx = _nodes_and_index(g)
        n = len(nodes)
        D: List[List[int]] = [[-1] * n for _ in range(n)]
        total = 0.0
        cnt = 0
        for i, src in enumerate(nodes):
            try:
                dmap = nx.single_source_shortest_path_length(g, src)
            except Exception:
                dmap = {}
            for j, dst in enumerate(nodes):
                d = dmap.get(dst)
                if d is None:
                    continue
                D[i][j] = int(d)
            for j in range(i + 1, n):
                d = D[i][j]
                if d is not None and d >= 0:
                    total += float(d)
                    cnt += 1
        return total, cnt, D

    def _delta_sum_with_edge(D: List[List[int]], du: List[int], dv: List[int]) -> float:
        # Return total sum over unordered pairs using best alt via a new edge (u,v)
        n = len(D)
        total_new = 0.0
        for i in range(n):
            Di = D[i]
            for j in range(i + 1, n):
                cur = Di[j]
                # alt via u->v or v->u
                alt = None
                a = du[i]; b = dv[j]
                if a >= 0 and b >= 0:
                    alt = a + 1 + b
                a2 = dv[i]; b2 = du[j]
                if a2 >= 0 and b2 >= 0:
                    t = a2 + 1 + b2
                    if alt is None or t < alt:
                        alt = t
                if cur >= 0 and (alt is None or cur <= alt):
                    total_new += float(cur)
                else:
                    total_new += float(alt if alt is not None else cur if cur >= 0 else 0.0)
        return total_new

    def _avg_sp_stats(g: nx.Graph) -> Tuple[float, int]:
        try:
            n = g.number_of_nodes()
            if n < 2:
                return 0.0, 0
            total = 0.0
            cnt = 0
            for u, dmap in nx.all_pairs_shortest_path_length(g):
                for v, d in dmap.items():
                    if v == u:
                        continue
                    if v <= u:
                        continue
                    total += float(d)
                    cnt += 1
            avg = (total / cnt) if cnt > 0 else 0.0
            return float(avg), int(cnt)
        except Exception:
            return 0.0, 0

    def _sp_gain_fixed_pairs_strict(sub_before: nx.Graph, sub_after: nx.Graph, eff: int) -> Tuple[float, float, float, int, int, List[Tuple[Node, Node, float, float]]]:
        """Compute ΔSP using the SAME pair set sampled on sub_before, with exact La on sub_after.

        Returns: (delta_sp_rel, Lb, La, pair_count)
        """
        try:
            # Align scope/boundary to core settings
            scope = str(core.sp_scope_mode).lower()
            bound = str(core.sp_boundary_mode).lower()
            g1 = sub_before
            g2 = sub_after
            if scope in ("union", "merge", "superset"):
                all_nodes = set(g1.nodes()) | set(g2.nodes())
                g1 = sub_before.subgraph(all_nodes).copy()
                g2 = sub_after.subgraph(all_nodes).copy()
            if bound in ("trim", "terminal", "nodes"):
                g1 = core._trim_terminal_edges(g1, anchors_core, max(1, int(eff)))
                g2 = core._trim_terminal_edges(g2, anchors_core, max(1, int(eff)))
            sig = distcache.signature(g1, anchors_core, max(1, int(eff)), str(core.sp_scope_mode), str(core.sp_boundary_mode))
            pairs_obj = distcache.get_fixed_pairs(sig, g1)
            pairs = list(pairs_obj.pairs)
            if not pairs or pairs_obj.lb_avg <= 0.0:
                return 0.0, 0.0, 0.0, 0, 0, []
            # SSSP on after graph for unique sources
            sources = []
            seen = set()
            for a, b, dab in pairs:
                if a not in seen:
                    seen.add(a)
                    sources.append(a)
            dmaps = {}
            for a in sources:
                try:
                    dmaps[a] = dict(nx.single_source_shortest_path_length(g2, a))
                except Exception:
                    dmaps[a] = {}
            total_la = 0.0; cnt = 0; improved = 0; examples: List[Tuple[Node, Node, float, float]] = []
            for a, b, dab in pairs:
                dmap = dmaps.get(a, {})
                da = dmap.get(b)
                la = float(dab) if da is None else float(da)
                total_la += la; cnt += 1
                try:
                    if la < float(dab):
                        improved += 1
                        if len(examples) < 3:
                            examples.append((a, b, float(dab), float(la)))
                except Exception:
                    pass
            la_avg = (total_la / cnt) if cnt > 0 else float(pairs_obj.lb_avg)
            rel = max(0.0, min(1.0, max(0.0, (float(pairs_obj.lb_avg) - la_avg) / float(pairs_obj.lb_avg))))
            return float(rel), float(pairs_obj.lb_avg), float(la_avg), int(cnt), int(improved), examples
        except Exception:
            return 0.0, 0.0, 0.0, 0, 0, []

    records_h: List[Tuple[int, float, float, float, float]] = []  # (h, g, ged, ig, sp)
    dh_values: List[float] = []  # ΔH per hop (after-before, normalized)
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
    dh_values.append(float(base_ig))
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
                    {"hop": int(0), "g": float(g0v), "ged": float(ged0), "ig": float(ig0), "h": float(base_ig), "sp": float(sp0)}
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
    # Allow special value sp_pair_samples <= 0 to mean "use ALL pairs"
    distcache = DistanceCache(mode="cached", pair_samples=int(sp_pair_samples))
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
        # If candidate endpoints are outside the before-subgraph, fall back to the
        # superset (g_before_for_expansion) and cache those SSSPs. This avoids
        # treating boundary-adjacent candidates as no-ops.
        sup_sig = distcache.signature(
            g_before_for_expansion, anchors_core, eff, str(core.sp_scope_mode), str(core.sp_boundary_mode)
        )
        # e_u side
        if not sp_b.has_node(e_u):
            du = distcache.get_sssp(sup_sig, e_u, g_before_for_expansion); sssp_calls_du_ct += 1
        else:
            du = distcache.get_sssp(sig, e_u, sp_b); sssp_calls_du_ct += 1
        # e_v side
        if not sp_b.has_node(e_v):
            dv = distcache.get_sssp(sup_sig, e_v, g_before_for_expansion); sssp_calls_dv_ct += 1
            if not dv:
                dv_leaf_skips_ct += 1
                return (0.0, False)
        else:
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
        rel = (lb - la_avg_new) / lb if lb > 0.0 else 0.0
        sp_new = rel if sp_signed else max(0.0, min(1.0, max(0.0, rel)))
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
    # For diagnostics: carry per-hop H_before/H_after when linkset IG is used
    h_before_vals: List[float] = []
    h_after_vals: List[float] = []
    # State for ALL-PAIRS exact (per effective hop) — optionally reused across steps
    apsp_state: Dict[int, Dict[str, Any]] = {}
    if sp_allpairs_exact and isinstance(apsp_carry, dict):
        # shallow copy reference for in-place update to be visible to caller
        apsp_state = apsp_carry  # type: ignore[assignment]

    for h in range(1, max(0, int(max_hops)) + 1):
        best_delta = 0.0
        best_item: Optional[Tuple[Node, Node, Dict[str, Any]]] = None
        eff_hop = h + int(max(0, int(getattr(core, 'sp_hop_expand', 0))))
        # evaluate δSP for each candidate
        if sp_allpairs_exact:
            # Build evaluation subgraphs for current eff-hop
            sb = _union_khop_subgraph(g_before_for_expansion, anchors_core, anchors_top_before, max(1, eff_hop))
            sa = _union_khop_subgraph(h_graph, anchors_core, anchors_top_after, max(1, eff_hop))
            scope = str(core.sp_scope_mode).lower(); bound = str(core.sp_boundary_mode).lower()
            if scope in ("union","merge","superset"):
                all_nodes = set(sb.nodes()) | set(sa.nodes())
                sb = sb.subgraph(all_nodes).copy(); sa = sa.subgraph(all_nodes).copy()
            if bound in ("trim","terminal","nodes"):
                sb = core._trim_terminal_edges(sb, anchors_core, max(1, eff_hop))
                sa = core._trim_terminal_edges(sa, anchors_core, max(1, eff_hop))
            # Prepare Lb (before) and current La matrix on SA
            st = apsp_state.get(eff_hop)
            # Optional: stabilize node set (monotonic) by reusing previous SA nodes when enabled
            if sp_exact_stable_nodes and st is not None and isinstance(st.get('nodes_sa'), list):
                try:
                    prev_nodes = set(st.get('nodes_sa'))
                    cur_nodes = set(sa.nodes())
                    # union of previous and current
                    merged = list(prev_nodes | cur_nodes)
                    sa = sa.subgraph(merged).copy()
                    if bound in ("trim","terminal","nodes"):
                        sa = core._trim_terminal_edges(sa, anchors_core, max(1, eff_hop))
                except Exception:
                    pass
            # Rebuild if missing or node_count changed
            if st is None or st.get('node_count') != sa.number_of_nodes():
                Lb_sum, Lb_cnt, _ = _apsp_sum_and_mat(sb)
                Sa_sum, Sa_cnt, Dcur = _apsp_sum_and_mat(sa)
                st = {
                    'Lb_sum': Lb_sum,
                    'Lb_cnt': Lb_cnt,
                    'sum': Sa_sum,
                    'cnt': Sa_cnt,
                    'D': Dcur,
                    'sa': sa,
                    'node_count': sa.number_of_nodes(),
                    'nodes_sa': list(sa.nodes()),
                }
                apsp_state[eff_hop] = st
            nodes_sa, idx_sa = _nodes_and_index(st['sa'])
            n = len(nodes_sa)
            best_new_sum = None
            for e_u, e_v, meta in ecand:
                key = (min(e_u, e_v), max(e_u, e_v))
                if key in used_edges:
                    continue
                if (e_u not in idx_sa) or (e_v not in idx_sa):
                    continue
                try:
                    du_map = nx.single_source_shortest_path_length(st['sa'], e_u)
                    dv_map = nx.single_source_shortest_path_length(st['sa'], e_v)
                except Exception:
                    continue
                du = [ -1 ] * n; dv = [ -1 ] * n
                for k, node in enumerate(nodes_sa):
                    a = du_map.get(node); b = dv_map.get(node)
                    du[k] = int(a) if a is not None else -1
                    dv[k] = int(b) if b is not None else -1
                new_sum = _delta_sum_with_edge(st['D'], du, dv)
                if (best_new_sum is None) or (new_sum < best_new_sum):
                    best_new_sum = new_sum
                    best_item = (e_u, e_v, meta)
                    best_delta = float(max(0.0, ((st['sum'] / st['cnt']) - (new_sum / st['cnt'])) / (st['sum'] / st['cnt']))) if st['cnt'] > 0 and st['sum'] > 0 else 0.0
        else:
            for e_u, e_v, meta in ecand:
                key = (min(e_u, e_v), max(e_u, e_v))
                if key in used_edges:
                    continue
                if (not sp_allpairs) and sp_cache and str(sp_cache_mode).lower() == 'cached_incr':
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
                    if sp_allpairs:
                        de = _sp_gain_allpairs(sub_b, sub_a, eff_hop)
                    else:
                        de_fp = _sp_gain_fixed_pairs(sub_b, sub_a, eff_hop=max(1, eff_hop))
                        add_nodes_c = max(0, sub_a.number_of_nodes() - sub_b.number_of_nodes())
                        add_edges_c = max(0, sub_a.number_of_edges() - sub_b.number_of_edges())
                        if (add_edges_c > add_nodes_c) or (de_fp <= 0.0):
                            Lb = _avg_sp_length(sub_b)
                            La = _avg_sp_length(sub_a)
                            de = float(max(0.0, (Lb - La) / Lb)) if Lb > 0.0 else 0.0
                        else:
                            de = de_fp
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
            # If eval_all_hops is True, adopt one candidate (even if ΔSP<=0) for diagnostic greedy
            if eval_all_hops:
                # pick first unused candidate
                picked = None
                for e_u, e_v, meta in ecand:
                    key = (min(e_u, e_v), max(e_u, e_v))
                    if key in used_edges:
                        continue
                    picked = (e_u, e_v, meta); break
                if picked is not None:
                    e_u, e_v, meta = picked
                    if not h_graph.has_edge(e_u, e_v):
                        h_graph.add_edge(e_u, e_v)
                    used_edges.add((min(e_u, e_v), max(e_u, e_v)))
                    chosen_edges_by_hop.append((e_u, e_v))
                else:
                    if sp_early_stop:
                        break
            else:
                # Early stop only when explicitly allowed (sp_early_stop) and not in full-eval mode
                if sp_early_stop:
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
        sp_mode_used = 'fp'
        sp_lb_val = None  # type: Optional[float]
        sp_la_val = None  # type: Optional[float]
        sp_pair_cnt = None  # type: Optional[int]
        if (not sp_allpairs) and (not sp_allpairs_exact) and sp_cache and str(sp_cache_mode).lower() == 'cached_incr' and best_item is not None:
            sp_b, lb, pairs, sig = _ensure_pairs_state(he)
            prev_sp = _current_sp(he)
            # apply chosen edge incrementally
            sp_fast = _apply_best_edge(he, sp_b, sig, best_item[0], best_item[1])
            # verify on suspected cycle and large gain
            add_nodes = max(0, sub_a.number_of_nodes() - sub_b.number_of_nodes())
            add_edges = max(0, sub_a.number_of_edges() - sub_b.number_of_edges())
            suspected_cycle = add_edges > add_nodes
            d_sp_gain = float((sp_fast - prev_sp) if sp_signed else max(0.0, sp_fast - prev_sp))
            sp_mode_used = 'cached_incr'
            sp_lb_val = float(lb)
            try:
                # infer La from current la_by_eff
                la_cur = la_by_eff.get(he)
                if la_cur:
                    sp_la_val = float(sum(la_cur) / float(len(la_cur)))
            except Exception:
                pass
            try:
                sp_pair_cnt = int(len(pairs) if pairs is not None else 0)
            except Exception:
                sp_pair_cnt = None
            if suspected_cycle and (d_sp_gain >= float(sp_verify_threshold)):
                cycle_verifies_ct += 1
                # Recompute by STRICT fixed-before pair set on after graph
                rel2, Lb2, La2, Pc2, Imp2, Ex2 = _sp_gain_fixed_pairs_strict(sub_b, sub_a, he)
                sp_h = float(rel2)
                sp_mode_used = 'fp_verify'
                sp_lb_val = float(Lb2); sp_la_val = float(La2); sp_pair_cnt = int(Pc2)
                sp_imp_cnt = int(Imp2); sp_imp_ex = Ex2
            else:
                sp_h = sp_fast
        else:
            if sp_allpairs or sp_allpairs_exact:
                # Force ALL-PAIRS average SP for hop evaluation
                if sp_allpairs_exact:
                    # use stored APSP state updated by adoption
                    st = apsp_state.get(he)
                    if st is None:
                        Lb2, Pc2 = _avg_sp_stats(sub_b)
                        La2, _ = _avg_sp_stats(sub_a)
                        sp_lb_val = float(Lb2); sp_la_val = float(La2); sp_pair_cnt = int(Pc2)
                        sp_h = _sp_gain_allpairs(sub_b, sub_a, he)
                    else:
                        Lb_avg = (st['Lb_sum'] / float(st['Lb_cnt'])) if st['Lb_cnt'] > 0 else 0.0
                        La_avg = (st['sum'] / float(st['cnt'])) if st['cnt'] > 0 else 0.0
                        if Lb_avg > 0.0:
                            rel = (Lb_avg - La_avg) / Lb_avg
                            sp_h = float(rel) if sp_signed else float(max(0.0, rel))
                        else:
                            sp_h = 0.0
                        sp_lb_val = float(Lb_avg); sp_la_val = float(La_avg); sp_pair_cnt = int(st['cnt'])
                    sp_mode_used = 'allpairs_exact'
                    sp_imp_cnt = 0; sp_imp_ex = []
                else:
                    sp_h = _sp_gain_allpairs(sub_b, sub_a, he)
                    sp_mode_used = 'allpairs_forced'
                    Lb2, Pc2 = _avg_sp_stats(sub_b)
                    La2, _ = _avg_sp_stats(sub_a)
                    sp_lb_val = float(Lb2); sp_la_val = float(La2); sp_pair_cnt = int(Pc2)
                    sp_imp_cnt = 0; sp_imp_ex = []
            else:
                # Default: fixed-before-pairs; on compressive structure OR no fixed-pair gain, fall back to ALL-PAIRS-like strict eval
                sp_fp = _sp_gain_fixed_pairs(sub_b, sub_a, eff_hop=he)
                add_nodes = max(0, sub_a.number_of_nodes() - sub_b.number_of_nodes())
                add_edges = max(0, sub_a.number_of_edges() - sub_b.number_of_edges())
            if (add_edges > add_nodes) or (sp_fp <= 0.0 and not sp_signed) or (sp_signed and sp_fp <= 1e-12):
                rel2, Lb2, La2, Pc2, Imp2, Ex2 = _sp_gain_fixed_pairs_strict(sub_b, sub_a, he)
                sp_h = float(rel2)
                sp_mode_used = 'fp_strict'
                sp_lb_val = float(Lb2); sp_la_val = float(La2); sp_pair_cnt = int(Pc2)
                sp_imp_cnt = int(Imp2); sp_imp_ex = Ex2
                # If improvements detected, refresh the fixed pairset to AFTER graph for next hops
                try:
                    if sp_imp_cnt > 0:
                        # Build AFTER-based pairset and install as new baseline for eff-hop
                        sig_after = distcache.signature(sub_a, anchors_core, he, str(core.sp_scope_mode), str(core.sp_boundary_mode))
                        ps_after = distcache.get_fixed_pairs(sig_after, sub_a)
                        pairs_by_eff[he] = ps_after
                        lb_by_eff[he] = float(ps_after.lb_avg)
                        la_by_eff[he] = [d for (_, _, d) in ps_after.pairs]
                        sp_mode_used = 'fp_strict_update'
                except Exception:
                    pass
                else:
                    sp_h = sp_fp
                    sp_mode_used = 'fp'
                    sp_imp_cnt = 0; sp_imp_ex = []
        # Refresh fixed pairset baseline to AFTER graph for next iterations (fp modesのみ)
        try:
            if not (sp_allpairs or sp_allpairs_exact):
                sig_after_he = distcache.signature(sub_a, anchors_core, he, str(core.sp_scope_mode), str(core.sp_boundary_mode))
                ps_after_he = distcache.get_fixed_pairs(sig_after_he, sub_a)
                pairs_by_eff[he] = ps_after_he
                lb_by_eff[he] = float(ps_after_he.lb_avg)
                la_by_eff[he] = [d for (_, _, d) in ps_after_he.pairs]
                if sp_mode_used == 'fp':
                    sp_mode_used = 'fp_update'
        except Exception:
            pass

        ig_h_val = base_ig
        h_before_cur: Optional[float] = None
        h_after_cur: Optional[float] = None
        if ig_recompute:
            src_mode = str(getattr(core, 'ig_source_mode', 'graph')).lower()
            if (pre_linkset_info is not None) and (src_mode in ("linkset", "paper", "strict")):
                # Paper/strict: per-hop, recompute ΔH via linkset metrics
                try:
                    # Rebuild a hop-aware linkset payload with updated query similarity
                    ls_info = dict(pre_linkset_info)
                    q_entry = dict(ls_info.get('query_entry') or {})
                    q_entry['similarity'] = _derive_query_similarity(sub_a, query_vec)
                    ls_info['query_entry'] = q_entry
                    ls_h = core._compute_linkset_metrics(prev_graph, h_graph, ls_info, query_vector=query_vec, ig_fixed_den=ig_fixed_den)  # type: ignore[attr-defined]
                    ig_h_val = float(ls_h.delta_h_norm)
                    try:
                        h_before_cur = float(getattr(ls_h, 'entropy_before', None))
                        h_after_cur = float(getattr(ls_h, 'entropy_after', None))
                    except Exception:
                        h_before_cur = None; h_after_cur = None
                except Exception:
                    ig_h_val = base_ig
            else:
                # Graph entropy fallback; align normalization with K★ if available
                try:
                    fb = _collect_features(sub_b)
                    fa = _collect_features(sub_a)
                    k_star = None
                    if isinstance(pre_linkset_info, dict):
                        try:
                            s_link = pre_linkset_info.get('s_link') or []
                            pool = pre_linkset_info.get('candidate_pool') or []
                            k_star = max(2, int(len(pool) or 0) or (int(len(s_link) or 0) + 1))
                        except Exception:
                            k_star = None
                    ig_res = core._calculate_entropy_variance_ig(  # type: ignore[attr-defined]
                        sub_b, fb, fa, query_vector=query_vec, fixed_den=ig_fixed_den, k_star=k_star,
                    )
                    ig_h_val = float(ig_res.get("ig_value", base_ig))
                except Exception:
                    ig_h_val = base_ig
        ig_h = ig_h_val + core.sp_beta * sp_h
        g_h = float(ged_h - core.lambda_weight * ig_h)
        records_h.append((h, g_h, ged_h, ig_h, sp_h))
        dh_values.append(float(ig_h_val))
        if h_before_cur is not None:
            h_before_vals.append(float(h_before_cur))
        else:
            h_before_vals.append(0.0)
        if h_after_cur is not None:
            h_after_vals.append(float(h_after_cur))
        else:
            h_after_vals.append(0.0)
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
    hop_series = []
    for idx, (h, g, ged, ig, sp) in enumerate(records_h):
        dh = dh_values[idx] if idx < len(dh_values) else 0.0
        row = {"hop": int(h), "g": float(g), "ged": float(ged), "ig": float(ig), "h": float(dh), "sp": float(sp)}
        # Attach H_before/H_after when available (per-hop linkset IG); include zeros for clarity
        try:
            if idx < len(h_before_vals) and idx < len(h_after_vals):
                row['h_before'] = float(h_before_vals[idx])
                row['h_after'] = float(h_after_vals[idx])
        except Exception:
            pass
        # Attach SP diagnostics: mode, Lb/La estimates, pair count, and structure deltas
        try:
            if sp_mode_used:
                row['sp_mode'] = str(sp_mode_used)
            if sp_lb_val is not None:
                row['sp_Lb'] = float(sp_lb_val)
            if sp_la_val is not None:
                row['sp_La'] = float(sp_la_val)
            if sp_pair_cnt is not None:
                row['sp_pairs'] = int(sp_pair_cnt)
            if 'sp_imp_cnt' not in locals():
                sp_imp_cnt = 0; sp_imp_ex = []
            row['sp_improved'] = int(sp_imp_cnt)
            if sp_imp_ex:
                try:
                    row['sp_examples'] = [
                        {
                            'u': [int(a[0]), int(a[1]), int(a[2])],
                            'v': [int(b[0]), int(b[1]), int(b[2])],
                            'd_before': float(db), 'd_after': float(da)
                        } for (a, b, db, da) in sp_imp_ex
                    ]
                except Exception:
                    pass
            # Structural deltas used in detection
            try:
                row['add_nodes'] = int(max(0, sub_a.number_of_nodes() - sub_b.number_of_nodes()))
                row['add_edges'] = int(max(0, sub_a.number_of_edges() - sub_b.number_of_edges()))
            except Exception:
                pass
        except Exception:
            pass
        hop_series.append(row)
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
        apsp_carry=apsp_state if sp_allpairs_exact else None,
    )
