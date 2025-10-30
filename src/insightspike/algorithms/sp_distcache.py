"""
DistanceCache for SP gain evaluation.

Provides a cache-backed pathway to estimate ΔSP for candidate edges using:
- Fixed-before-pairs (sampled) on the before-subgraph
- SSSP (BFS) distance maps for candidate endpoints
- Relative gain ΔSP = max(0, (Lb - La) / Lb)

Modes:
- core: delegate to GeDIGCore._compute_sp_gain_norm (keeps current semantics)
- cached: use cached pairs + SSSP to estimate ΔSP (experimental)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set
import hashlib

import networkx as nx


def _hash_nodes_edges(g: nx.Graph) -> str:
    nodes = ",".join(sorted(str(n) for n in g.nodes()))
    edges = ",".join(sorted(f"{a}|{b}" for a, b in (tuple(sorted((u, v))) for u, v in g.edges())))
    h = hashlib.sha256()
    h.update(nodes.encode("utf-8"))
    h.update(b"#")
    h.update(edges.encode("utf-8"))
    return h.hexdigest()


@dataclass
class PairSet:
    pairs: List[Tuple[object, object, float]]  # (u, v, d_before)
    lb_avg: float


class DistanceCache:
    def __init__(
        self,
        *,
        mode: str = "core",  # 'core' | 'cached'
        pair_samples: int = 400,
    ) -> None:
        self.mode = str(mode)
        self.pair_samples = int(max(1, pair_samples))
        self._pair_cache: Dict[str, PairSet] = {}
        self._sssp_cache: Dict[Tuple[str, object], Dict[object, int]] = {}

    def signature(self, g_before: nx.Graph, anchors: Set[object], hop: int, scope: str, boundary: str) -> str:
        base = _hash_nodes_edges(g_before)
        meta = f"|A:{len(anchors)}|H:{hop}|S:{scope}|B:{boundary}"
        return base + meta

    def _sample_pairs(self, g: nx.Graph) -> PairSet:
        # exact all-pairs for small graphs; sampling for larger graphs
        n = g.number_of_nodes()
        if n < 2:
            return PairSet([], 0.0)
        pairs: List[Tuple[object, object, float]] = []
        total = 0.0
        count = 0
        try:
            # collect up to pair_samples pairs
            for u, dmap in nx.all_pairs_shortest_path_length(g):
                for v, d in dmap.items():
                    if v == u:
                        continue
                    if str(v) <= str(u):
                        continue
                    pairs.append((u, v, float(d)))
                    total += float(d)
                    count += 1
                    if count >= self.pair_samples:
                        lb = (total / count) if count > 0 else 0.0
                        return PairSet(pairs, lb)
        except Exception:
            pass
        lb = (total / count) if count > 0 else 0.0
        return PairSet(pairs, lb)

    def get_fixed_pairs(self, sig: str, g_before: nx.Graph) -> PairSet:
        ps = self._pair_cache.get(sig)
        if ps is not None:
            return ps
        ps = self._sample_pairs(g_before)
        self._pair_cache[sig] = ps
        return ps

    def get_sssp(self, sig: str, src: object, g_before: nx.Graph) -> Dict[object, int]:
        key = (sig, src)
        dm = self._sssp_cache.get(key)
        if dm is not None:
            return dm
        try:
            dm = dict(nx.single_source_shortest_path_length(g_before, src))
        except Exception:
            dm = {}
        self._sssp_cache[key] = dm
        return dm

    def estimate_sp_cached(
        self,
        *,
        sig: str,
        g_before: nx.Graph,
        pairs: PairSet,
        endpoint_u: object,
        endpoint_v: object,
    ) -> float:
        if not pairs.pairs or pairs.lb_avg <= 0.0:
            return 0.0
        du = self.get_sssp(sig, endpoint_u, g_before)
        dv = self.get_sssp(sig, endpoint_v, g_before)
        total = 0.0
        count = 0
        for a, b, dab in pairs.pairs:
            # La = min(dab, d(a,u)+1+d(v,b), d(a,v)+1+d(u,b)) when paths exist
            cand = [dab]
            au = du.get(a); vb = dv.get(b)
            if au is not None and vb is not None:
                cand.append(float(au + 1 + vb))
            av = dv.get(a); ub = du.get(b)
            if av is not None and ub is not None:
                cand.append(float(av + 1 + ub))
            la = min(cand) if cand else dab
            total += la
            count += 1
        if count == 0:
            return 0.0
        la_avg = total / count
        gain = max(0.0, pairs.lb_avg - la_avg)
        return max(0.0, min(1.0, gain / pairs.lb_avg))

