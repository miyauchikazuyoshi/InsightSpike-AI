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
from typing import Dict, Tuple, List, Optional, Set, Protocol
import hashlib
import json
import os

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
        # Allow special value pair_samples <= 0 to mean "use ALL pairs"
        # (suitable for parity checks and small subgraphs)
        self.pair_samples = int(pair_samples)
        self._pair_cache: Dict[str, PairSet] = {}
        self._sssp_cache: Dict[Tuple[str, object], Dict[object, int]] = {}
        # Optional persistent registry for PairSet reuse (signature -> PairSet)
        self._registry: Optional[_PairsetRegistry] = _create_registry_from_env()

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
            # collect pairs; if pair_samples <= 0, collect ALL pairs
            cap = None if self.pair_samples <= 0 else int(self.pair_samples)
            for u, dmap in nx.all_pairs_shortest_path_length(g):
                for v, d in dmap.items():
                    if v == u:
                        continue
                    if str(v) <= str(u):
                        continue
                    pairs.append((u, v, float(d)))
                    total += float(d)
                    count += 1
                    if cap is not None and count >= cap:
                        lb = (total / count) if count > 0 else 0.0
                        return PairSet(pairs, lb)
        except Exception:
            pass
        lb = (total / count) if count > 0 else 0.0
        return PairSet(pairs, lb)

    def get_fixed_pairs(self, sig: str, g_before: nx.Graph) -> PairSet:
        # 1) in-proc cache
        ps = self._pair_cache.get(sig)
        if ps is not None:
            return ps
        # 2) registry
        if self._registry is not None:
            loaded = self._registry.load(sig)
            if loaded is not None:
                self._pair_cache[sig] = loaded
                return loaded
        # 3) sample now
        ps = self._sample_pairs(g_before)
        self._pair_cache[sig] = ps
        # save to registry
        if self._registry is not None:
            try:
                self._registry.save(sig, ps)
            except Exception:
                pass
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

    def estimate_sp_between_graphs(
        self,
        *,
        sig: str,
        g_before: nx.Graph,
        g_after: nx.Graph,
    ) -> float:
        """Estimate ΔSP_rel between two graphs using fixed-before pairs.

        - Uses (up to) `pair_samples` shortest-path pairs sampled on g_before
        - Computes average shortest-path on g_after for the same pair set using SSSP per source
        - Returns relative signed gain: (Lb - La) / Lb, clamped to [0, 1]
        """
        pairs = self.get_fixed_pairs(sig, g_before)
        if not pairs.pairs or pairs.lb_avg <= 0.0:
            return 0.0

        # Pre-compute SSSP on after-graph for unique sources in pairs
        # Reuse the same signature to hit the SSSP cache when available
        sources: List[object] = []
        seen_src = set()
        for a, b, _ in pairs.pairs:
            if a not in seen_src:
                sources.append(a)
                seen_src.add(a)
        sssp_after: Dict[object, Dict[object, int]] = {}
        for a in sources:
            try:
                sssp_after[a] = dict(nx.single_source_shortest_path_length(g_after, a))
            except Exception:
                sssp_after[a] = {}

        total_la = 0.0
        count = 0
        for a, b, dab in pairs.pairs:
            dmap = sssp_after.get(a, {})
            dafter = dmap.get(b)
            if dafter is None:
                # If disconnected, assume no improvement over before distance
                la = float(dab)
            else:
                la = float(dafter)
            total_la += la
            count += 1
        if count == 0:
            return 0.0
        la_avg = total_la / count
        gain = max(0.0, pairs.lb_avg - la_avg)
        return max(0.0, min(1.0, gain / pairs.lb_avg))

    @staticmethod
    def current_sp_from_pairs(pairs: PairSet) -> float:
        """Compute current SP_rel for the before-graph from a PairSet.

        SP_rel = max(0, (Lb - La) / Lb), where La is the average of pair distances in the set.
        """
        if not pairs.pairs or pairs.lb_avg <= 0.0:
            return 0.0
        total = 0.0
        for _, _, dab in pairs.pairs:
            total += float(dab)
        la_avg = total / float(len(pairs.pairs))
        return max(0.0, min(1.0, max(0.0, (pairs.lb_avg - la_avg) / pairs.lb_avg)))


# ---------------- Registry (optional) ----------------
class _PairsetRegistry(Protocol):
    def load(self, signature: str) -> Optional[PairSet]: ...
    def save(self, signature: str, pairset: PairSet) -> None: ...


class _MemoryRegistry:
    def __init__(self) -> None:
        self._mem: Dict[str, Dict] = {}

    def load(self, signature: str) -> Optional[PairSet]:
        obj = self._mem.get(signature)
        if not obj:
            return None
        return _decode_pairset(obj)

    def save(self, signature: str, pairset: PairSet) -> None:
        self._mem[signature] = _encode_pairset(pairset)


class _FileRegistry:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_dir()
        self._cache: Optional[Dict[str, Dict]] = None

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _load_all(self) -> Dict[str, Dict]:
        if self._cache is not None:
            return self._cache
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._cache = data
                else:
                    self._cache = {}
        except Exception:
            self._cache = {}
        return self._cache

    def _flush(self) -> None:
        if self._cache is None:
            return
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    def load(self, signature: str) -> Optional[PairSet]:
        data = self._load_all().get(signature)
        if not data:
            return None
        return _decode_pairset(data)

    def save(self, signature: str, pairset: PairSet) -> None:
        data = self._load_all()
        data[signature] = _encode_pairset(pairset)
        self._flush()


def _encode_pairset(ps: PairSet) -> Dict:
    return {"lb_avg": float(ps.lb_avg), "pairs": [(u, v, float(d)) for (u, v, d) in ps.pairs]}


def _decode_pairset(obj: Dict) -> PairSet:
    lb = float(obj.get("lb_avg", 0.0))
    pairs_raw = obj.get("pairs", [])
    pairs: List[Tuple[object, object, float]] = []
    for it in pairs_raw:
        try:
            u, v, d = it
            pairs.append((u, v, float(d)))
        except Exception:
            continue
    return PairSet(pairs=pairs, lb_avg=lb)


def _create_registry_from_env() -> Optional[_PairsetRegistry]:
    path = os.getenv('INSIGHTSPIKE_SP_REGISTRY', '').strip()
    if path:
        try:
            return _FileRegistry(path)
        except Exception:
            return _MemoryRegistry()
    # No file path → use in-memory registry to allow cross-call reuse
    return _MemoryRegistry()
