from __future__ import annotations

import os
import tempfile

import networkx as nx

from insightspike.algorithms.sp_distcache import DistanceCache


def _simple_graphs():
    g1 = nx.cycle_graph(6)
    g2 = g1.copy()
    g2.add_edge(0, 3)  # shortcut
    return g1, g2


def test_pairset_memory_registry_and_between_graphs():
    g1, g2 = _simple_graphs()
    cache = DistanceCache(mode="cached", pair_samples=50)
    sig = cache.signature(g1, anchors=set([0]), hop=1, scope='auto', boundary='trim')
    ps = cache.get_fixed_pairs(sig, g1)
    assert ps.lb_avg >= 0
    # second call hits in-proc cache
    ps2 = cache.get_fixed_pairs(sig, g1)
    assert len(ps2.pairs) == len(ps.pairs)
    sp_rel = cache.estimate_sp_between_graphs(sig=sig, g_before=g1, g_after=g2)
    assert 0.0 <= sp_rel <= 1.0


def test_pairset_file_registry_roundtrip():
    g1, _ = _simple_graphs()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'pairsets.json')
        os.environ['INSIGHTSPIKE_SP_REGISTRY'] = path
        cache = DistanceCache(mode="cached", pair_samples=20)
        sig = cache.signature(g1, anchors=set([0]), hop=1, scope='auto', boundary='trim')
        ps = cache.get_fixed_pairs(sig, g1)
        assert ps.lb_avg >= 0
        # new cache should load from file registry
        cache2 = DistanceCache(mode="cached", pair_samples=20)
        ps2 = cache2.get_fixed_pairs(sig, g1)
        assert len(ps2.pairs) == len(ps.pairs)
    os.environ.pop('INSIGHTSPIKE_SP_REGISTRY', None)

