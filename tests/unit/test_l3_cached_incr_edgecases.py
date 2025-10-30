from __future__ import annotations

import os
import numpy as np

from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner


def _make_docs(n: int = 5, dim: int = 8):
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    docs = []
    for i in range(n):
        v = base.copy()
        v[min(i, dim - 1)] += 0.1
        docs.append({"text": f"doc{i}", "embedding": v})
    return docs


def test_l3_cached_incr_empty_candidates_fallback():
    os.environ.pop('INSIGHTSPIKE_MIN_IMPORT', None)
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '30'
    docs = _make_docs()
    l3 = L3GraphReasoner()
    context = {
        "centers": [0],
        "candidate_edges": [],  # explicit empty
        "candidate_selection": {"k_star": 2, "l1_candidates": 4, "log_k_star": 0.693},
    }
    res = l3.analyze_documents(docs, context)
    metrics = res.get("metrics", {})
    # Should fallback to at least cached (or auto-candidate may promote to cached_incr)
    assert metrics.get("sp_engine") in ("cached", "cached_incr")


def test_l3_cached_incr_invalid_candidates_filtered():
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '30'
    docs = _make_docs()
    l3 = L3GraphReasoner()
    # invalid indices (-1, 999)
    context = {
        "centers": [0],
        "candidate_edges": [(-1, 2, {}), (0, 999, {})],
        "candidate_selection": {"k_star": 1},
    }
    res = l3.analyze_documents(docs, context)
    metrics = res.get("metrics", {})
    assert metrics.get("sp_engine") in ("cached", "cached_incr")


def test_l3_cached_incr_auto_candidates_from_centers():
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '30'
    os.environ['INSIGHTSPIKE_CAND_TOPK'] = '4'
    docs = _make_docs()
    l3 = L3GraphReasoner()
    context = {
        "centers": [0],
        # no candidate_edges on purpose -> auto-propose
        "candidate_selection": {"k_star": 2},
        "norm_spec": {
            "metric": "cosine",
            "radius_mode": "intuitive",
            "intuitive": {"outer": 0.6, "inner": 0.2},
            "dimension": 8,
            "scope": "sphere",
            "effective": {"theta_link": 0.3, "theta_cand": 0.4},
        },
    }
    res = l3.analyze_documents(docs, context)
    metrics = res.get("metrics", {})
    # Expect auto-candidates to enable cached_incr on small graphs
    assert metrics.get("sp_engine") in ("cached_incr", "cached")


def test_l3_cached_incr_budget_larger_than_candidates():
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '30'
    os.environ['INSIGHTSPIKE_SP_BUDGET'] = '5'  # larger than candidates
    docs = _make_docs()
    l3 = L3GraphReasoner()
    candidate_edges = [(0, 1, {"score": 0.9})]
    context = {
        "centers": [0],
        "candidate_edges": candidate_edges,
    }
    res = l3.analyze_documents(docs, context)
    metrics = res.get("metrics", {})
    assert metrics.get("sp_engine") in ("cached_incr", "cached")

