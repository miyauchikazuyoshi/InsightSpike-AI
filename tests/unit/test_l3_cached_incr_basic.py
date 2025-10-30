from __future__ import annotations

import os
import numpy as np

from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner


def test_l3_cached_incr_with_candidate_edges():
    # Ensure non-lite path (but without torch, L3 provides fallback Data)
    os.environ.pop('INSIGHTSPIKE_MIN_IMPORT', None)
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '50'
    os.environ['INSIGHTSPIKE_SP_BUDGET'] = '1'

    # Create simple 5-node embeddings with clear similarity around node 0
    dim = 8
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    docs = []
    for i in range(5):
        v = base.copy()
        v[min(i, dim-1)] += 0.1
        docs.append({"text": f"doc{i}", "embedding": v})

    l3 = L3GraphReasoner()
    centers = [0]
    # candidates: edges from center 0 to others, sorted by simple assumption
    candidate_edges = [(0, 1, {"score": 0.9}), (0, 2, {"score": 0.8})]
    context = {
        "centers": centers,
        "candidate_edges": candidate_edges,
        "candidate_selection": {"k_star": 2, "l1_candidates": 4, "log_k_star": 0.693},
        "norm_spec": {
            "metric": "cosine",
            "radius_mode": "intuitive",
            "intuitive": {"outer": 0.6, "inner": 0.2},
            "dimension": dim,
            "scope": "sphere",
            "effective": {"theta_link": 0.3, "theta_cand": 0.4},
        },
    }

    res = l3.analyze_documents(docs, context)
    metrics = res.get("metrics", {})
    assert metrics.get("sp_engine") in ("cached_incr", "cached")
    # gmin should not be worse than g0 for improvement-oriented engines
    if "g0" in metrics and "gmin" in metrics:
        assert metrics["gmin"] <= metrics["g0"] + 1e-6
    # NormSpec must be echoed
    assert "norm_spec" in metrics

