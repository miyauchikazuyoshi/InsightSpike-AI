"""Linkset-driven multihop path coverage for GeDIGCore."""

from __future__ import annotations

import math
import networkx as nx
import numpy as np
import pytest

from insightspike.algorithms.gedig_core import GeDIGCore


def _make_graph(n):
    g = nx.path_graph(n)
    for i in range(n):
        g.nodes[i]["feature"] = np.array([float(i), 0.0], dtype=float)
    return g


def test_multihop_linkset_ig_source():
    g_before = _make_graph(2)
    g_after = _make_graph(2)

    linkset_info = {
        "s_link": [
            {"index": 1, "similarity": 0.6, "origin": "mem"},
            {"index": 2, "similarity": 0.4, "origin": "mem"},
        ],
        "candidate_pool": [
            {"index": 1, "similarity": 0.6, "origin": "mem"},
            {"index": 2, "similarity": 0.4, "origin": "mem"},
        ],
        "decision": {"index": 1, "similarity": 0.6},
        "base_mode": "link",
    }

    core = GeDIGCore(
        enable_multihop=True,
        max_hops=1,
        use_multihop_sp_gain=False,
        lambda_weight=0.0,  # focus on IG term
        ig_source_mode="linkset",
    )
    res = core.calculate(g_prev=g_before, g_now=g_after, focal_nodes={0, 1}, linkset_info=linkset_info)

    assert res.linkset_metrics is not None
    hop0 = res.hop_results[0]
    # IG should come from linkset_metrics
    assert hop0.h_component == pytest.approx(res.linkset_metrics.delta_h_norm, rel=1e-6, abs=1e-6)
    assert math.isfinite(res.gedig_value)
