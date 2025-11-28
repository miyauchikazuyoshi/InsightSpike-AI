"""Multihop coverage smoke for GeDIGCore."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from insightspike.algorithms.gedig_core import GeDIGCore


def _make_graph(n):
    g = nx.path_graph(n)
    for i in range(n):
        g.nodes[i]["feature"] = np.array([float(i), 0.0], dtype=float)
    return g


def test_multihop_basic_path():
    g_before = _make_graph(2)
    g_after = _make_graph(3)  # added node/edge

    core = GeDIGCore(enable_multihop=True, max_hops=1, use_multihop_sp_gain=False, sp_beta=0.0)
    res = core.calculate(g_prev=g_before, g_now=g_after, focal_nodes={0, 1})

    assert res.hop_results is not None and 0 in res.hop_results
    hop0 = res.hop_results[0]
    assert hop0.ged >= 0.0  # structural cost present or neutral
    assert np.isfinite(res.gedig_value)
    assert isinstance(res.spike, bool)
