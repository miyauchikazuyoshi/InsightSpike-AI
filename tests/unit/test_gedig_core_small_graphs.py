"""Lightweight property tests for GeDIGCore on tiny graphs.

These tests aim to raise coverage of the non-multihop path with deterministic
toy graphs (no torch/pyg dependency).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from insightspike.algorithms.gedig_core import GeDIGCore


def _path_graph(features):
    """Build a simple path graph with per-node feature vectors."""
    G = nx.Graph()
    for i, feat in enumerate(features):
        G.add_node(i, feature=np.asarray(feat, dtype=float))
    for i in range(len(features) - 1):
        G.add_edge(i, i + 1)
    return G


def test_gedig_core_identical_graph_zero_delta():
    """Identical before/after graphs should yield near-zero deltas."""
    feats = [[0.0, 0.0], [0.0, 0.0]]
    g_before = _path_graph(feats)
    g_after = _path_graph(feats)

    core = GeDIGCore(enable_multihop=False, use_multihop_sp_gain=False)
    res = core.calculate(g_prev=g_before, g_now=g_after)

    assert res.delta_ged_norm == pytest.approx(0.0, abs=1e-9)
    assert res.delta_h_norm == pytest.approx(0.0, abs=1e-9)
    assert res.gedig_value == pytest.approx(0.0, abs=1e-9)
    assert res.spike is False


def test_gedig_core_entropy_reduction_increases_score():
    """Entropy change without structural cost should directly shape geDIG."""
    g_before = _path_graph([[1.0, 0.0], [0.0, 1.0]])  # diverse features
    g_after = _path_graph([[0.0, 0.0], [0.0, 0.0]])    # collapsed features -> lower entropy

    core = GeDIGCore(enable_multihop=False, use_multihop_sp_gain=False, lambda_weight=1.0)
    res = core.calculate(g_prev=g_before, g_now=g_after, k_star=2, l1_candidates=2)

    assert res.delta_ged_norm == pytest.approx(0.0, abs=1e-9)
    assert abs(res.delta_h_norm) > 1e-6  # entropy changed
    # With zero structural cost, geDIG should mirror the IG term (sign included)
    assert res.gedig_value == pytest.approx(-res.delta_h_norm, rel=1e-6, abs=1e-6)


def test_gedig_core_additional_edge_increases_struct_cost():
    """Adding a node/edge should raise structural cost and geDIG score."""
    g_before = _path_graph([[0.0, 0.0], [0.0, 0.0]])  # 2 nodes, 1 edge
    g_after = _path_graph([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # 3 nodes, 2 edges

    core = GeDIGCore(enable_multihop=False, use_multihop_sp_gain=False, lambda_weight=1.0)
    res = core.calculate(g_prev=g_before, g_now=g_after)

    assert res.delta_ged_norm > 0.0
    assert res.delta_h_norm == pytest.approx(0.0, abs=1e-9)
    assert res.gedig_value > 0.0
    # Structural cost dominates; geDIG should be bounded by the GED term
    assert res.gedig_value < res.delta_ged_norm + 1e-6
