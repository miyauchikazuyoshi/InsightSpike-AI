"""Small coverage-oriented tests for core metrics helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from insightspike.algorithms.core import metrics as core_metrics


def test_normalized_ged_with_override_and_spectral():
    g1 = nx.path_graph(2)  # 2 nodes, 1 edge
    g2 = nx.path_graph(1)  # 1 node, 0 edge
    res = core_metrics.normalized_ged(g1, g2, norm_override=2.0, enable_spectral=True, spectral_weight=0.5)
    assert res["normalized_ged"] == pytest.approx(1.0, rel=1e-6)
    assert res["structural_cost"] >= 0.0
    assert "structural_improvement" in res


def test_entropy_ig_with_extra_vectors_and_fixed_den():
    g = nx.path_graph(2)
    feats_before = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    feats_after = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    res = core_metrics.entropy_ig(
        g,
        feats_before,
        feats_after,
        smoothing=1e-6,
        fixed_den=np.log(3.0),
        extra_vectors=[[1.0, 0.0]],
        k_star=2,
    )
    assert "ig_value" in res
    assert res["candidate_count"] >= 2.0
    assert np.isfinite(res["ig_value"])
