"""Tiny coverage tests for pyg_compatible_metrics helpers (no torch required)."""

from __future__ import annotations

import numpy as np

from insightspike.metrics import pyg_compatible_metrics as pm


class DummyPyg:
    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = x.shape[0]


def test_pyg_to_networkx_and_delta_metrics():
    g_old = DummyPyg(
        x=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
    )
    g_new = DummyPyg(
        x=np.array([[1.0, 1.0]], dtype=np.float32),
        edge_index=np.array([[0], [0]], dtype=np.int64),
    )

    nx_old = pm.pyg_to_networkx(g_old)
    nx_new = pm.pyg_to_networkx(g_new)
    assert nx_old.number_of_nodes() == 2
    assert nx_new.number_of_nodes() == 1

    dged = pm.delta_ged_pyg(g_old, g_new)
    dig = pm.delta_ig_pyg(g_old, g_new)
    assert dged < 0.0  # graph simplified
    assert np.isfinite(dig)
