import math

import networkx as nx
import numpy as np
import pytest

from insightspike.algorithms.core.metrics import entropy_ig


def _make_line_graph(nodes: int) -> nx.Graph:
    g = nx.path_graph(nodes)
    return g


def test_entropy_ig_uses_fixed_denominator():
    g = _make_line_graph(4)
    before = np.ones((4, 3), dtype=np.float32)
    after = before * 0.5

    fixed_den = math.log(5.0)
    result = entropy_ig(g, before, after, fixed_den=fixed_den, k_star=5)

    assert result["normalization_den"] == pytest.approx(fixed_den)


def test_entropy_ig_small_k_returns_zero():
    g = _make_line_graph(3)
    before = np.ones((3, 2), dtype=np.float32)
    after = before.copy()

    result = entropy_ig(g, before, after, k_star=1)

    assert result["ig_value"] == 0.0
    assert result["variance_reduction"] == 0.0
