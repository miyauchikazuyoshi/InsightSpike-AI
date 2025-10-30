import math

import networkx as nx
import numpy as np
import pytest

from insightspike.algorithms.gedig_core import GeDIGCore


def test_gedig_core_applies_local_cmax_and_fixed_den():
    g_prev = nx.path_graph(2)
    g_now = nx.path_graph(3)

    features_prev = np.eye(g_prev.number_of_nodes(), dtype=np.float32)
    features_now = np.eye(g_now.number_of_nodes(), dtype=np.float32)

    core = GeDIGCore(use_local_normalization=True)
    result = core.calculate(
        g_prev=g_prev,
        g_now=g_now,
        features_prev=features_prev,
        features_now=features_now,
        k_star=4,
    )

    assert result.ged_norm_den == pytest.approx(3.0)
    assert result.ig_norm_den == pytest.approx(math.log(5.0))
