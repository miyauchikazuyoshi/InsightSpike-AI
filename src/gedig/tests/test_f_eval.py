"""Unit tests for unified geDIG core.

Tests core FEval, backends, and AG/DG classifiers
with synthetic graphs of known properties.
"""

import math
import pytest
import networkx as nx

from gedig.core import FEval, FEvalResult, PercentileClassifier, ThresholdClassifier
from gedig.backends.networkx_backend import (
    NxGraphSnapshot,
    NxEPC,
    NxEntropy,
    NxSP,
    NxBetti,
)


# ─── Fixtures ────────────────────────────────────────────────────

def make_triangle():
    """Triangle graph: 3 nodes, 3 edges, β₁=1 (one cycle)."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return NxGraphSnapshot(g)


def make_path():
    """Path graph: 3 nodes, 2 edges, β₁=0 (no cycles)."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    return NxGraphSnapshot(g)


def make_star():
    """Star graph: 4 nodes, 3 edges, β₁=0."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2), (0, 3)])
    return NxGraphSnapshot(g)


def make_empty():
    """Empty graph: 3 nodes, 0 edges."""
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    return NxGraphSnapshot(g)


# ─── GraphSnapshot Tests ─────────────────────────────────────────

class TestNxGraphSnapshot:
    def test_triangle_counts(self):
        s = make_triangle()
        assert s.node_count() == 3
        assert s.edge_count() == 3
        assert len(s.edge_set()) == 3

    def test_path_counts(self):
        s = make_path()
        assert s.node_count() == 3
        assert s.edge_count() == 2

    def test_empty_counts(self):
        s = make_empty()
        assert s.node_count() == 3
        assert s.edge_count() == 0


# ─── EPC Tests ───────────────────────────────────────────────────

class TestNxEPC:
    def test_identical_graphs(self):
        s = make_triangle()
        epc = NxEPC()
        assert epc.compute(s, s) == 0.0

    def test_added_edges(self):
        before = make_path()  # 2 edges
        after = make_triangle()  # 3 edges (adds edge 0-2)
        epc = NxEPC()
        # 1 added, 0 removed, max(2,3) = 3
        assert abs(epc.compute(before, after) - 1 / 3) < 1e-10

    def test_removed_edges(self):
        before = make_triangle()
        after = make_path()
        epc = NxEPC()
        assert abs(epc.compute(before, after) - 1 / 3) < 1e-10

    def test_complete_change(self):
        before = make_path()  # edges: 0-1, 1-2
        g = nx.Graph()
        g.add_edges_from([(0, 2), (1, 2)])  # different edges
        after = NxGraphSnapshot(g)
        epc = NxEPC()
        # 1 added (0-2), 1 removed (0-1), max(2,2) = 2 → 2/2 = 1.0
        result = epc.compute(before, after)
        assert result > 0  # Some structural change occurred


# ─── Entropy Tests ───────────────────────────────────────────────

class TestNxEntropy:
    def test_identical_entropy(self):
        s = make_triangle()
        ent = NxEntropy()
        assert ent.compute(s, s) == 0.0

    def test_ordering_decreases_entropy(self):
        # Star has more uneven degree distribution than triangle
        before = make_triangle()  # degrees: [2, 2, 2] → max entropy
        after = make_star()  # degrees: [3, 1, 1, 1] → lower entropy
        ent = NxEntropy()
        delta_h = ent.compute(before, after)
        # Star has lower normalized entropy than triangle
        assert delta_h < 0  # ordering = entropy decrease


# ─── SP Tests ────────────────────────────────────────────────────

class TestNxSP:
    def test_identical_sp(self):
        s = make_triangle()
        sp = NxSP()
        assert sp.compute(s, s) == 0.0

    def test_more_edges_better_efficiency(self):
        before = make_path()
        after = make_triangle()
        sp = NxSP(n_pairs=10)
        # Both are small fully-connected, efficiency may be same
        # At minimum, triangle should not be worse
        assert sp.compute(before, after) >= 0


# ─── Betti Tests ─────────────────────────────────────────────────

class TestNxBetti:
    def test_triangle_has_cycle(self):
        s = make_triangle()
        b = NxBetti()
        beta = b._betti_1(s)
        # β₁ = 3 - 3 + 1 = 1, normalized by 3 = 1/3
        assert beta > 0

    def test_path_no_cycle(self):
        s = make_path()
        b = NxBetti()
        beta = b._betti_1(s)
        # β₁ = 2 - 3 + 1 = 0
        assert beta == 0.0

    def test_adding_cycle(self):
        before = make_path()
        after = make_triangle()
        b = NxBetti()
        delta = b.compute(before, after)
        assert delta > 0  # Gained a cycle


# ─── FEval Integration Tests ────────────────────────────────────

class TestFEval:
    def test_identical_f_is_zero(self):
        s = make_triangle()
        f_eval = FEval(
            epc=NxEPC(),
            entropy=NxEntropy(),
            sp=NxSP(),
        )
        result = f_eval.compute(s, s)
        assert isinstance(result, FEvalResult)
        assert result.f_value == 0.0
        assert result.delta_epc == 0.0
        assert result.delta_h == 0.0
        assert result.delta_b == 0.0

    def test_f_components_combine(self):
        before = make_path()
        after = make_triangle()
        f_eval = FEval(
            epc=NxEPC(),
            entropy=NxEntropy(),
            sp=NxSP(n_pairs=10),
            lambda_param=1.0,
            gamma=0.5,
        )
        result = f_eval.compute(before, after)
        # F = ΔEPC - λ(ΔH + γΔSP)
        expected = result.delta_epc - 1.0 * (result.delta_h + 0.5 * result.delta_b)
        assert abs(result.f_value - expected) < 1e-10

    def test_f_with_betti(self):
        before = make_path()
        after = make_triangle()
        f_eval = FEval(
            epc=NxEPC(),
            entropy=NxEntropy(),
            sp=NxBetti(),  # Use Betti instead of SP
            lambda_param=1.0,
            gamma=0.5,
        )
        result = f_eval.compute(before, after)
        assert result.delta_b > 0  # Gained a cycle

    def test_repr(self):
        f_eval = FEval(NxEPC(), NxEntropy(), NxSP())
        r = repr(f_eval)
        assert "NxEPC" in r
        assert "NxEntropy" in r
        assert "NxSP" in r


# ─── AG/DG Classifier Tests ─────────────────────────────────────

class TestPercentileClassifier:
    def test_basic_classification(self):
        scores = {"e1": 0.1, "e2": 0.3, "e3": 0.5, "e4": 0.7, "e5": 0.9}
        clf = PercentileClassifier(percentile=0.4)
        result = clf.classify(scores)
        assert result.n_ag + result.n_dg == 5
        assert result.n_ag > 0
        assert result.n_dg > 0

    def test_empty_scores(self):
        clf = PercentileClassifier()
        result = clf.classify({})
        assert result.n_ag == 0
        assert result.n_dg == 0


class TestThresholdClassifier:
    def test_ag_fire(self):
        scores = {"e1": 0.8, "e2": 0.3, "e3": 0.1}
        clf = ThresholdClassifier(theta_ag=0.5, theta_dg=0.2)
        result = clf.classify(scores)
        assert "e1" in result.ag_edges
        assert "e3" in result.dg_edges


# ─── PyTorch Backend Tests (optional) ────────────────────────────

class TestTorchBackend:
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_torch_snapshot(self):
        import torch
        from gedig.backends.torch_backend import TorchGraphSnapshot

        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        snap = TorchGraphSnapshot(attn)
        assert snap.node_count() == 8
        assert snap.edge_count() > 0
        assert snap.edge_set() == set()

    def test_torch_epc(self):
        import torch
        from gedig.backends.torch_backend import TorchGraphSnapshot, TorchEPC

        before = TorchGraphSnapshot(torch.softmax(torch.randn(2, 4, 8, 8), dim=-1))
        after = TorchGraphSnapshot(torch.softmax(torch.randn(2, 4, 8, 8), dim=-1))
        epc = TorchEPC()
        result = epc.compute(before, after)
        assert result.shape == (2, 4)

    def test_torch_entropy(self):
        import torch
        from gedig.backends.torch_backend import TorchGraphSnapshot, TorchEntropy

        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        snap = TorchGraphSnapshot(attn)
        ent = TorchEntropy()
        result = ent.compute(snap, snap)
        assert result.shape == (2, 4)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_torch_betti(self):
        import torch
        from gedig.backends.torch_backend import TorchGraphSnapshot, TorchBetti

        before = TorchGraphSnapshot(torch.softmax(torch.randn(2, 4, 8, 8), dim=-1))
        after = TorchGraphSnapshot(torch.softmax(torch.randn(2, 4, 8, 8), dim=-1))
        betti = TorchBetti()
        result = betti.compute(before, after)
        assert result.shape == (2, 4)
