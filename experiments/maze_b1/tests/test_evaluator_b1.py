"""Tests for β₁-based maze evaluator.

B1-B4: β₁ computation correctness
B5-B7: Δβ₁ properties
B8-B10: evaluate_multihop_b1 behavior
"""
import pytest
import networkx as nx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qhlib"))
from evaluator_b1 import compute_betti_1, delta_betti_1, evaluate_multihop_b1, EvalResult


# ── B1-B4: β₁ computation ──────────────────────────────────

class TestBetti1:
    def test_b1_empty_graph(self):
        """B1: Empty graph → β₁ = 0."""
        g = nx.Graph()
        assert compute_betti_1(g) == 0

    def test_b1_tree(self):
        """B2: Tree (no cycles) → β₁ = 0."""
        g = nx.path_graph(5)  # 0-1-2-3-4
        assert compute_betti_1(g) == 0

    def test_b1_single_cycle(self):
        """B3: Single cycle → β₁ = 1."""
        g = nx.cycle_graph(4)  # 0-1-2-3-0
        assert compute_betti_1(g) == 1

    def test_b1_grid(self):
        """B4: 3x3 grid → β₁ = 4 (4 independent cycles)."""
        g = nx.grid_2d_graph(3, 3)
        assert compute_betti_1(g) == 4

    def test_b1_two_components(self):
        """B4b: Two separate triangles → β₁ = 2."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle 1
        g.add_edges_from([(3, 4), (4, 5), (5, 3)])  # triangle 2
        assert compute_betti_1(g) == 2

    def test_b1_star(self):
        """B4c: Star graph (no cycles) → β₁ = 0."""
        g = nx.star_graph(5)  # center + 5 leaves
        assert compute_betti_1(g) == 0


# ── B5-B7: Δβ₁ properties ──────────────────────────────────

class TestDeltaBetti1:
    def test_db1_same_graph(self):
        """B5: Same graph → Δβ₁ = 0."""
        g = nx.cycle_graph(4)
        assert delta_betti_1(g, g) == 0.0

    def test_db1_add_cycle(self):
        """B6: Adding a cycle increases Δβ₁ > 0."""
        g_before = nx.path_graph(4)  # tree, β₁=0
        g_after = nx.cycle_graph(4)  # cycle, β₁=1
        d = delta_betti_1(g_before, g_after)
        assert d > 0

    def test_db1_remove_cycle(self):
        """B7: Removing a cycle decreases Δβ₁ < 0."""
        g_before = nx.cycle_graph(4)  # cycle, β₁=1
        g_after = nx.path_graph(4)    # tree, β₁=0
        d = delta_betti_1(g_before, g_after)
        assert d < 0


# ── B8-B10: evaluate_multihop_b1 ────────────────────────────

class TestEvaluateMultihop:
    def _make_simple_maze(self):
        """Create a simple 3x3 grid maze for testing."""
        g = nx.grid_2d_graph(3, 3)
        return g

    def test_b8_returns_eval_result(self):
        """B8: evaluate_multihop_b1 returns EvalResult."""
        g = self._make_simple_maze()
        result = evaluate_multihop_b1(
            lambda_weight=1.0,
            gamma=0.5,
            prev_graph=g,
            stage_graph=g,
            g_before_for_expansion=g,
            anchors_core={(1, 1)},
            anchors_top_before=set(g.nodes()),
            anchors_top_after=set(g.nodes()),
            ecand=[],
            base_ig=0.1,
            denom_cmax_base=10.0,
            max_hops=3,
        )
        assert isinstance(result, EvalResult)
        assert len(result.hop_series) >= 1
        assert result.hop_series[0]["hop"] == 0

    def test_b9_ag_gate_skips(self):
        """B9: AG gate (g0 < theta_ag) skips multi-hop."""
        g = self._make_simple_maze()
        result = evaluate_multihop_b1(
            lambda_weight=1.0,
            gamma=0.5,
            prev_graph=g,
            stage_graph=g,
            g_before_for_expansion=g,
            anchors_core={(1, 1)},
            anchors_top_before=set(g.nodes()),
            anchors_top_after=set(g.nodes()),
            ecand=[((0, 0), (2, 2), {})],
            base_ig=0.1,
            denom_cmax_base=10.0,
            max_hops=5,
            theta_ag=100.0,  # very high → g0 always below
        )
        assert result.best_hop == 0
        assert len(result.hop_series) == 1

    def test_b10_greedy_adds_edges(self):
        """B10: With candidates, greedy selection adds edges."""
        g_before = nx.path_graph(4)
        g_after = nx.path_graph(4)

        # Candidate edge creates a shortcut
        ecand = [(0, 3, {"weight": 1.0})]

        result = evaluate_multihop_b1(
            lambda_weight=1.0,
            gamma=0.5,
            prev_graph=g_before,
            stage_graph=g_after,
            g_before_for_expansion=g_before,
            anchors_core={0},
            anchors_top_before=set(g_before.nodes()),
            anchors_top_after=set(g_after.nodes()),
            ecand=ecand,
            base_ig=0.1,
            denom_cmax_base=10.0,
            max_hops=3,
        )
        assert len(result.hop_series) >= 2  # hop 0 + at least 1 hop
        assert len(result.chosen_edges_by_hop) >= 1

    def test_b10b_has_b1_in_hop_series(self):
        """B10b: hop_series contains 'b1' field (not 'sp')."""
        g = self._make_simple_maze()
        result = evaluate_multihop_b1(
            lambda_weight=1.0,
            gamma=0.5,
            prev_graph=g,
            stage_graph=g,
            g_before_for_expansion=g,
            anchors_core={(1, 1)},
            anchors_top_before=set(g.nodes()),
            anchors_top_after=set(g.nodes()),
            ecand=[],
            base_ig=0.1,
            denom_cmax_base=10.0,
            max_hops=1,
        )
        assert "b1" in result.hop_series[0]
        assert "sp" not in result.hop_series[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
