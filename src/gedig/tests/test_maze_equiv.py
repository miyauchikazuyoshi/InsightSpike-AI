"""Maze adapter equivalence tests (M1-M3).

Verifies that the maze adapter produces compatible results
with known-answer inputs. Full E2E tests (M4-M5) require
the maze experiment runner.
"""

import networkx as nx
import pytest

from gedig.adapters.maze import MazeFEval
from gedig.core import FEvalResult


def _build_simple_maze():
    """Simple 3x3 maze graph for testing."""
    g = nx.Graph()
    # Grid positions as nodes
    for r in range(3):
        for c in range(3):
            g.add_node((r, c), pos=(r, c))
    # Horizontal passages
    for r in range(3):
        for c in range(2):
            g.add_edge((r, c), (r, c + 1))
    # Vertical passages
    for r in range(2):
        for c in range(3):
            g.add_edge((r, c), (r + 1, c))
    return g


# ─── M1: g-value computation ────────────────────────────────────

class TestM1_GValueComputation:
    """M1: g-value from adapter matches F formula."""

    def test_f_formula_matches(self):
        """g = ΔEPC - λ(ΔH + γΔB) matches manual computation."""
        before = nx.Graph()
        before.add_edges_from([(0, 1), (1, 2)])

        after = nx.Graph()
        after.add_edges_from([(0, 1), (1, 2), (2, 3)])

        f_eval = MazeFEval(lambda_param=1.0, gamma=0.5)
        result = f_eval.evaluate_hop(before, after)

        # Verify F formula consistency
        expected = result.delta_epc - 1.0 * (result.delta_h + 0.5 * result.delta_b)
        assert abs(result.f_value - expected) < 1e-10

    def test_no_change_zero_g(self):
        g = _build_simple_maze()
        f_eval = MazeFEval()
        result = f_eval.evaluate_hop(g, g)
        assert result.f_value == 0.0

    def test_adding_shortcut_changes_g(self):
        """Adding a diagonal shortcut should produce nonzero g."""
        before = _build_simple_maze()
        after = _build_simple_maze()
        after.add_edge((0, 0), (2, 2))  # Diagonal shortcut

        f_eval = MazeFEval()
        result = f_eval.evaluate_hop(before, after)
        assert result.delta_epc != 0.0


# ─── M2: AG/DG classification ───────────────────────────────────

class TestM2_AGDGClassification:
    """M2: AG/DG fire matches threshold logic."""

    def test_ag_fires_above_threshold(self):
        f_eval = MazeFEval(theta_ag=0.3)
        result = f_eval.classify_step(g0=0.5, gmin_mh=0.2, best_hop=0)
        assert result["ag_fire"] is True

    def test_ag_does_not_fire_below(self):
        f_eval = MazeFEval(theta_ag=0.8)
        result = f_eval.classify_step(g0=0.5, gmin_mh=0.2, best_hop=0)
        assert result["ag_fire"] is False

    def test_dg_fires_multihop(self):
        f_eval = MazeFEval(theta_dg=0.5)
        result = f_eval.classify_step(g0=0.1, gmin_mh=0.2, best_hop=3)
        assert result["dg_fire"] is True

    def test_dg_requires_multihop(self):
        """DG requires best_hop >= 1."""
        f_eval = MazeFEval(theta_dg=0.5)
        result = f_eval.classify_step(g0=0.1, gmin_mh=0.2, best_hop=0)
        assert result["dg_fire"] is False


# ─── M3: Sleep propagation ──────────────────────────────────────

class TestM3_SleepPropagation:
    """M3: Q-learning propagation matches manual computation."""

    def test_single_step_propagation(self):
        """One iteration: propagated = reward + γ·max(neighbor)."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2)])
        rewards = {0: 0.0, 1: 0.0, 2: 1.0}

        f_eval = MazeFEval()
        result = f_eval.sleep_propagate(g, rewards, gamma=0.9, n_iterations=1)

        # Node 2: reward=1.0 + 0.9 * max(propagated_neighbors)
        # After 1 iter: node2 = 1.0 + 0.9 * 0.0 = 1.0 (neighbors have 0)
        # Actually: node2 neighbors are [1], node1 neighbors are [0,2]
        # 1 iteration: propagated[2] = 1.0 + 0.9 * max(rewards[1]) = 1.0
        # propagated[1] = 0.0 + 0.9 * max(rewards[0], rewards[2]) = 0.9
        # propagated[0] = 0.0 + 0.9 * max(rewards[1]) = 0.0
        assert result[2] == 1.0
        assert abs(result[1] - 0.9) < 1e-10
        assert result[0] == 0.0

    def test_multi_step_convergence(self):
        """After many iterations, values should propagate through chain."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}

        f_eval = MazeFEval()
        result = f_eval.sleep_propagate(g, rewards, gamma=0.9, n_iterations=50)

        # Value should decay with distance from goal
        assert result[4] > result[3] > result[2] > result[1] > result[0]

    def test_betti_mode_works(self):
        """use_betti=True should work for maze."""
        before = _build_simple_maze()
        after = _build_simple_maze()
        after.add_edge((0, 0), (2, 2))  # Creates cycle

        f_eval = MazeFEval(use_betti=True)
        result = f_eval.evaluate_hop(before, after)
        # Adding a shortcut creates cycles → β₁ increases
        assert result.delta_b > 0


# ─── Cross-check: Maze formula = unified formula ────────────────

class TestMazeFormulaUnification:
    """Verify maze's g = ΔGED - λ(ΔH + γΔSP) maps to unified F."""

    def test_parameter_mapping(self):
        """Maze's core.lambda_weight maps to FEval.lambda_param."""
        f_eval = MazeFEval(lambda_param=2.0, gamma=0.3)
        # The underlying FEval should have these parameters
        assert f_eval.f_eval.lambda_param == 2.0
        assert f_eval.f_eval.gamma == 0.3

    def test_multihop_produces_series(self):
        """evaluate_multihop returns one FEvalResult per hop."""
        base = nx.Graph()
        base.add_edges_from([(0, 1)])

        hops = []
        for i in range(3):
            g = base.copy()
            for j in range(i + 1):
                g.add_edge(j + 1, j + 2)
            hops.append(g)

        f_eval = MazeFEval()
        results = f_eval.evaluate_multihop(base, hops)
        assert len(results) == 3
        assert all(isinstance(r, FEvalResult) for r in results)
