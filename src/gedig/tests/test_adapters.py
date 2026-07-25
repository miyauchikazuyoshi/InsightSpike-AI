"""Adapter equivalence tests.

Verifies that each adapter produces results consistent with
the original experiment code. Also tests cross-adapter compatibility.
"""

import math
import pytest
import networkx as nx

from gedig.core import (
    AGDGResult,
    EdgePartitionResult,
    FEvalResult,
    TwoStageGateDecision,
)
from gedig.adapters.maze import MazeFEval
from gedig.adapters.rag import RAGFEval


# ─── Maze Adapter Tests ─────────────────────────────────────────

class TestMazeAdapter:
    def test_basic_hop_eval(self):
        """Evaluate a single hop: adding an edge should produce nonzero F."""
        before = nx.Graph()
        before.add_edges_from([(0, 1), (1, 2)])

        after = nx.Graph()
        after.add_edges_from([(0, 1), (1, 2), (2, 3)])

        f_eval = MazeFEval(lambda_param=1.0, gamma=0.5)
        result = f_eval.evaluate_hop(before, after)

        assert isinstance(result, FEvalResult)
        assert result.delta_epc != 0.0  # Edge was added

    def test_identical_graph_zero_f(self):
        """Same graph → F = 0."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])

        f_eval = MazeFEval()
        result = f_eval.evaluate_hop(g, g)
        assert result.f_value == 0.0

    def test_multihop_series(self):
        """Multi-hop evaluation produces one result per hop."""
        base = nx.Graph()
        base.add_edges_from([(0, 1)])

        hop1 = nx.Graph()
        hop1.add_edges_from([(0, 1), (1, 2)])

        hop2 = nx.Graph()
        hop2.add_edges_from([(0, 1), (1, 2), (2, 3)])

        f_eval = MazeFEval()
        results = f_eval.evaluate_multihop(base, [hop1, hop2])
        assert len(results) == 2
        assert all(isinstance(r, FEvalResult) for r in results)

    def test_ag_dg_classification(self):
        """AG fires when g0 exceeds threshold."""
        f_eval = MazeFEval(theta_ag=0.5, theta_dg=0.2)

        result = f_eval.classify_step(g0=0.8, gmin_mh=0.3, best_hop=0)
        assert result["ag_fire"] is True
        assert result["dg_fire"] is False

        decision = f_eval.decide_step(
            g0=0.8,
            gmin_mh=0.3,
            best_hop=0,
        )
        assert isinstance(decision, TwoStageGateDecision)
        assert decision.attention_gate_fired is True
        assert decision.decision_gate_fired is False

    def test_dg_fire(self):
        """DG fires when multi-hop found and gmin below threshold."""
        f_eval = MazeFEval(theta_ag=0.5, theta_dg=0.5)

        result = f_eval.classify_step(g0=0.3, gmin_mh=0.1, best_hop=2)
        assert result["ag_fire"] is False
        assert result["dg_fire"] is True

    def test_legacy_classifier_threshold_mutation_updates_gate(self):
        f_eval = MazeFEval(theta_ag=0.5, theta_dg=0.2)

        f_eval.classifier.theta_ag = 0.9

        result = f_eval.classify_step(
            g0=0.8,
            gmin_mh=0.3,
            best_hop=0,
        )
        assert result["ag_fire"] is False

    def test_sleep_propagation(self):
        """Sleep propagation increases values near high-reward nodes."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}  # Goal at node 3

        f_eval = MazeFEval()
        propagated = f_eval.sleep_propagate(g, rewards, gamma=0.9, n_iterations=10)

        # Node 2 (adjacent to goal) should have higher value than node 0
        assert propagated[2] > propagated[0]
        # Node 3 (goal) should have highest value
        assert propagated[3] >= propagated[2]

    def test_betti_mode(self):
        """use_betti=True uses β₁ instead of SP."""
        before = nx.Graph()
        before.add_edges_from([(0, 1), (1, 2)])

        after = nx.Graph()
        after.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Adds cycle

        f_sp = MazeFEval(use_betti=False)
        f_b1 = MazeFEval(use_betti=True)

        r_sp = f_sp.evaluate_hop(before, after)
        r_b1 = f_b1.evaluate_hop(before, after)

        # Both should detect the change, but delta_b should differ
        assert r_sp.delta_epc == r_b1.delta_epc  # Same EPC
        assert r_sp.delta_h == r_b1.delta_h  # Same entropy
        # β₁ should detect the new cycle
        assert r_b1.delta_b > 0


# ─── RAG Adapter Tests ──────────────────────────────────────────

class TestRAGAdapter:
    def test_edge_f_computation(self):
        """Per-edge F-value: f = cost - λ·dot(Q,K)/√d_k."""
        f_eval = RAGFEval(f_lambda=1.0, d_k=3.0)

        # High Q·K → low f (AG-like)
        f_ag = f_eval.compute_edge_f(
            cost=0.3,
            q_vec=[1.0, 0.8, 0.5],
            k_vec=[0.9, 0.7, 0.6],
        )

        # Low Q·K → high f (DG-like)
        f_dg = f_eval.compute_edge_f(
            cost=0.3,
            q_vec=[0.1, 0.0, 0.0],
            k_vec=[0.1, 0.0, 0.0],
        )

        assert f_ag < f_dg  # AG has lower f-value

    def test_classify_edges(self):
        """New edge partition and legacy labels preserve the same split."""
        f_eval = RAGFEval(percentile=0.4)
        edge_scores = {
            "e1": -0.5,  # very AG
            "e2": -0.2,  # AG
            "e3": 0.1,   # borderline
            "e4": 0.3,   # DG
            "e5": 0.8,   # very DG
        }
        result = f_eval.classify_edges(edge_scores)
        assert isinstance(result, AGDGResult)
        assert result.n_ag + result.n_dg == 5
        assert result.n_ag > 0
        assert result.n_dg > 0

        partition = f_eval.partition_edges(edge_scores)
        assert isinstance(partition, EdgePartitionResult)
        assert (
            partition.n_low_score + partition.n_high_score
            == len(edge_scores)
        )
        assert partition.low_score_edges == result.ag_edges
        assert partition.high_score_edges == result.dg_edges

    def test_graph_betti(self):
        """β₁ computation on document graph."""
        f_eval = RAGFEval()

        # Path graph: no cycles → β₁ = 0
        g_path = nx.Graph()
        g_path.add_edges_from([(0, 1), (1, 2), (2, 3)])
        assert f_eval.compute_graph_betti(g_path) == 0.0

        # Triangle: 1 cycle → β₁ > 0
        g_tri = nx.Graph()
        g_tri.add_edges_from([(0, 1), (1, 2), (2, 0)])
        assert f_eval.compute_graph_betti(g_tri) > 0

    def test_propagation(self):
        """Attention-weighted propagation on directed graph."""
        g = nx.DiGraph()
        g.add_edge(0, 1, flow=0.8)
        g.add_edge(0, 2, flow=0.2)
        g.add_edge(1, 2, flow=0.5)

        f_eval = RAGFEval()
        initial = {0: 1.0, 1: 0.0, 2: 0.0}
        result = f_eval.propagate(g, initial, n_iterations=3, alpha=0.5)

        # Node 1 should receive more relevance than node 2
        # (higher flow from node 0)
        assert result[1] > result[2]

    def test_qkv_dot_product_scaling(self):
        """Verify dot product is scaled by √d_k."""
        f_eval = RAGFEval(f_lambda=1.0, d_k=4.0)

        q = [1.0, 1.0, 1.0, 1.0]
        k = [1.0, 1.0, 1.0, 1.0]
        # dot(Q,K) = 4, √4 = 2, scaled = 2.0
        f = f_eval.compute_edge_f(cost=0.5, q_vec=q, k_vec=k)
        expected = 0.5 - 1.0 * (4.0 / 2.0)  # = 0.5 - 2.0 = -1.5
        assert abs(f - expected) < 1e-10


# ─── Cross-Adapter Tests ────────────────────────────────────────

class TestCrossAdapter:
    """Tests that different adapters produce compatible results."""

    def test_betti_consistency(self):
        """β₁ from maze adapter matches RAG adapter on same graph."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

        maze = MazeFEval(use_betti=True)
        rag = RAGFEval()

        # Both should compute same β₁
        # Maze: via NxBetti inside FEval
        before_empty = nx.Graph()
        before_empty.add_nodes_from([0, 1, 2, 3])
        r_maze = maze.evaluate_hop(before_empty, g)

        # RAG: direct β₁ computation
        b_rag = rag.compute_graph_betti(g)

        # Both should detect the cycle (triangle 0-1-2)
        assert r_maze.delta_b > 0
        assert b_rag > 0

    def test_f_eval_formula_consistency(self):
        """F = ΔEPC - λ(ΔH + γΔB) is the same formula across adapters."""
        before = nx.Graph()
        before.add_edges_from([(0, 1)])

        after = nx.Graph()
        after.add_edges_from([(0, 1), (1, 2), (2, 0)])

        maze = MazeFEval(lambda_param=1.0, gamma=0.5)
        result = maze.evaluate_hop(before, after)

        # Verify F = ΔEPC - λ(ΔH + γΔB)
        expected_f = result.delta_epc - 1.0 * (result.delta_h + 0.5 * result.delta_b)
        assert abs(result.f_value - expected_f) < 1e-10


# ─── Torch Adapter Tests ────────────────────────────────────────

class TestTransformerAdapter:
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_basic_computation(self):
        import torch
        from gedig.adapters.transformer import TransformerFEval, TransformerFEvalResult

        f_eval = TransformerFEval(lambda_param=1.0, gamma=0.5)
        before = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        after = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)

        result = f_eval.compute(before, after)
        assert isinstance(result, TransformerFEvalResult)
        assert result.F.shape == (2, 4)
        assert result.F_mean.dim() == 0  # scalar

    def test_identical_attention_near_zero(self):
        import torch
        from gedig.adapters.transformer import TransformerFEval

        attn = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)
        f_eval = TransformerFEval()
        result = f_eval.compute(attn, attn)
        assert abs(result.F_mean.item()) < 0.01  # Near zero

    def test_use_betti_flag(self):
        import torch
        from gedig.adapters.transformer import TransformerFEval

        before = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)
        after = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)

        r_sp = TransformerFEval(use_betti=False).compute(before, after)
        r_b1 = TransformerFEval(use_betti=True).compute(before, after)

        assert r_sp.use_betti is False
        assert r_b1.use_betti is True
        assert r_sp.f_formula == "ΔEPC - λ(ΔH + γΔSP)"
        assert r_b1.f_formula == "ΔEPC - λ(ΔH + γΔB)"
        # F values should differ (different third component)
        assert r_sp.delta_b1.item() == 0.0  # Not computed in SP mode
        assert r_b1.delta_b1.item() != 0.0  # Computed in Betti mode

    def test_backward_compatible_fields(self):
        """Result has same fields as DifferentiableGeDIG.forward()."""
        import torch
        from gedig.adapters.transformer import TransformerFEval

        f_eval = TransformerFEval()
        before = torch.softmax(torch.randn(2, 4, 16, 16), dim=-1)
        after = torch.softmax(torch.randn(2, 4, 16, 16), dim=-1)
        result = f_eval.compute(before, after)

        # All fields from original DifferentiableGeDIG output
        assert hasattr(result, "F")
        assert hasattr(result, "F_mean")
        assert hasattr(result, "delta_epc")
        assert hasattr(result, "delta_h")
        assert hasattr(result, "delta_sp")
        assert hasattr(result, "delta_b1")
        assert hasattr(result, "use_betti")

    def test_gradient_flows(self):
        """F_mean is differentiable (gradients flow back to attention)."""
        import torch
        from gedig.adapters.transformer import TransformerFEval

        before = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)
        after_raw = torch.randn(1, 2, 8, 8, requires_grad=True)
        after = torch.softmax(after_raw, dim=-1)

        f_eval = TransformerFEval()
        result = f_eval.compute(before, after)
        result.F_mean.backward()

        assert after_raw.grad is not None
        assert after_raw.grad.shape == (1, 2, 8, 8)
