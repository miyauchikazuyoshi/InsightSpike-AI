"""Unit tests for the v2 knowledge graph builder.

Key tests:
- Disconnected components (beta_0 > 1) when cross-title facts lack entity overlap
- Bridge fact merging components (delta_beta_0 = -1)
- Q connected to only top-k_q facts (not all)
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.hotpotqa_v2.src.graph_builder import (
    GraphBuildConfig,
    KnowledgeGraphBuilder,
    compute_edge_dg_scores,
)
from experiments.hotpotqa_v2.src.retriever import RetrievedFact


def _make_facts() -> list[RetrievedFact]:
    """Create a test corpus with 2 distinct title groups and 1 bridge fact."""
    return [
        # Title A — about Albert Einstein
        RetrievedFact("Albert Einstein", 0, "Albert Einstein was born in Ulm.", 5.0),
        RetrievedFact("Albert Einstein", 1, "He developed the theory of relativity.", 4.0),
        # Title B — about Max Planck
        RetrievedFact("Max Planck", 0, "Max Planck originated quantum theory.", 3.0),
        RetrievedFact("Max Planck", 1, "Planck received the Nobel Prize in 1918.", 2.0),
        # Bridge fact — mentions both
        RetrievedFact("Physics History", 0, "Albert Einstein and Max Planck were colleagues in Berlin.", 1.0),
    ]


class TestGraphStructure:
    def test_q_only_linked_to_top_k(self):
        """Q should be connected to only q_link_top_k facts, not all."""
        cfg = GraphBuildConfig(q_link_top_k=3)
        builder = KnowledgeGraphBuilder(cfg)
        facts = _make_facts()
        g = builder.build_graph("Who influenced Einstein?", facts)

        q_neighbors = set(g.neighbors("Q"))
        assert len(q_neighbors) == 3
        # Top-3 by score: F0 (5.0), F1 (4.0), F2 (3.0)
        assert q_neighbors == {"F0", "F1", "F2"}

    def test_intra_title_edges(self):
        """Facts from same title with adjacent sent_id should be connected."""
        cfg = GraphBuildConfig(q_link_top_k=2)
        builder = KnowledgeGraphBuilder(cfg)
        facts = _make_facts()
        g = builder.build_graph("test question", facts)

        # Albert Einstein: F0 (sent 0) and F1 (sent 1) should be connected
        assert g.has_edge("F0", "F1")
        # Max Planck: F2 (sent 0) and F3 (sent 1) should be connected
        assert g.has_edge("F2", "F3")

    def test_node_count(self):
        """Graph should have Q + len(facts) nodes."""
        builder = KnowledgeGraphBuilder()
        facts = _make_facts()
        g = builder.build_graph("test", facts)
        assert g.number_of_nodes() == 6  # Q + 5 facts


class TestBeta0Sensitivity:
    def test_disconnected_without_bridge(self):
        """Without the bridge fact, two title groups should be disconnected."""
        cfg = GraphBuildConfig(
            q_link_top_k=2,
            entity_overlap_threshold=0.9,  # Very high → no cross-title edges
        )
        builder = KnowledgeGraphBuilder(cfg)

        # Only facts from two different titles, no bridge
        facts = [
            RetrievedFact("Albert Einstein", 0, "Albert Einstein was born in Ulm.", 5.0),
            RetrievedFact("Albert Einstein", 1, "He developed relativity.", 4.0),
            RetrievedFact("Max Planck", 0, "Max Planck originated quantum theory.", 3.0),
            RetrievedFact("Max Planck", 1, "Planck received the Nobel Prize.", 2.0),
        ]
        g = builder.build_graph("Who are the physicists?", facts)

        # Q is connected to top-2 (F0, F1 = Einstein)
        # F2, F3 (Planck) are NOT connected to Q and have no cross-title edges
        beta_0 = nx.number_connected_components(g)
        assert beta_0 >= 2, f"Expected beta_0 >= 2, got {beta_0}"

    def test_bridge_reduces_beta0(self):
        """Adding a bridge fact should merge components (delta_beta_0 <= -1)."""
        cfg = GraphBuildConfig(
            q_link_top_k=2,
            entity_overlap_threshold=0.3,
        )
        builder = KnowledgeGraphBuilder(cfg)

        # Without bridge
        facts_no_bridge = [
            RetrievedFact("Albert Einstein", 0, "Albert Einstein was born in Ulm.", 5.0),
            RetrievedFact("Albert Einstein", 1, "He developed relativity.", 4.0),
            RetrievedFact("Max Planck", 0, "Max Planck originated quantum theory.", 3.0),
            RetrievedFact("Max Planck", 1, "Planck received the Nobel Prize.", 2.0),
        ]

        # With bridge
        bridge_fact = RetrievedFact(
            "Physics History", 0,
            "Albert Einstein and Max Planck were colleagues in Berlin.",
            1.0,
        )
        facts_with_bridge = facts_no_bridge + [bridge_fact]

        g_before = builder.build_graph("Who are the physicists?", facts_no_bridge)
        g_after = builder.build_graph("Who are the physicists?", facts_with_bridge)

        b0_before = nx.number_connected_components(g_before)
        b0_after = nx.number_connected_components(g_after)
        delta_b0 = b0_after - b0_before

        assert delta_b0 <= -1, (
            f"Expected delta_beta_0 <= -1, got {delta_b0} "
            f"(before={b0_before}, after={b0_after})"
        )


class TestEmptyGraph:
    def test_empty_graph_has_q_only(self):
        builder = KnowledgeGraphBuilder()
        g = builder.build_empty_graph("test question")
        assert g.number_of_nodes() == 1
        assert "Q" in g.nodes
        assert g.number_of_edges() == 0

    def test_query_vector_dimensions(self):
        cfg = GraphBuildConfig(tfidf_dim=64)
        builder = KnowledgeGraphBuilder(cfg)
        vec = builder.get_query_vector("test question")
        assert len(vec) == 5 + 64  # base (5) + tfidf (64)


class TestEntityExtraction:
    def test_extracts_capitalized(self):
        entities = KnowledgeGraphBuilder._extract_entities(
            "Albert Einstein worked at Princeton University."
        )
        assert "albert einstein" in entities
        assert "princeton university" in entities

    def test_empty_text(self):
        entities = KnowledgeGraphBuilder._extract_entities("")
        assert len(entities) == 0


class TestTwoEdgeArchitecture:
    """Tests for v5 Two-Edge Architecture (context + similarity attention edges)."""

    def test_context_edges_wider_than_adjacent(self):
        """v5: Context edges should connect non-adjacent same-title facts."""
        cfg = GraphBuildConfig(two_edge_mode=True, ctx_max_sent_distance=6)
        builder = KnowledgeGraphBuilder(cfg)
        facts = [
            RetrievedFact("Physics", 0, "Sentence zero about physics.", 5.0),
            RetrievedFact("Physics", 1, "Sentence one about physics.", 4.0),
            RetrievedFact("Physics", 3, "Sentence three about physics.", 3.0),
        ]
        g = builder.build_graph("test", facts)
        # F0-F1 (dist=1), F1-F2 (dist=2), F0-F2 (dist=3) should all have edges
        assert g.has_edge("F0", "F1"), "Adjacent sentences should be connected"
        assert g.has_edge("F1", "F2"), "dist=2 should be connected in v5"
        assert g.has_edge("F0", "F2"), "dist=3 should be connected in v5"

    def test_context_edge_weights_decay(self):
        """v5: Context edge weights should decay with distance."""
        cfg = GraphBuildConfig(two_edge_mode=True, ctx_max_sent_distance=6)
        builder = KnowledgeGraphBuilder(cfg)
        facts = [
            RetrievedFact("Doc", 0, "First sentence.", 5.0),
            RetrievedFact("Doc", 1, "Second sentence.", 4.0),
            RetrievedFact("Doc", 3, "Fourth sentence.", 3.0),
            RetrievedFact("Doc", 7, "Eighth sentence.", 2.0),  # dist=4 from F2
        ]
        g = builder.build_graph("test", facts)
        # Adjacent (dist=1) → w=0.9
        w_adj = g.edges["F0", "F1"]["w_ctx"]
        # dist=2 → w=0.6
        w_near = g.edges["F1", "F2"]["w_ctx"]
        # dist=4 → w=0.3
        w_far = g.edges["F2", "F3"]["w_ctx"]
        assert w_adj > w_near > w_far, f"Weights should decay: {w_adj} > {w_near} > {w_far}"

    def test_similarity_edges_use_tfidf(self):
        """v5: Cross-title edges should fire on TF-IDF similarity even without entities."""
        cfg = GraphBuildConfig(
            two_edge_mode=True,
            sim_alpha=0.6,
            sim_beta=0.4,
            sim_edge_threshold=0.1,  # low threshold to catch TF-IDF matches
        )
        builder = KnowledgeGraphBuilder(cfg)
        # Two facts sharing vocabulary but NO named entities (all lowercase context)
        facts = [
            RetrievedFact("Doc A", 0, "the quantum theory of light and photons in vacuum", 5.0),
            RetrievedFact("Doc B", 0, "the quantum theory of waves and light propagation", 4.0),
        ]
        g = builder.build_graph("quantum light", facts)
        # Should have a similarity edge because of shared vocabulary
        assert g.has_edge("F0", "F1"), "TF-IDF similar facts should be connected"
        data = g.edges["F0", "F1"]
        assert data.get("edge_type") == "similarity"
        assert data.get("w_sim", 0) > 0

    def test_backward_compat_no_two_edge(self):
        """When two_edge_mode=False, graph should match legacy behavior."""
        cfg_legacy = GraphBuildConfig(two_edge_mode=False, q_link_top_k=3)
        cfg_v5_off = GraphBuildConfig(two_edge_mode=False, q_link_top_k=3)
        builder_a = KnowledgeGraphBuilder(cfg_legacy)
        builder_b = KnowledgeGraphBuilder(cfg_v5_off)
        facts = _make_facts()
        q = "Who influenced Einstein?"
        g_a = builder_a.build_graph(q, facts)
        g_b = builder_b.build_graph(q, facts)
        assert set(g_a.edges()) == set(g_b.edges())

    def test_edge_type_annotation(self):
        """v5: Fact-to-fact edges should carry edge_type metadata."""
        cfg = GraphBuildConfig(
            two_edge_mode=True,
            sim_edge_threshold=0.1,
            q_link_top_k=3,
        )
        builder = KnowledgeGraphBuilder(cfg)
        facts = _make_facts()
        g = builder.build_graph("Who influenced Einstein?", facts)
        for u, v, data in g.edges(data=True):
            if u == "Q" or v == "Q":
                continue  # Q edges don't have edge_type
            assert "edge_type" in data, f"Edge ({u},{v}) missing edge_type"
            assert data["edge_type"] in ("context", "similarity")


class TestEdgeDGScore:
    """Phase 1: Tests for compute_edge_dg_scores() structural importance."""

    def test_bridge_gets_negative_dg_score(self):
        """Bridge edges should get dg_score = -1.0 (structurally important)."""
        # Linear chain: F0 - F1 - F2 (F1 is a bridge node, both edges are bridges)
        g = nx.Graph()
        g.add_node("Q", type="question")
        g.add_node("F0", type="fact")
        g.add_node("F1", type="fact")
        g.add_node("F2", type="fact")
        g.add_edge("Q", "F0", weight=0.9)
        g.add_edge("F0", "F1", weight=0.8)
        g.add_edge("F1", "F2", weight=0.6)

        stats = compute_edge_dg_scores(g)
        assert stats["dg_bridge_edges"] == 2, "Linear chain should have 2 bridges"

        # Both fact-fact edges should have dg_score = -1.0
        for u, v, data in g.edges(data=True):
            if u == "Q" or v == "Q":
                continue
            assert data.get("dg_score") == -1.0, (
                f"Bridge edge ({u},{v}) should have dg_score=-1.0"
            )

    def test_cycle_gets_positive_dg_score(self):
        """Cycle edges should get dg_score = +1.0 (potentially noisy)."""
        # Build a complete triangle: 3 nodes, 3 edges → 1 cycle
        g = nx.Graph()
        g.add_node("Q", type="question")
        g.add_node("F0", type="fact")
        g.add_node("F1", type="fact")
        g.add_node("F2", type="fact")
        g.add_edge("Q", "F0", weight=0.9)
        g.add_edge("F0", "F1", weight=0.8)
        g.add_edge("F1", "F2", weight=0.6)
        g.add_edge("F0", "F2", weight=0.5)  # creates a cycle F0-F1-F2

        stats = compute_edge_dg_scores(g)
        assert stats["dg_cycle_edges"] > 0, "Triangle should have cycle edges"

        # All fact-fact edges in a triangle are cycle edges (no bridges in a cycle)
        for u, v, data in g.edges(data=True):
            if u == "Q" or v == "Q":
                continue
            assert data.get("dg_score") == 1.0, (
                f"Edge ({u},{v}) in triangle should be cycle (dg_score=+1.0)"
            )

    def test_q_edges_excluded(self):
        """Q edges should NOT have dg_score attribute."""
        builder = KnowledgeGraphBuilder()
        facts = _make_facts()
        g = builder.build_graph("test", facts)
        compute_edge_dg_scores(g)

        for u, v, data in g.edges(data=True):
            if u == "Q" or v == "Q":
                assert "dg_score" not in data, (
                    f"Q edge ({u},{v}) should not have dg_score"
                )

    def test_empty_graph_returns_zeros(self):
        """Graph with no fact-fact edges should return zero stats."""
        g = nx.Graph()
        g.add_node("Q", type="question")
        g.add_node("F0", type="fact")
        g.add_edge("Q", "F0", weight=0.9)

        stats = compute_edge_dg_scores(g)
        assert stats["dg_bridge_edges"] == 0
        assert stats["dg_cycle_edges"] == 0
        assert stats["dg_score_mean"] == 0.0

    def test_dg_score_mean_correct(self):
        """dg_score_mean should be (-bridges + cycles) / total."""
        # Linear chain: F0-F1-F2 → 2 bridges, 0 cycles → mean = -1.0
        g = nx.Graph()
        g.add_node("Q", type="question")
        g.add_node("F0", type="fact")
        g.add_node("F1", type="fact")
        g.add_node("F2", type="fact")
        g.add_edge("Q", "F0", weight=0.9)
        g.add_edge("F0", "F1", weight=0.8)
        g.add_edge("F1", "F2", weight=0.6)

        stats = compute_edge_dg_scores(g)
        assert stats["dg_bridge_edges"] == 2
        assert stats["dg_cycle_edges"] == 0
        assert stats["dg_score_mean"] == -1.0
