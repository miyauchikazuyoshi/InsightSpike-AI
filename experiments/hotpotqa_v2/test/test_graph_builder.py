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
