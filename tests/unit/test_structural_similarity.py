"""
Unit tests for structural similarity evaluation.

Tests the StructuralSimilarityEvaluator and related functions.
"""

import pytest
import networkx as nx
import numpy as np

from insightspike.config.models import StructuralSimilarityConfig
from insightspike.algorithms.structural_similarity import (
    StructuralSimilarityEvaluator,
    SimilarityResult,
    compute_structural_similarity,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default enabled configuration."""
    return StructuralSimilarityConfig(
        enabled=True,
        method="signature",
        analogy_threshold=0.7,
        cross_domain_only=False,
    )


@pytest.fixture
def cross_domain_config():
    """Configuration with cross_domain_only enabled."""
    return StructuralSimilarityConfig(
        enabled=True,
        method="signature",
        analogy_threshold=0.7,
        cross_domain_only=True,
    )


@pytest.fixture
def star_graph():
    """Create a star graph (hub-and-spoke pattern)."""
    G = nx.star_graph(5)
    # Add domain attribute
    for node in G.nodes():
        G.nodes[node]["domain"] = "astronomy"
    return G


@pytest.fixture
def chain_graph():
    """Create a chain/path graph."""
    G = nx.path_graph(6)
    for node in G.nodes():
        G.nodes[node]["domain"] = "chemistry"
    return G


@pytest.fixture
def solar_system_graph():
    """Create a graph representing solar system structure."""
    G = nx.Graph()
    G.add_node("sun", domain="astronomy", role="hub")
    planets = ["mercury", "venus", "earth", "mars", "jupiter"]
    for planet in planets:
        G.add_node(planet, domain="astronomy", role="spoke")
        G.add_edge("sun", planet, relation="orbits")
    return G


@pytest.fixture
def atom_graph():
    """Create a graph representing atomic structure."""
    G = nx.Graph()
    G.add_node("nucleus", domain="physics", role="hub")
    electrons = ["e1", "e2", "e3", "e4", "e5"]
    for e in electrons:
        G.add_node(e, domain="physics", role="spoke")
        G.add_edge("nucleus", e, relation="orbits")
    return G


@pytest.fixture
def company_graph():
    """Create a graph representing company structure."""
    G = nx.Graph()
    G.add_node("hq", domain="business", role="hub")
    branches = ["branch_a", "branch_b", "branch_c", "branch_d", "branch_e"]
    for b in branches:
        G.add_node(b, domain="business", role="spoke")
        G.add_edge("hq", b, relation="manages")
    return G


# =============================================================================
# Basic Tests
# =============================================================================

class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_disabled_config_returns_zero(self):
        """Disabled config should return similarity=0."""
        config = StructuralSimilarityConfig(enabled=False)
        evaluator = StructuralSimilarityEvaluator(config)

        G1 = nx.star_graph(5)
        G2 = nx.star_graph(5)

        result = evaluator.evaluate(G1, G2)
        assert result.similarity == 0.0
        assert result.method == "disabled"
        assert result.is_analogy is False

    def test_empty_graph_returns_zero(self, default_config):
        """Empty graphs should return similarity=0."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        G1 = nx.Graph()
        G2 = nx.star_graph(5)

        result = evaluator.evaluate(G1, G2)
        assert result.similarity == 0.0

    def test_identical_graphs_high_similarity(self, default_config):
        """Identical graphs should have similarity close to 1.0."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        G = nx.star_graph(5)
        result = evaluator.evaluate(G, G)

        assert result.similarity == pytest.approx(1.0, rel=0.01)
        assert result.method == "signature"

    def test_similar_structure_different_size(self, default_config):
        """Same structure type with different sizes should still be similar."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        star_small = nx.star_graph(3)
        star_large = nx.star_graph(10)

        result = evaluator.evaluate(star_small, star_large)
        # Should still recognize the hub-spoke pattern
        assert result.similarity > 0.5


# =============================================================================
# Pattern Recognition Tests
# =============================================================================

class TestPatternRecognition:
    """Tests for recognizing structural patterns."""

    def test_star_vs_chain_different_structure(self, default_config):
        """Star and chain graphs have different structural properties."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        # Star: center node connected to many spokes
        star = nx.star_graph(5)
        for node in star.nodes():
            star.nodes[node]["domain"] = "astronomy"

        # Chain: linear path (different structure)
        chain = nx.path_graph(10)
        for node in chain.nodes():
            chain.nodes[node]["domain"] = "chemistry"

        # Evaluate from center nodes
        result = evaluator.evaluate(star, chain, center1=0, center2=0)

        # The signature method captures structural differences
        # Chain grows linearly (1->2->3->4 nodes per hop)
        # Star reaches all nodes at hop 1
        # Similarity is not 1.0, indicating structural difference detected
        assert result.similarity < 1.0
        # Verify signatures are different
        assert not np.allclose(result.signature_a, result.signature_b)

    def test_two_stars_high_similarity(self, default_config):
        """Two star graphs should have high similarity."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        star1 = nx.star_graph(5)
        star2 = nx.star_graph(7)

        result = evaluator.evaluate(star1, star2)
        assert result.similarity > 0.7

    def test_two_chains_high_similarity(self, default_config):
        """Two chain graphs should have high similarity."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        chain1 = nx.path_graph(5)
        chain2 = nx.path_graph(8)

        result = evaluator.evaluate(chain1, chain2)
        assert result.similarity > 0.7

    def test_clique_vs_star_different(self, default_config):
        """Clique and star should be recognized as different."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        clique = nx.complete_graph(5)
        star = nx.star_graph(5)

        result = evaluator.evaluate(clique, star)
        assert result.similarity < 0.8  # Different patterns


# =============================================================================
# Cross-Domain Analogy Tests
# =============================================================================

class TestCrossDomainAnalogy:
    """Tests for cross-domain analogy detection."""

    def test_solar_system_atom_analogy(
        self, cross_domain_config, solar_system_graph, atom_graph
    ):
        """Solar system and atom should be recognized as analogous."""
        evaluator = StructuralSimilarityEvaluator(cross_domain_config)

        result = evaluator.evaluate(
            solar_system_graph, atom_graph,
            center1="sun", center2="nucleus"
        )

        # Should detect structural similarity
        assert result.similarity > 0.7
        # Should recognize as cross-domain
        assert result.is_cross_domain is True
        # Should mark as analogy
        assert result.is_analogy is True

    def test_solar_system_company_analogy(
        self, cross_domain_config, solar_system_graph, company_graph
    ):
        """Solar system and company (hub-spoke) should be analogous."""
        evaluator = StructuralSimilarityEvaluator(cross_domain_config)

        result = evaluator.evaluate(
            solar_system_graph, company_graph,
            center1="sun", center2="hq"
        )

        assert result.similarity > 0.7
        assert result.is_cross_domain is True

    def test_same_domain_not_analogy_when_cross_only(
        self, cross_domain_config, solar_system_graph
    ):
        """Same domain graphs should not be marked as analogy when cross_domain_only=True."""
        evaluator = StructuralSimilarityEvaluator(cross_domain_config)

        # Create another astronomy graph
        G2 = solar_system_graph.copy()

        result = evaluator.evaluate(solar_system_graph, G2)

        # High similarity but same domain
        assert result.similarity > 0.9
        assert result.is_cross_domain is False
        assert result.is_analogy is False  # Not marked as analogy


# =============================================================================
# Analogy Bonus Tests
# =============================================================================

class TestAnalogyBonus:
    """Tests for analogy bonus calculation."""

    def test_analogy_bonus_when_detected(
        self, cross_domain_config, solar_system_graph, atom_graph
    ):
        """Should return positive bonus when analogy detected."""
        evaluator = StructuralSimilarityEvaluator(cross_domain_config)

        bonus = evaluator.compute_analogy_bonus(
            solar_system_graph, atom_graph,
            center1="sun", center2="nucleus"
        )

        assert bonus > 0
        # Should be similarity * analogy_weight
        expected_max = cross_domain_config.analogy_weight
        assert bonus <= expected_max

    def test_no_bonus_when_disabled(self, solar_system_graph, atom_graph):
        """Should return 0 bonus when disabled."""
        config = StructuralSimilarityConfig(enabled=False)
        evaluator = StructuralSimilarityEvaluator(config)

        bonus = evaluator.compute_analogy_bonus(solar_system_graph, atom_graph)
        assert bonus == 0.0

    def test_no_bonus_below_threshold(self):
        """Should return 0 bonus when similarity is below threshold."""
        config = StructuralSimilarityConfig(
            enabled=True,
            analogy_threshold=0.99,  # Very high threshold
            cross_domain_only=False,
        )
        evaluator = StructuralSimilarityEvaluator(config)

        # Create structurally different graphs
        star = nx.star_graph(5)  # Hub-spoke
        clique = nx.complete_graph(6)  # Fully connected

        # These have different structure, so similarity should be below 0.99
        bonus = evaluator.compute_analogy_bonus(star, clique, center1=0)
        # With high threshold, bonus should be 0
        assert bonus == 0.0


# =============================================================================
# Signature Extraction Tests
# =============================================================================

class TestSignatureExtraction:
    """Tests for signature extraction."""

    def test_signature_length(self, default_config):
        """Signature length should match expected dimensions."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        G = nx.star_graph(5)
        sig = evaluator._extract_signature(G, center=None)

        # Each hop: nodes, edges, density, triangles, clustering, degree/cycle stats
        # max_hops + 1 = 4 hops (0, 1, 2, 3)
        expected_features_per_hop = 2 + int(default_config.include_density) + \
                                    int(default_config.include_triangles) + \
                                    int(default_config.include_clustering) + 3
        expected_length = (default_config.max_signature_hops + 1) * expected_features_per_hop

        assert len(sig) == expected_length

    def test_signature_with_center(self, default_config, solar_system_graph):
        """Signature extraction with center node."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        sig = evaluator._extract_signature(solar_system_graph, center="sun")

        # Should be valid numpy array
        assert isinstance(sig, np.ndarray)
        assert len(sig) > 0
        assert not np.isnan(sig).any()


# =============================================================================
# Method Variants Tests
# =============================================================================

class TestMethodVariants:
    """Tests for different similarity methods."""

    def test_spectral_method(self):
        """Test spectral similarity method."""
        config = StructuralSimilarityConfig(
            enabled=True,
            method="spectral",
            spectral_k=5,
        )
        evaluator = StructuralSimilarityEvaluator(config)

        G1 = nx.star_graph(5)
        G2 = nx.star_graph(5)

        result = evaluator.evaluate(G1, G2)
        assert result.method == "spectral"
        assert result.similarity > 0.9

    def test_motif_method(self):
        """Test motif similarity method."""
        config = StructuralSimilarityConfig(
            enabled=True,
            method="motif",
        )
        evaluator = StructuralSimilarityEvaluator(config)

        G1 = nx.star_graph(5)
        G2 = nx.star_graph(5)

        result = evaluator.evaluate(G1, G2)
        assert result.method == "motif"
        assert result.similarity > 0.8

    def test_invalid_method_raises_validation_error(self):
        """Invalid method should raise validation error from Pydantic."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            StructuralSimilarityConfig(
                enabled=True,
                method="unknown_method",  # type: ignore
            )


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunction:
    """Tests for compute_structural_similarity function."""

    def test_quick_similarity(self):
        """Test convenience function."""
        G1 = nx.star_graph(5)
        G2 = nx.star_graph(5)

        result = compute_structural_similarity(G1, G2)

        assert isinstance(result, SimilarityResult)
        assert result.similarity > 0.9

    def test_with_custom_threshold(self):
        """Test with custom threshold."""
        G1 = nx.star_graph(5)
        G2 = nx.path_graph(6)

        result = compute_structural_similarity(G1, G2, threshold=0.3)

        # May or may not be analogy depending on actual similarity
        assert isinstance(result, SimilarityResult)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_node_graph(self, default_config):
        """Single node graphs should handle gracefully."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        G1 = nx.Graph()
        G1.add_node("a")
        G2 = nx.Graph()
        G2.add_node("b")

        result = evaluator.evaluate(G1, G2)
        assert result.similarity >= 0.0
        assert result.similarity <= 1.0

    def test_disconnected_graph(self, default_config):
        """Disconnected graphs should handle gracefully."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3])  # No edges
        G2 = nx.star_graph(3)

        result = evaluator.evaluate(G1, G2)
        # Should return valid result
        assert 0.0 <= result.similarity <= 1.0

    def test_invalid_center_node(self, default_config):
        """Invalid center node should handle gracefully."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        G = nx.star_graph(5)
        result = evaluator.evaluate(G, G, center1="nonexistent", center2="also_nonexistent")

        # Should still work using full graph
        assert result.similarity > 0.9


# =============================================================================
# Integration-like Tests
# =============================================================================

class TestRealWorldPatterns:
    """Tests with more realistic patterns."""

    def test_hierarchical_patterns(self, default_config):
        """Test recognition of hierarchical patterns."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        # Create tree-like hierarchies
        tree1 = nx.balanced_tree(2, 3)  # Binary tree, height 3
        tree2 = nx.balanced_tree(2, 3)

        result = evaluator.evaluate(tree1, tree2)
        assert result.similarity > 0.9

    def test_ring_patterns(self, default_config):
        """Test recognition of ring/cycle patterns."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        ring1 = nx.cycle_graph(6)
        ring2 = nx.cycle_graph(8)

        result = evaluator.evaluate(ring1, ring2)
        assert result.similarity > 0.7

    def test_grid_patterns(self, default_config):
        """Test recognition of grid patterns."""
        evaluator = StructuralSimilarityEvaluator(default_config)

        grid1 = nx.grid_2d_graph(3, 3)
        grid2 = nx.grid_2d_graph(4, 4)

        result = evaluator.evaluate(grid1, grid2)
        assert result.similarity > 0.6
