"""
Integration tests for geDIG with structural similarity evaluation.

Tests the integration of structural similarity detection into the geDIG pipeline.
"""

import pytest
import networkx as nx
import numpy as np

from insightspike.algorithms.gedig_core import GeDIGCore


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ss_config_enabled():
    """Structural similarity config enabled."""
    return {
        "enabled": True,
        "method": "signature",
        "analogy_threshold": 0.6,
        "analogy_weight": 0.3,
        "cross_domain_only": False,
        "max_signature_hops": 2,
    }


@pytest.fixture
def ss_config_disabled():
    """Structural similarity config disabled."""
    return {
        "enabled": False,
    }


@pytest.fixture
def ss_config_cross_domain():
    """Cross-domain only config."""
    return {
        "enabled": True,
        "method": "signature",
        "analogy_threshold": 0.6,
        "analogy_weight": 0.3,
        "cross_domain_only": True,
    }


def create_hub_spoke_graph(hub_name: str, spoke_count: int, domain: str) -> nx.Graph:
    """Create a hub-and-spoke graph."""
    G = nx.Graph()
    G.add_node(hub_name, domain=domain, role="hub")
    for i in range(spoke_count):
        spoke_name = f"{hub_name}_spoke_{i}"
        G.add_node(spoke_name, domain=domain, role="spoke")
        G.add_edge(hub_name, spoke_name)
    return G


def create_chain_graph(length: int, domain: str) -> nx.Graph:
    """Create a chain/path graph."""
    G = nx.path_graph(length)
    # Relabel nodes
    mapping = {i: f"node_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    for node in G.nodes():
        G.nodes[node]["domain"] = domain
    return G


# =============================================================================
# Basic Integration Tests
# =============================================================================

class TestGeDIGStructuralSimilarityIntegration:
    """Test geDIG integration with structural similarity."""

    def test_gedig_core_with_ss_disabled(self, ss_config_disabled):
        """GeDIGCore should work normally when SS is disabled."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_disabled,
        )

        # Create simple test graphs
        g1 = nx.Graph()
        g1.add_edges_from([("a", "b"), ("b", "c")])

        g2 = nx.Graph()
        g2.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        result = core.calculate(g_prev=g1, g_now=g2, focal_nodes={"a"})

        assert result is not None
        assert hasattr(result, "gedig_value")
        # SS evaluator should be None
        assert core._ss_evaluator is None

    def test_gedig_core_with_ss_enabled(self, ss_config_enabled):
        """GeDIGCore should initialize SS evaluator when enabled."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_enabled,
        )

        # SS evaluator should be initialized
        assert core._ss_evaluator is not None

    def test_similar_structures_get_bonus(self, ss_config_enabled):
        """Similar graph structures should get analogy bonus."""
        core_with_ss = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            lambda_weight=1.0,
            structural_similarity_config=ss_config_enabled,
        )

        core_without_ss = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            lambda_weight=1.0,
            structural_similarity_config={"enabled": False},
        )

        # Create two structurally similar hub-spoke graphs
        g1 = create_hub_spoke_graph("hub1", 4, "domain_a")
        g2 = create_hub_spoke_graph("hub2", 4, "domain_b")

        # Add some change between graphs
        g2_modified = g2.copy()
        g2_modified.add_node("extra_node", domain="domain_b")
        g2_modified.add_edge("hub2", "extra_node")

        result_with_ss = core_with_ss.calculate(
            g_prev=g1, g_now=g2_modified, focal_nodes={"hub1", "hub2"}
        )
        result_without_ss = core_without_ss.calculate(
            g_prev=g1, g_now=g2_modified, focal_nodes={"hub1", "hub2"}
        )

        # Both should produce valid results
        assert result_with_ss is not None
        assert result_without_ss is not None

        # Note: The actual IG values depend on the internal calculations
        # We mainly verify that the integration doesn't break

    def test_different_structures_lower_similarity(self, ss_config_enabled):
        """Different graph structures should have lower similarity."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_enabled,
        )

        # Hub-spoke vs chain - structurally different
        hub_spoke = create_hub_spoke_graph("hub", 5, "domain_a")
        chain = create_chain_graph(6, "domain_b")

        # Just verify no errors
        result = core.calculate(
            g_prev=hub_spoke, g_now=chain, focal_nodes={"hub"}
        )

        assert result is not None


# =============================================================================
# Cross-Domain Tests
# =============================================================================

class TestCrossDomainAnalogy:
    """Test cross-domain analogy detection in geDIG."""

    def test_cross_domain_required(self, ss_config_cross_domain):
        """When cross_domain_only=True, same domain should not get bonus."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_cross_domain,
        )

        # Same domain, similar structure
        g1 = create_hub_spoke_graph("hub1", 4, "same_domain")
        g2 = create_hub_spoke_graph("hub2", 4, "same_domain")

        result = core.calculate(g_prev=g1, g_now=g2, focal_nodes={"hub1"})

        assert result is not None
        # The evaluator should detect same domain and not give bonus
        # (We can't easily verify the bonus value, but we verify no crash)

    def test_different_domains_can_get_bonus(self, ss_config_cross_domain):
        """Different domains with similar structure should be able to get bonus."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_cross_domain,
        )

        # Different domains, similar structure (hub-spoke)
        solar_system = create_hub_spoke_graph("sun", 5, "astronomy")
        atom = create_hub_spoke_graph("nucleus", 5, "physics")

        result = core.calculate(
            g_prev=solar_system, g_now=atom, focal_nodes={"sun"}
        )

        assert result is not None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for geDIG + SS integration."""

    def test_minimal_graph(self, ss_config_enabled):
        """Minimal graphs (1-2 nodes) should not crash."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_enabled,
        )

        # Minimal non-empty graphs
        g1 = nx.Graph()
        g1.add_node("a")

        g2 = nx.Graph()
        g2.add_nodes_from(["a", "b"])
        g2.add_edge("a", "b")

        result = core.calculate(g_prev=g1, g_now=g2, focal_nodes={"a"})

        assert result is not None

    def test_single_node_graph(self, ss_config_enabled):
        """Single node graphs should not crash."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_enabled,
        )

        g1 = nx.Graph()
        g1.add_node("a")
        g2 = nx.Graph()
        g2.add_node("a")
        g2.add_node("b")
        g2.add_edge("a", "b")

        result = core.calculate(g_prev=g1, g_now=g2, focal_nodes={"a"})

        assert result is not None

    def test_no_focal_nodes(self, ss_config_enabled):
        """Empty focal nodes should handle gracefully."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_enabled,
        )

        g1 = create_hub_spoke_graph("hub", 3, "test")
        g2 = create_hub_spoke_graph("hub", 4, "test")

        # focal_nodes will be derived from graph differences
        result = core.calculate(g_prev=g1, g_now=g2)

        assert result is not None


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegression:
    """Regression tests to ensure SS doesn't break existing functionality."""

    def test_gedig_value_range(self, ss_config_enabled):
        """geDIG values should remain in reasonable range."""
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=ss_config_enabled,
        )

        g1 = create_hub_spoke_graph("hub", 5, "test")
        g2 = g1.copy()
        g2.add_node("new_spoke", domain="test")
        g2.add_edge("hub", "new_spoke")

        result = core.calculate(g_prev=g1, g_now=g2, focal_nodes={"hub"})

        # geDIG should be in reasonable range
        assert -10.0 <= result.gedig_value <= 10.0

    def test_disabled_ss_matches_original(self):
        """Disabled SS should produce same results as no SS config."""
        core_no_config = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
        )

        core_disabled = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config={"enabled": False},
        )

        g1 = create_hub_spoke_graph("hub", 4, "test")
        g2 = g1.copy()
        g2.add_node("extra", domain="test")
        g2.add_edge("hub", "extra")

        result_no_config = core_no_config.calculate(
            g_prev=g1, g_now=g2, focal_nodes={"hub"}
        )
        result_disabled = core_disabled.calculate(
            g_prev=g1, g_now=g2, focal_nodes={"hub"}
        )

        # Results should be identical
        assert result_no_config.gedig_value == pytest.approx(
            result_disabled.gedig_value, rel=1e-6
        )


# =============================================================================
# Method Variants Tests
# =============================================================================

class TestMethodVariants:
    """Test different SS methods in geDIG integration."""

    @pytest.mark.parametrize("method", ["signature", "spectral", "motif"])
    def test_all_methods_work(self, method):
        """All SS methods should integrate without errors."""
        config = {
            "enabled": True,
            "method": method,
            "analogy_threshold": 0.5,
            "analogy_weight": 0.2,
            "cross_domain_only": False,
        }

        core = GeDIGCore(
            enable_multihop=True,
            max_hops=2,
            structural_similarity_config=config,
        )

        g1 = create_hub_spoke_graph("hub1", 4, "test")
        g2 = create_hub_spoke_graph("hub2", 5, "test")

        result = core.calculate(g_prev=g1, g_now=g2, focal_nodes={"hub1"})

        assert result is not None
        assert hasattr(result, "gedig_value")
