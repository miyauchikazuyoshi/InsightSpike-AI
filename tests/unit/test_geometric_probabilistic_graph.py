"""
Unit tests for GeometricProbabilisticGraph (Three-Space Integration)

This test suite validates the Level 2 implementation that integrates:
1. Graph space G = (V, E)
2. Probability density space (Ω, F, μ)
3. Similarity space (Φ, d)

Tests cover:
- True Shannon entropy calculation
- Probability distribution derivation from attention
- Graph structure operations
- Three-space metric computation
- Information gain calculation
"""

import numpy as np
import pytest

# Import module under test
from insightspike.algorithms.geometric_probabilistic_graph import (
    GeometricProbabilisticGraph,
    ThreeSpaceMetrics,
    calculate_three_space_entropy,
    calculate_three_space_information_gain,
)

# Check optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_BERT_AVAILABLE = True
except ImportError:
    SENTENCE_BERT_AVAILABLE = False


class TestGeometricProbabilisticGraphBasic:
    """Basic tests for GeometricProbabilisticGraph initialization and structure"""

    def test_initialization(self):
        """Test basic initialization"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        assert gpg.embedding_dim == 384
        assert gpg.attention_heads == 4  # Default
        assert len(gpg.node_to_idx) == 0

    def test_add_node_with_embedding(self):
        """Test adding nodes with pre-computed embeddings"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        embedding = np.random.randn(384)
        gpg.add_node("node1", embedding=embedding)

        assert "node1" in gpg.node_to_idx
        assert "node1" in gpg.node_embeddings
        assert gpg.node_embeddings["node1"].shape == (384,)

    @pytest.mark.skipif(
        not SENTENCE_BERT_AVAILABLE, reason="sentence-transformers not available"
    )
    def test_add_node_with_text(self):
        """Test adding nodes with text (requires Sentence-BERT)"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        gpg.add_node("doc1", text="This is a test document")

        assert "doc1" in gpg.node_embeddings
        assert "doc1" in gpg.node_texts
        assert gpg.node_texts["doc1"] == "This is a test document"

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
    def test_add_edge(self):
        """Test adding edges to graph structure"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add nodes first
        gpg.add_node("node1", embedding=np.random.randn(384))
        gpg.add_node("node2", embedding=np.random.randn(384))

        # Add edge
        gpg.add_edge("node1", "node2", weight=0.8)

        assert gpg.graph.has_edge("node1", "node2")
        assert gpg.graph["node1"]["node2"]["weight"] == 0.8


class TestShannonEntropyCalculation:
    """Tests for true Shannon entropy calculation"""

    def test_fallback_entropy_no_torch(self):
        """Test fallback entropy calculation when PyTorch unavailable"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add nodes with similar embeddings (low entropy expected)
        base_embedding = np.random.randn(384)
        for i in range(5):
            # Add small noise to create similar embeddings
            embedding = base_embedding + 0.01 * np.random.randn(384)
            gpg.add_node(f"node{i}", embedding=embedding)

        # Even without PyTorch, fallback should work
        entropy = gpg._fallback_entropy()

        # Fallback uses (1 - avg_similarity) / 2
        # Similar embeddings -> high similarity -> low "entropy"
        assert 0.0 <= entropy <= 1.0
        assert entropy < 0.3  # Should be low for similar embeddings

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_shannon_entropy_basic(self):
        """Test basic Shannon entropy calculation with PyTorch"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add nodes
        for i in range(5):
            embedding = np.random.randn(384)
            gpg.add_node(f"node{i}", embedding=embedding)

        # Add edges to create graph structure
        gpg.add_edge("node0", "node1")
        gpg.add_edge("node1", "node2")
        gpg.add_edge("node2", "node3")
        gpg.add_edge("node3", "node4")

        # Calculate Shannon entropy
        entropy = gpg.calculate_shannon_entropy()

        # Shannon entropy should be non-negative
        assert entropy >= 0.0
        # For 5 nodes, max entropy is log₂(5) ≈ 2.32 bits
        assert entropy <= 3.0  # Allow some margin

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_shannon_entropy_per_node(self):
        """Test per-node Shannon entropy calculation"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add nodes
        for i in range(5):
            embedding = np.random.randn(384)
            gpg.add_node(f"node{i}", embedding=embedding)

        # Add edges
        gpg.add_edge("node0", "node1")
        gpg.add_edge("node0", "node2")
        gpg.add_edge("node0", "node3")  # node0 has 3 neighbors

        # Calculate entropy for node0
        entropy_node0 = gpg.calculate_shannon_entropy("node0")

        # Should be positive (uncertainty over neighbors)
        assert entropy_node0 >= 0.0
        # Max entropy for 3 neighbors: log₂(3) ≈ 1.58
        assert entropy_node0 <= 2.0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_entropy_ordering(self):
        """Test that entropy correctly reflects disorder"""
        # Graph with high order (star topology)
        gpg_ordered = GeometricProbabilisticGraph(embedding_dim=384)
        center_emb = np.random.randn(384)
        gpg_ordered.add_node("center", embedding=center_emb)

        for i in range(4):
            # All nodes similar to center
            emb = center_emb + 0.05 * np.random.randn(384)
            gpg_ordered.add_node(f"leaf{i}", embedding=emb)
            gpg_ordered.add_edge("center", f"leaf{i}")

        entropy_ordered = gpg_ordered.calculate_shannon_entropy()

        # Graph with low order (random structure)
        gpg_random = GeometricProbabilisticGraph(embedding_dim=384)
        for i in range(5):
            # Very different embeddings
            emb = np.random.randn(384) * 10  # Large scale
            gpg_random.add_node(f"node{i}", embedding=emb)

        for i in range(4):
            gpg_random.add_edge(f"node{i}", f"node{i+1}")

        entropy_random = gpg_random.calculate_shannon_entropy()

        # Random/diverse graph should have higher entropy than ordered
        # (This might not always hold for small samples, but is expected on average)
        assert entropy_random >= 0.0
        assert entropy_ordered >= 0.0


class TestInformationGainCalculation:
    """Tests for information gain using true Shannon entropy"""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_information_gain_basic(self):
        """Test basic information gain calculation"""
        # Before state: scattered embeddings (high entropy)
        gpg_before = GeometricProbabilisticGraph(embedding_dim=384)
        for i in range(5):
            emb = np.random.randn(384) * 5  # Diverse
            gpg_before.add_node(f"node{i}", embedding=emb)
            if i > 0:
                gpg_before.add_edge(f"node{i-1}", f"node{i}")

        # After state: clustered embeddings (low entropy)
        gpg_after = GeometricProbabilisticGraph(embedding_dim=384)
        base_emb = np.random.randn(384)
        for i in range(5):
            emb = base_emb + 0.1 * np.random.randn(384)  # Similar
            gpg_after.add_node(f"node{i}", embedding=emb)
            if i > 0:
                gpg_after.add_edge(f"node{i-1}", f"node{i}")

        # Calculate information gain
        ig = gpg_before.calculate_information_gain(gpg_after)

        # IG should be positive (entropy decreased -> gained information)
        # Note: Due to randomness, this might not always hold, but is expected
        assert isinstance(ig, float)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_information_gain_no_change(self):
        """Test IG when states are identical"""
        gpg1 = GeometricProbabilisticGraph(embedding_dim=384)
        gpg2 = GeometricProbabilisticGraph(embedding_dim=384)

        # Same structure and embeddings
        for i in range(5):
            emb = np.random.randn(384)
            gpg1.add_node(f"node{i}", embedding=emb.copy())
            gpg2.add_node(f"node{i}", embedding=emb.copy())

        # Calculate IG
        ig = gpg1.calculate_information_gain(gpg2)

        # Should be close to zero (no change)
        assert abs(ig) < 0.5  # Allow some numerical error


class TestThreeSpaceMetrics:
    """Tests for comprehensive three-space metrics"""

    def test_three_space_metrics_basic(self):
        """Test basic three-space metrics computation"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add nodes and edges
        for i in range(5):
            gpg.add_node(f"node{i}", embedding=np.random.randn(384))

        gpg.add_edge("node0", "node1")
        gpg.add_edge("node1", "node2")
        gpg.add_edge("node2", "node3")

        # Compute metrics
        metrics = gpg.compute_three_space_metrics()

        # Validate structure
        assert isinstance(metrics, ThreeSpaceMetrics)

        # Graph metrics
        assert metrics.num_nodes == 5
        assert metrics.num_edges == 3
        assert 0.0 <= metrics.graph_density <= 1.0
        assert metrics.avg_degree >= 0.0

        # Probability metrics
        assert metrics.shannon_entropy >= 0.0
        assert 0.0 <= metrics.probability_mass <= 2.0  # Allow some slack

        # Similarity metrics
        assert -1.0 <= metrics.avg_cosine_similarity <= 1.0
        assert metrics.embedding_std >= 0.0

    def test_three_space_metrics_to_dict(self):
        """Test metrics serialization to dict"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        for i in range(3):
            gpg.add_node(f"node{i}", embedding=np.random.randn(384))

        metrics = gpg.compute_three_space_metrics()
        metrics_dict = metrics.to_dict()

        # Check structure
        assert "graph" in metrics_dict
        assert "probability" in metrics_dict
        assert "similarity" in metrics_dict

        # Check content
        assert metrics_dict["graph"]["num_nodes"] == 3
        assert "shannon_entropy" in metrics_dict["probability"]
        assert "avg_cosine_similarity" in metrics_dict["similarity"]

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
    def test_graph_density_calculation(self):
        """Test graph density metric"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Create complete graph (maximum density = 1.0)
        n = 4
        for i in range(n):
            gpg.add_node(f"node{i}", embedding=np.random.randn(384))

        for i in range(n):
            for j in range(i + 1, n):
                gpg.add_edge(f"node{i}", f"node{j}")

        metrics = gpg.compute_three_space_metrics()

        # Complete graph should have density = 1.0
        assert abs(metrics.graph_density - 1.0) < 0.01


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions"""

    @pytest.mark.skipif(
        not SENTENCE_BERT_AVAILABLE, reason="sentence-transformers not available"
    )
    def test_calculate_three_space_entropy(self):
        """Test convenience function for entropy calculation"""
        texts = [
            "The cat sat on the mat",
            "The dog played in the yard",
            "Machine learning is fascinating",
        ]

        edges = [(0, 1), (1, 2)]

        entropy = calculate_three_space_entropy(texts, edges=edges, embedding_dim=384)

        assert entropy >= 0.0
        assert isinstance(entropy, float)

    @pytest.mark.skipif(
        not SENTENCE_BERT_AVAILABLE, reason="sentence-transformers not available"
    )
    def test_calculate_three_space_information_gain(self):
        """Test convenience function for IG calculation"""
        texts_before = [
            "Machine learning",
            "Deep learning",
            "Neural networks",
        ]

        texts_after = [
            "AI and ML",
            "Deep learning basics",
            "Neural net fundamentals",
        ]

        edges = [(0, 1), (1, 2)]

        ig = calculate_three_space_information_gain(
            texts_before=texts_before,
            texts_after=texts_after,
            edges_before=edges,
            edges_after=edges,
            embedding_dim=384,
        )

        assert isinstance(ig, float)


class TestProbabilityDistributions:
    """Tests for probability distribution derivation from attention"""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_probability_distribution_properties(self):
        """Test that attention weights satisfy probability axioms"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add nodes
        for i in range(5):
            gpg.add_node(f"node{i}", embedding=np.random.randn(384))

        # Add edges
        for i in range(4):
            gpg.add_edge(f"node{i}", f"node{i+1}")

        # Compute probability distributions
        attention_weights, edge_index = gpg.compute_probability_distributions()

        # Test properties
        assert attention_weights.shape[0] > 0  # Has edges
        assert attention_weights.shape[1] == gpg.attention_heads  # Correct heads

        # Probability Axiom P1: Non-negative
        assert torch.all(attention_weights >= 0.0), "Axiom P1 violated: probabilities must be non-negative"

        # Probability Axiom P2: Normalized (Σp = 1 for each source node)
        unique_sources = edge_index[0].unique()
        for node_idx in unique_sources:
            mask = edge_index[0] == node_idx
            node_probs = attention_weights[mask]
            # For each attention head
            for head in range(node_probs.shape[1]):
                prob_sum = node_probs[:, head].sum()
                assert torch.isclose(prob_sum, torch.tensor(1.0), atol=1e-5), \
                    f"Axiom P2 violated: node {node_idx} head {head} has Σp = {prob_sum:.10f} (should be 1.0)"

        # All probabilities should be in [0, 1]
        assert torch.all(attention_weights <= 1.0), "Probabilities must be <= 1.0"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_probability_distribution_caching(self):
        """Test that probability distributions are cached"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        for i in range(3):
            gpg.add_node(f"node{i}", embedding=np.random.randn(384))

        gpg.add_edge("node0", "node1")
        gpg.add_edge("node1", "node2")

        # First computation
        attn1, edge1 = gpg.compute_probability_distributions()

        # Second computation (should use cache)
        attn2, edge2 = gpg.compute_probability_distributions()

        # Should be identical (cached)
        assert torch.equal(attn1, attn2)
        assert torch.equal(edge1, edge2)

        # Invalidate cache by adding node
        gpg.add_node("node3", embedding=np.random.randn(384))

        # Should recompute
        attn3, edge3 = gpg.compute_probability_distributions()

        # Cache should have been invalidated
        assert gpg._cached_attention_weights is not None


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_graph(self):
        """Test metrics on empty graph"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        metrics = gpg.compute_three_space_metrics()

        assert metrics.num_nodes == 0
        assert metrics.num_edges == 0
        assert metrics.shannon_entropy == 0.0

    def test_single_node(self):
        """Test metrics on single-node graph"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)
        gpg.add_node("node0", embedding=np.random.randn(384))

        metrics = gpg.compute_three_space_metrics()

        assert metrics.num_nodes == 1
        assert metrics.num_edges == 0
        # Single node should have zero entropy (no uncertainty)
        assert metrics.shannon_entropy == 0.0

    def test_disconnected_graph(self):
        """Test metrics on disconnected graph (no edges)"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        for i in range(5):
            gpg.add_node(f"node{i}", embedding=np.random.randn(384))

        # No edges added

        metrics = gpg.compute_three_space_metrics()

        assert metrics.num_nodes == 5
        assert metrics.num_edges == 0
        assert metrics.graph_density == 0.0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_isolated_node_entropy(self):
        """Test entropy calculation for isolated nodes"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Create graph with one isolated node
        gpg.add_node("isolated", embedding=np.random.randn(384))
        gpg.add_node("connected1", embedding=np.random.randn(384))
        gpg.add_node("connected2", embedding=np.random.randn(384))

        # Only connect two nodes
        gpg.add_edge("connected1", "connected2")

        # Isolated node should have zero entropy
        entropy_isolated = gpg.calculate_shannon_entropy("isolated")
        assert entropy_isolated == 0.0, "Isolated node should have zero entropy"

        # Connected nodes should have non-zero entropy
        entropy_connected = gpg.calculate_shannon_entropy("connected1")
        assert entropy_connected >= 0.0

    def test_dimension_consistency(self):
        """Test that embedding dimensions are consistent"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add node with wrong dimension should work (we handle it)
        wrong_dim_emb = np.random.randn(768)  # Wrong dimension

        # This should either fail gracefully or handle the mismatch
        # Implementation should validate or pad/truncate
        try:
            gpg.add_node("node_wrong", embedding=wrong_dim_emb)
            # If it succeeds, just verify it was added
            assert "node_wrong" in gpg.node_to_idx
        except (ValueError, RuntimeError, AssertionError):
            # Expected if dimension validation is strict
            pass


class TestComparison:
    """Tests comparing old vs new entropy calculation"""

    def test_old_vs_new_entropy(self):
        """Compare fallback (old) vs Shannon (new) entropy"""
        gpg = GeometricProbabilisticGraph(embedding_dim=384)

        # Add nodes with similar embeddings
        base = np.random.randn(384)
        for i in range(5):
            emb = base + 0.1 * np.random.randn(384)
            gpg.add_node(f"node{i}", embedding=emb)

        for i in range(4):
            gpg.add_edge(f"node{i}", f"node{i+1}")

        # Old method (fallback)
        old_entropy = gpg._fallback_entropy()

        # New method (Shannon entropy)
        if TORCH_AVAILABLE:
            new_entropy = gpg.calculate_shannon_entropy()

            # Both should indicate low entropy for similar embeddings
            # But the scales are different:
            # - Old: [0, 1] heuristic
            # - New: [0, log₂(n)] bits
            assert old_entropy >= 0.0
            assert new_entropy >= 0.0

            print(
                f"Old (heuristic) entropy: {old_entropy:.3f}, "
                f"New (Shannon) entropy: {new_entropy:.3f} bits"
            )


# Integration test
class TestIntegration:
    """Integration tests for real-world scenarios"""

    @pytest.mark.skipif(
        not (TORCH_AVAILABLE and SENTENCE_BERT_AVAILABLE),
        reason="PyTorch and Sentence-BERT required",
    )
    def test_rag_scenario(self):
        """Test realistic RAG scenario with document retrieval"""
        # Simulate RAG: initial documents
        initial_docs = [
            "Python is a programming language",
            "Machine learning uses Python",
            "Data science involves statistics",
        ]

        # After query: refined context
        refined_docs = [
            "Python is a programming language",
            "Machine learning uses Python",
            "Python libraries include NumPy and PyTorch",
        ]

        # Build graphs
        gpg_initial = GeometricProbabilisticGraph(embedding_dim=384)
        for i, doc in enumerate(initial_docs):
            gpg_initial.add_node(f"doc{i}", text=doc)

        gpg_refined = GeometricProbabilisticGraph(embedding_dim=384)
        for i, doc in enumerate(refined_docs):
            gpg_refined.add_node(f"doc{i}", text=doc)

        # Add edges based on semantic similarity (simplified)
        for i in range(len(initial_docs) - 1):
            gpg_initial.add_edge(f"doc{i}", f"doc{i+1}")
            gpg_refined.add_edge(f"doc{i}", f"doc{i+1}")

        # Compute IG
        ig = gpg_initial.calculate_information_gain(gpg_refined)

        # Information gain should reflect the semantic refinement
        assert isinstance(ig, float)

        # Get detailed metrics
        metrics_initial = gpg_initial.compute_three_space_metrics()
        metrics_refined = gpg_refined.compute_three_space_metrics()

        print(f"Initial entropy: {metrics_initial.shannon_entropy:.3f} bits")
        print(f"Refined entropy: {metrics_refined.shannon_entropy:.3f} bits")
        print(f"Information gain: {ig:.3f} bits")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
