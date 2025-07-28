"""
Unit tests for the VectorIntegrator module.
"""

import pytest
import numpy as np
from insightspike.core.vector_integrator import VectorIntegrator


class TestVectorIntegrator:
    """Test the VectorIntegrator module."""
    
    def test_initialization(self):
        """Test VectorIntegrator initialization."""
        vi = VectorIntegrator()
        assert vi.configs is not None
        assert "insight" in vi.configs
        assert "episode_branching" in vi.configs
    
    def test_insight_vector_creation(self):
        """Test insight vector creation with query."""
        vi = VectorIntegrator()
        
        # Create test embeddings
        doc_embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        
        query_vector = np.array([1.0, 1.0, 0.0])
        
        # Create insight vector
        result = vi.create_insight_vector(doc_embeddings, query_vector)
        
        assert result is not None
        assert result.shape == (3,)
        # Should be normalized
        assert np.allclose(np.linalg.norm(result), 1.0)
    
    def test_branch_vector_creation(self):
        """Test episode branch vector creation."""
        vi = VectorIntegrator()
        
        # Parent and neighbor vectors
        parent_vector = np.array([1.0, 0.0, 0.0, 0.0])
        neighbor_vectors = [
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0])
        ]
        
        # Create branch vector
        result = vi.create_branch_vector(parent_vector, neighbor_vectors)
        
        assert result is not None
        assert result.shape == (4,)
        # Should have influence from parent (0.4 weight)
        assert result[0] > 0
        # Should be normalized
        assert np.allclose(np.linalg.norm(result), 1.0)
    
    def test_custom_weights(self):
        """Test integration with custom weights."""
        vi = VectorIntegrator()
        
        vectors = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0])
        ]
        
        # Equal weights should give average
        result1 = vi.integrate_vectors(
            vectors,
            custom_weights=[0.5, 0.5]
        )
        assert np.allclose(result1, np.array([0.707107, 0.707107]), atol=1e-5)
        
        # Heavily weighted to first vector
        result2 = vi.integrate_vectors(
            vectors,
            custom_weights=[0.9, 0.1]
        )
        assert result2[0] > result2[1]  # First component should dominate
    
    def test_different_aggregation_methods(self):
        """Test different aggregation methods."""
        vi = VectorIntegrator()
        
        vectors = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        
        # Mean aggregation
        result_mean = vi.integrate_vectors(
            vectors,
            config_overrides={"aggregation": "mean", "normalize": False}
        )
        assert np.allclose(result_mean, np.array([4.0, 5.0, 6.0]))
        
        # Max aggregation
        result_max = vi.integrate_vectors(
            vectors,
            config_overrides={"aggregation": "max", "normalize": False}
        )
        assert np.allclose(result_max, np.array([7.0, 8.0, 9.0]))
        
        # Sum aggregation
        result_sum = vi.integrate_vectors(
            vectors,
            config_overrides={"aggregation": "sum", "normalize": False}
        )
        assert np.allclose(result_sum, np.array([12.0, 15.0, 18.0]))
    
    def test_similarity_based_weights(self):
        """Test similarity-based weight calculation."""
        vi = VectorIntegrator()
        
        # Primary vector
        primary = np.array([1.0, 0.0, 0.0])
        
        # Secondary vectors with varying similarity
        vectors = [
            np.array([0.9, 0.1, 0.0]),  # High similarity
            np.array([0.1, 0.9, 0.0]),  # Low similarity
            np.array([0.5, 0.5, 0.0])   # Medium similarity
        ]
        
        result = vi.integrate_vectors(
            vectors,
            primary_vector=primary,
            integration_type="insight"
        )
        
        # Result should be influenced more by the similar vector
        assert result[0] > result[1]  # First component should dominate
    
    def test_empty_vectors_error(self):
        """Test error handling for empty vectors."""
        vi = VectorIntegrator()
        
        with pytest.raises(ValueError, match="No vectors provided"):
            vi.integrate_vectors([])
    
    def test_normalization(self):
        """Test vector normalization."""
        vi = VectorIntegrator()
        
        # Large magnitude vectors
        vectors = [
            np.array([10.0, 0.0, 0.0]),
            np.array([0.0, 10.0, 0.0])
        ]
        
        # With normalization (default)
        result_norm = vi.integrate_vectors(vectors)
        assert np.allclose(np.linalg.norm(result_norm), 1.0)
        
        # Without normalization
        result_no_norm = vi.integrate_vectors(
            vectors,
            config_overrides={"normalize": False}
        )
        assert np.linalg.norm(result_no_norm) > 1.0
    
    def test_integration_types(self):
        """Test predefined integration types."""
        vi = VectorIntegrator()
        
        vectors = [np.random.randn(5) for _ in range(3)]
        primary = np.random.randn(5)
        
        # Test each predefined type
        for integration_type in ["insight", "episode_branching", "message_passing", "context_merging"]:
            result = vi.integrate_vectors(
                vectors,
                primary_vector=primary,
                integration_type=integration_type
            )
            assert result is not None
            assert result.shape == (5,)