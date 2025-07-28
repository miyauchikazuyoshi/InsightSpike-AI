"""
Test Embedding Shape Normalization
==================================

Tests that embeddings are consistently shaped as (dim,) for single embeddings
and (batch_size, dim) for batches.
"""

import pytest
import numpy as np
import torch

from insightspike.processing.embedder import EmbeddingManager
from insightspike.utils.embedding_utils import (
    normalize_embedding_shape,
    normalize_batch_embeddings,
    validate_embedding_dimension
)
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.layers.layer2_compatibility import CompatibleL2MemoryManager
from insightspike.config import InsightSpikeConfig


class TestEmbeddingShapeNormalization:
    """Test embedding shape normalization across the system"""
    
    def test_normalize_embedding_shape_utils(self):
        """Test normalize_embedding_shape utility function"""
        # Test 1D array (already correct)
        emb_1d = np.random.randn(384)
        result = normalize_embedding_shape(emb_1d)
        assert result.shape == (384,)
        assert np.array_equal(result, emb_1d)
        
        # Test 2D array shape (1, 384)
        emb_2d_row = np.random.randn(1, 384)
        result = normalize_embedding_shape(emb_2d_row)
        assert result.shape == (384,)
        
        # Test 2D array shape (384, 1)
        emb_2d_col = np.random.randn(384, 1)
        result = normalize_embedding_shape(emb_2d_col)
        assert result.shape == (384,)
        
        # Test torch tensor
        emb_torch = torch.randn(1, 384)
        result = normalize_embedding_shape(emb_torch)
        assert result.shape == (384,)
        assert isinstance(result, np.ndarray)
        
        # Test list
        emb_list = np.random.randn(384).tolist()
        result = normalize_embedding_shape(emb_list)
        assert result.shape == (384,)
    
    def test_normalize_batch_embeddings_utils(self):
        """Test normalize_batch_embeddings utility function"""
        # Test 2D array (already correct)
        batch_2d = np.random.randn(5, 384)
        result = normalize_batch_embeddings(batch_2d)
        assert result.shape == (5, 384)
        
        # Test 1D array (single embedding)
        single_1d = np.random.randn(384)
        result = normalize_batch_embeddings(single_1d)
        assert result.shape == (1, 384)
        
        # Test 3D array shape (batch, 1, dim)
        batch_3d = np.random.randn(5, 1, 384)
        result = normalize_batch_embeddings(batch_3d)
        assert result.shape == (5, 384)
        
        # Test list of embeddings
        emb_list = [np.random.randn(384) for _ in range(3)]
        result = normalize_batch_embeddings(emb_list)
        assert result.shape == (3, 384)
        
        # Test list of 2D embeddings
        emb_list_2d = [np.random.randn(1, 384) for _ in range(3)]
        result = normalize_batch_embeddings(emb_list_2d)
        assert result.shape == (3, 384)
    
    def test_embedding_manager_single_text(self):
        """Test EmbeddingManager returns correct shape for single text"""
        manager = EmbeddingManager()
        
        # Single text should return shape (384,)
        embedding = manager.encode("Hello world")
        assert embedding.shape == (384,)
        assert embedding.ndim == 1
        
        # get_embedding should also return (384,)
        embedding2 = manager.get_embedding("Hello world")
        assert embedding2.shape == (384,)
        assert embedding2.ndim == 1
    
    def test_embedding_manager_batch_texts(self):
        """Test EmbeddingManager returns correct shape for batch"""
        manager = EmbeddingManager()
        
        # Batch of texts should return shape (batch_size, 384)
        texts = ["Hello world", "How are you?", "Test embedding"]
        embeddings = manager.encode(texts)
        assert embeddings.shape == (3, 384)
        assert embeddings.ndim == 2
    
    def test_validate_embedding_dimension(self):
        """Test embedding dimension validation"""
        # Valid 1D
        emb_1d = np.random.randn(384)
        assert validate_embedding_dimension(emb_1d, 384) is True
        assert validate_embedding_dimension(emb_1d, 768) is False
        
        # Valid 2D
        emb_2d = np.random.randn(5, 384)
        assert validate_embedding_dimension(emb_2d, 384) is True
        assert validate_embedding_dimension(emb_2d, 768) is False
        
        # Invalid shape
        emb_3d = np.random.randn(5, 1, 384)
        assert validate_embedding_dimension(emb_3d, 384) is False
    
    def test_main_agent_add_knowledge_shape(self):
        """Test MainAgent handles embedding shapes correctly"""
        config = InsightSpikeConfig()
        agent = MainAgent(config=config)
        
        # Add knowledge should handle embedding shape internally
        # No errors should occur
        agent.add_knowledge("Test knowledge item")
        
        # Verify stored episode has correct shape
        # This would require accessing internal storage
    
    def test_layer2_memory_manager_shape(self):
        """Test Layer2MemoryManager handles shapes correctly"""
        manager = CompatibleL2MemoryManager()
        
        # _encode_text should return shape (384,)
        embedding = manager._encode_text("Test text")
        assert embedding.shape == (384,)
        assert embedding.ndim == 1
    
    def test_edge_case_empty_batch(self):
        """Test handling of empty batches"""
        # Empty list
        result = normalize_batch_embeddings([])
        assert result.shape == (0,) or result.shape == (0, 0)  # Both are acceptable for empty
        
        # Empty numpy array
        empty_array = np.array([])
        result = normalize_batch_embeddings(empty_array)
        assert result.shape == (1, 0)  # Single empty embedding
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Invalid dimensions
        with pytest.raises(ValueError, match="Cannot normalize embedding with 3 dimensions"):
            normalize_embedding_shape(np.random.randn(2, 3, 384))
        
        # Invalid 2D shape
        with pytest.raises(ValueError, match="Cannot normalize 2D embedding"):
            normalize_embedding_shape(np.random.randn(5, 5))
        
        # Scalar
        with pytest.raises(ValueError, match="Embedding cannot be a scalar"):
            normalize_embedding_shape(np.array(5.0))
    
    def test_consistency_across_system(self):
        """Test that embeddings maintain consistent shapes across components"""
        config = InsightSpikeConfig()
        
        # Create components
        embedder = EmbeddingManager()
        memory_manager = CompatibleL2MemoryManager()
        
        # Test text
        text = "Test consistency across system"
        
        # All should produce same shape
        emb1 = embedder.encode(text)
        emb2 = embedder.get_embedding(text)
        emb3 = memory_manager._encode_text(text)
        
        assert emb1.shape == (384,)
        assert emb2.shape == (384,)
        assert emb3.shape == (384,)
        
        # Batch processing
        texts = [text, "Another text", "Third text"]
        batch_emb = embedder.encode(texts)
        assert batch_emb.shape == (3, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])