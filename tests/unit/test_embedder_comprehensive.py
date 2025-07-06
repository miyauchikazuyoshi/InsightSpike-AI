"""
Comprehensive unit tests for embedder module
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from insightspike.utils.embedder import get_model, EmbeddingManager


class TestEmbeddingManager:
    """Test EmbeddingManager class functionality."""
    
    def test_init_with_config(self):
        """Test initialization with config object."""
        mock_config = Mock()
        mock_config.embedding.model_name = "test-model"
        mock_config.embedding.dimension = 512
        
        manager = EmbeddingManager(config=mock_config)
        assert manager.model_name == "test-model"
        assert manager.dimension == 512
    
    def test_init_without_config(self):
        """Test initialization without config (fallback)."""
        with patch('insightspike.utils.embedder.get_config', side_effect=ImportError):
            manager = EmbeddingManager()
            assert manager.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert manager.dimension == 384
    
    def test_init_with_model_name(self):
        """Test initialization with explicit model name."""
        manager = EmbeddingManager(model_name="custom-model")
        assert manager.model_name == "custom-model"
    
    @patch('insightspike.utils.embedder.SentenceTransformer')
    def test_get_model_caching(self, mock_st):
        """Test model caching mechanism."""
        # Clear cache
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        manager = EmbeddingManager(model_name="test-model")
        
        # First call should create model
        model1 = manager.get_model()
        assert mock_st.called
        assert model1 == mock_model
        
        # Second call should use cache
        mock_st.reset_mock()
        model2 = manager.get_model()
        assert not mock_st.called
        assert model2 == model1
    
    @patch('insightspike.utils.embedder.SentenceTransformer')
    def test_get_model_error_handling(self, mock_st):
        """Test error handling in model loading."""
        mock_st.side_effect = Exception("Model load error")
        
        manager = EmbeddingManager(model_name="bad-model")
        
        # Should fall back to mock model
        model = manager.get_model()
        assert hasattr(model, 'encode')
        
        # Test the mock model works
        result = model.encode("test text")
        assert isinstance(result, np.ndarray)
        assert result.shape == (manager.dimension,)
    
    def test_encode_single_text(self):
        """Test encoding single text."""
        manager = EmbeddingManager()
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(1, 384)
        manager._model = mock_model
        
        result = manager.encode("test text")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        mock_model.encode.assert_called_once()
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        manager = EmbeddingManager()
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(3, 384)
        manager._model = mock_model
        
        texts = ["text1", "text2", "text3"]
        result = manager.encode(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)
    
    def test_encode_with_normalization(self):
        """Test encoding with normalization."""
        manager = EmbeddingManager()
        mock_model = Mock()
        # Return non-normalized vectors
        embeddings = np.array([[3.0, 4.0], [6.0, 8.0]])
        mock_model.encode.return_value = embeddings
        manager._model = mock_model
        manager.dimension = 2
        
        result = manager.encode(["text1", "text2"], normalize_embeddings=True)
        
        # Check normalization
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-6)
    
    def test_encode_empty_text(self):
        """Test encoding empty text."""
        manager = EmbeddingManager()
        result = manager.encode("")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
    
    def test_encode_none_text(self):
        """Test encoding None text."""
        manager = EmbeddingManager()
        result = manager.encode(None)
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)


class TestGetModelFunction:
    """Test the global get_model function."""
    
    def test_get_model_singleton(self):
        """Test get_model returns singleton."""
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2
    
    @patch.dict(os.environ, {'INSIGHTSPIKE_EMBEDDING_MODEL': 'custom-model'})
    def test_get_model_with_env_var(self):
        """Test get_model respects environment variable."""
        # Clear global manager
        import insightspike.utils.embedder as embedder_module
        embedder_module._global_manager = None
        
        with patch('insightspike.utils.embedder.EmbeddingManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            get_model()
            
            # Check that EmbeddingManager was called with env var model
            mock_manager_class.assert_called_once_with(model_name='custom-model')
    
    def test_get_model_encode_interface(self):
        """Test the model returned by get_model has encode interface."""
        model = get_model()
        assert hasattr(model, 'encode')
        
        # Test encode works
        result = model.encode("test")
        assert isinstance(result, np.ndarray)
        
        # Test batch encode
        results = model.encode(["test1", "test2"])
        assert isinstance(results, np.ndarray)
        assert len(results) == 2


class TestMockModel:
    """Test the mock model functionality."""
    
    def test_mock_model_encode_consistency(self):
        """Test mock model returns consistent embeddings."""
        manager = EmbeddingManager()
        # Force mock model
        manager._model = None
        with patch('insightspike.utils.embedder.SentenceTransformer', side_effect=Exception):
            model = manager.get_model()
            
            # Same text should give same embedding
            text = "test text"
            emb1 = model.encode(text)
            emb2 = model.encode(text)
            np.testing.assert_array_equal(emb1, emb2)
    
    def test_mock_model_batch_processing(self):
        """Test mock model handles batch processing."""
        manager = EmbeddingManager()
        with patch('insightspike.utils.embedder.SentenceTransformer', side_effect=Exception):
            model = manager.get_model()
            
            texts = ["text1", "text2", "text3"]
            embeddings = model.encode(texts, batch_size=2)
            
            assert embeddings.shape == (3, 384)
            # Each embedding should be different
            assert not np.array_equal(embeddings[0], embeddings[1])
            assert not np.array_equal(embeddings[1], embeddings[2])
    
    def test_mock_model_kwargs_handling(self):
        """Test mock model handles various kwargs."""
        manager = EmbeddingManager()
        with patch('insightspike.utils.embedder.SentenceTransformer', side_effect=Exception):
            model = manager.get_model()
            
            # Should handle various kwargs without error
            result = model.encode(
                "test",
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            assert isinstance(result, np.ndarray)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_batch_encoding(self):
        """Test encoding large batch of texts."""
        manager = EmbeddingManager()
        texts = ["text" + str(i) for i in range(1000)]
        
        result = manager.encode(texts)
        assert result.shape == (1000, 384)
    
    def test_mixed_input_types(self):
        """Test encoding with mixed input types."""
        manager = EmbeddingManager()
        
        # Should handle string
        result1 = manager.encode("text")
        assert result1.shape == (384,)
        
        # Should handle list
        result2 = manager.encode(["text"])
        assert result2.shape == (1, 384)
        
        # Should handle empty list
        result3 = manager.encode([])
        assert result3.shape == (0, 384)
    
    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches."""
        manager = EmbeddingManager()
        manager.dimension = 512
        
        # Mock model returns wrong dimension
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(1, 384)
        manager._model = mock_model
        
        # Should still work but log warning
        with patch('insightspike.utils.embedder.logger') as mock_logger:
            result = manager.encode("test")
            assert result.shape == (384,)  # Uses actual dimension
    
    @patch('insightspike.utils.embedder._model_cache', {})
    def test_cache_clearing(self):
        """Test model cache can be cleared."""
        # Add model to cache
        manager1 = EmbeddingManager(model_name="model1")
        model1 = manager1.get_model()
        
        # Clear cache
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        # Should create new model
        manager2 = EmbeddingManager(model_name="model1")
        with patch('insightspike.utils.embedder.SentenceTransformer') as mock_st:
            mock_st.return_value = Mock()
            model2 = manager2.get_model()
            assert mock_st.called