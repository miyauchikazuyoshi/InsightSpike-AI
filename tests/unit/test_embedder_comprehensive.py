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
        
        # Pass config directly
        manager = EmbeddingManager(config=mock_config)
        assert manager.model_name == "test-model"
        assert manager.dimension == 512
    
    def test_init_without_config(self):
        """Test initialization without config (fallback)."""
        # Create manager without config - will use defaults
        manager = EmbeddingManager()
        # Either uses config defaults or fallback
        assert hasattr(manager, 'model_name')
        assert hasattr(manager, 'dimension')
        assert manager.dimension > 0
    
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
        assert model2 == mock_model
    
    def test_safe_mode(self):
        """Test safe mode with environment variable."""
        # Clear cache and set safe mode
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        with patch.dict(os.environ, {'INSIGHTSPIKE_SAFE_MODE': '1'}):
            manager = EmbeddingManager()
            
            with patch.object(manager, '_fallback_model') as mock_fallback:
                mock_fallback_model = Mock()
                mock_fallback.return_value = mock_fallback_model
                
                model = manager.get_model()
                assert mock_fallback.called
                assert model == mock_fallback_model
    
    @patch('insightspike.utils.embedder.SentenceTransformer')
    @patch('torch.set_num_threads')
    def test_model_loading_with_env_setup(self, mock_set_threads, mock_st):
        """Test model loading sets up environment correctly."""
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        manager = EmbeddingManager()
        model = manager.get_model()
        
        # Check environment was configured
        assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') == '0.0'
        assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false'
        mock_set_threads.assert_called_once_with(1)
        
        # Check model was created with correct params
        mock_st.assert_called_once_with(
            manager.model_name,
            device="cpu",
            cache_folder=None,
            trust_remote_code=False
        )
    
    def test_model_loading_fallback_on_error(self):
        """Test fallback when model loading fails."""
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        with patch('insightspike.utils.embedder.SentenceTransformer', side_effect=Exception("Load failed")):
            manager = EmbeddingManager()
            
            with patch.object(manager, '_fallback_model') as mock_fallback:
                mock_fallback_model = Mock()
                mock_fallback.return_value = mock_fallback_model
                
                model = manager.get_model()
                assert mock_fallback.called
                assert model == mock_fallback_model
    
    @patch('insightspike.utils.embedder.SentenceTransformer')
    def test_encode_single_text(self, mock_st):
        """Test encoding single text."""
        mock_model = Mock()
        mock_embeddings = np.random.rand(1, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        manager = EmbeddingManager()
        
        result = manager.encode("Test text")
        
        mock_model.encode.assert_called_once_with(
            ["Test text"],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        assert np.array_equal(result, mock_embeddings)
    
    @patch('insightspike.utils.embedder.SentenceTransformer')
    def test_encode_multiple_texts(self, mock_st):
        """Test encoding multiple texts."""
        mock_model = Mock()
        mock_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        manager = EmbeddingManager()
        
        texts = ["Text 1", "Text 2", "Text 3"]
        result = manager.encode(texts, batch_size=64, show_progress_bar=True)
        
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        assert np.array_equal(result, mock_embeddings)
    
    def test_encode_fallback(self):
        """Test encoding with fallback model."""
        manager = EmbeddingManager()
        manager._model = None
        
        with patch.object(manager, 'get_model') as mock_get_model:
            mock_fallback = Mock()
            mock_get_model.return_value = mock_fallback
            
            with patch.object(manager, '_fallback_encode') as mock_fallback_encode:
                expected_embeddings = np.random.rand(1, 384)
                mock_fallback_encode.return_value = expected_embeddings
                
                # When model.encode doesn't exist
                mock_fallback.encode = None
                
                result = manager.encode("Test")
                mock_fallback_encode.assert_called_once()
                assert np.array_equal(result, expected_embeddings)
    
    def test_fallback_model(self):
        """Test fallback model creation."""
        manager = EmbeddingManager()
        
        fallback = manager._fallback_model()
        
        assert hasattr(fallback, 'encode')
        assert hasattr(fallback, 'get_sentence_embedding_dimension')
        assert fallback.get_sentence_embedding_dimension() == manager.dimension
    
    def test_fallback_encode(self):
        """Test fallback encoding method."""
        manager = EmbeddingManager()
        
        # Single text
        result1 = manager._fallback_encode(["Test text"])
        assert result1.shape == (1, manager.dimension)
        assert result1.dtype == np.float32
        
        # Multiple texts
        result2 = manager._fallback_encode(["Text 1", "Text 2", "Text 3"])
        assert result2.shape == (3, manager.dimension)
        
        # Check normalization
        norms = np.linalg.norm(result2, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestGetModelFunction:
    """Test the global get_model function."""
    
    def test_get_model_returns_global_manager(self):
        """Test get_model returns the global manager's model."""
        import insightspike.utils.embedder as embedder_module
        
        # Reset global state
        embedder_module._global_manager = None
        embedder_module._model_cache.clear()
        
        with patch('insightspike.utils.embedder.EmbeddingManager') as mock_manager_class:
            mock_manager = Mock()
            mock_model = Mock()
            mock_manager.get_model.return_value = mock_model
            mock_manager_class.return_value = mock_manager
            
            result = get_model()
            
            assert result == mock_model
            assert embedder_module._global_manager == mock_manager
    
    def test_get_model_reuses_global_manager(self):
        """Test get_model reuses existing global manager."""
        import insightspike.utils.embedder as embedder_module
        
        # Set up existing manager
        mock_manager = Mock()
        mock_model = Mock()
        mock_manager.get_model.return_value = mock_model
        embedder_module._global_manager = mock_manager
        
        with patch('insightspike.utils.embedder.EmbeddingManager') as mock_manager_class:
            result = get_model()
            
            # Should not create new manager
            assert not mock_manager_class.called
            assert result == mock_model
    
    def test_get_model_with_custom_model_name(self):
        """Test get_model with custom model name creates new manager."""
        import insightspike.utils.embedder as embedder_module
        
        # Reset global state
        embedder_module._global_manager = None
        
        with patch('insightspike.utils.embedder.EmbeddingManager') as mock_manager_class:
            mock_manager = Mock()
            mock_model = Mock()
            mock_manager.get_model.return_value = mock_model
            mock_manager_class.return_value = mock_manager
            
            result = get_model(model_name="custom-model")
            
            mock_manager_class.assert_called_once_with(model_name="custom-model")
            assert result == mock_model


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text_encoding(self):
        """Test encoding empty text."""
        manager = EmbeddingManager()
        
        # Use fallback for predictable behavior
        result = manager._fallback_encode([""])
        assert result.shape == (1, manager.dimension)
        assert not np.any(np.isnan(result))
    
    def test_very_long_text_encoding(self):
        """Test encoding very long text."""
        manager = EmbeddingManager()
        
        long_text = "word " * 10000  # Very long text
        result = manager._fallback_encode([long_text])
        assert result.shape == (1, manager.dimension)
        assert not np.any(np.isnan(result))
    
    def test_unicode_text_encoding(self):
        """Test encoding unicode text."""
        manager = EmbeddingManager()
        
        unicode_texts = ["Hello 世界", "Привет мир", "مرحبا بالعالم"]
        result = manager._fallback_encode(unicode_texts)
        assert result.shape == (3, manager.dimension)
        assert not np.any(np.isnan(result))
    
    @patch('insightspike.utils.embedder.SentenceTransformer', side_effect=ImportError)
    def test_sentence_transformers_not_installed(self, mock_st):
        """Test behavior when sentence-transformers is not installed."""
        import insightspike.utils.embedder as embedder_module
        embedder_module._model_cache.clear()
        
        manager = EmbeddingManager()
        
        with patch.object(manager, '_fallback_model') as mock_fallback:
            mock_fallback_model = Mock()
            mock_fallback.return_value = mock_fallback_model
            
            model = manager.get_model()
            assert mock_fallback.called
            assert model == mock_fallback_model