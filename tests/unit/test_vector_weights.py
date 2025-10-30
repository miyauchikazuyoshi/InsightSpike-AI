"""
Unit tests for vector weight functionality.
"""

import numpy as np
import pytest
from insightspike.config.vector_weights import VectorWeightConfig
from insightspike.core.weight_vector_manager import WeightVectorManager


class TestVectorWeightConfig:
    """Test vector weight configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = VectorWeightConfig()
        
        assert config.enabled == False
        assert config.weights is None
        assert config.active_preset is None
        assert isinstance(config.presets, dict)
    
    def test_config_with_weights(self):
        """Test configuration with direct weights."""
        weights = [1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3]
        config = VectorWeightConfig(
            enabled=True,
            weights=weights
        )
        
        assert config.enabled == True
        assert config.weights == weights
    
    def test_config_with_preset(self):
        """Test configuration with preset."""
        config = VectorWeightConfig(
            enabled=True,
            active_preset="maze_8d",
            presets={
                "maze_8d": [1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3]
            }
        )
        
        assert config.active_preset == "maze_8d"
        assert "maze_8d" in config.presets


class TestWeightVectorManager:
    """Test weight vector manager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = WeightVectorManager()
        
        assert manager.config.enabled == False
        assert not manager.is_enabled()
    
    def test_weight_application(self):
        """Test weight application to vector."""
        config = VectorWeightConfig(
            enabled=True,
            weights=[1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3]
        )
        manager = WeightVectorManager(config)
        
        # 8-dim vector
        vector = np.array([0.5, 0.5, 1.0, 1.0, 0.3, 0.3, 0.3, 0.8])
        weighted = manager.apply_weights(vector)
        
        # Check weights were applied correctly
        assert weighted[0] == 0.5 * 1.0  # position unchanged
        assert weighted[1] == 0.5 * 1.0  # position unchanged
        assert weighted[2] == 1.0 * 0.1  # direction scaled down
        assert weighted[3] == 1.0 * 0.1  # direction scaled down
        assert weighted[4] == 0.3 * 0.5  # result scaled
        assert weighted[7] == 0.8 * 0.3  # goal_dist scaled
    
    def test_dimension_mismatch(self):
        """Test dimension mismatch handling."""
        config = VectorWeightConfig(
            enabled=True,
            weights=[1.0, 0.5]  # 2-dim weights
        )
        manager = WeightVectorManager(config)
        
        # 8-dim vector
        vector = np.array([0.5, 0.5, 1.0, 1.0, 0.3, 0.3, 0.3, 0.8])
        weighted = manager.apply_weights(vector)
        
        # Should return original vector on dimension mismatch
        np.testing.assert_array_equal(weighted, vector)
    
    def test_disabled_feature(self):
        """Test that weights are not applied when disabled."""
        config = VectorWeightConfig(enabled=False)
        manager = WeightVectorManager(config)
        
        vector = np.array([0.5, 0.5, 1.0, 1.0])
        weighted = manager.apply_weights(vector)
        
        # Should return original vector when disabled
        np.testing.assert_array_equal(weighted, vector)
    
    def test_batch_processing(self):
        """Test batch processing of vectors."""
        config = VectorWeightConfig(
            enabled=True,
            weights=[1.0, 0.1, 0.5, 0.2]
        )
        manager = WeightVectorManager(config)
        
        # Batch of 4-dim vectors
        batch = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        ])
        
        weighted_batch = manager.apply_to_batch(batch)
        
        assert weighted_batch.shape == batch.shape
        assert weighted_batch[0, 1] == 2.0 * 0.1  # Second dim scaled
        assert weighted_batch[1, 2] == 7.0 * 0.5  # Third dim scaled
    
    def test_preset_switching(self):
        """Test switching between presets."""
        config = VectorWeightConfig(
            enabled=True,
            presets={
                "preset1": [1.0, 0.5],
                "preset2": [2.0, 0.1]
            }
        )
        manager = WeightVectorManager(config)
        
        # Switch to preset1
        manager.switch_preset("preset1")
        assert manager.config.active_preset == "preset1"
        assert manager.get_weights() == [1.0, 0.5]
        
        # Switch to preset2
        manager.switch_preset("preset2")
        assert manager.config.active_preset == "preset2"
        assert manager.get_weights() == [2.0, 0.1]
        
        # Invalid preset should raise error
        with pytest.raises(ValueError):
            manager.switch_preset("invalid_preset")
    
    def test_set_weights_directly(self):
        """Test setting weights directly."""
        manager = WeightVectorManager()
        
        weights = [1.0, 0.5, 0.2]
        manager.set_weights(weights)
        
        assert manager.get_weights() == weights
        assert manager.config.weights == weights
        assert manager.config.active_preset is None  # Should clear preset
    
    def test_default_presets(self):
        """Test that default presets are available."""
        manager = WeightVectorManager()
        
        assert "maze_8d" in manager.config.presets
        assert "maze_aggressive" in manager.config.presets
        assert "language_384d" in manager.config.presets


class TestIntegration:
    """Test integration with other components."""
    
    def test_with_vector_integrator(self):
        """Test integration with VectorIntegrator."""
        from insightspike.core.vector_integrator import VectorIntegrator
        
        config = VectorWeightConfig(
            enabled=True,
            weights=[1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3]
        )
        manager = WeightVectorManager(config)
        integrator = VectorIntegrator(weight_manager=manager)
        
        # Test vector integration with weights
        vectors = [
            np.array([0.5, 0.5, 1.0, 0.0, 0.3, 0.3, 0.3, 0.8]),
            np.array([0.4, 0.4, 0.0, 1.0, 0.4, 0.4, 0.4, 0.7])
        ]
        primary = np.array([0.45, 0.45, 0.5, 0.5, 0.35, 0.35, 0.35, 0.75])
        
        result = integrator.integrate_vectors(
            vectors,
            primary_vector=primary,
            integration_type="insight"
        )
        
        # Result should be normalized
        assert np.abs(np.linalg.norm(result) - 1.0) < 0.01
    
    def test_with_embedder(self):
        """Test integration with EmbeddingManager."""
        from insightspike.processing.embedder import EmbeddingManager
        
        config = VectorWeightConfig(
            enabled=True,
            weights=[1.0, 0.5]  # Small weight vector for test
        )
        manager = WeightVectorManager(config)
        
        # Create embedder with weight manager
        embedder = EmbeddingManager(weight_manager=manager)
        
        # Test embedding generation (weights won't apply due to dimension mismatch)
        text = "test text"
        embedding = embedder.get_embedding(text)
        
        assert embedding is not None
        assert len(embedding.shape) == 1  # Should be 1D