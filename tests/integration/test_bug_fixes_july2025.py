#!/usr/bin/env python3
"""
Bug Fixes Verification Tests - July 2025
========================================

Tests to verify all bug fixes implemented in July 2025.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from insightspike.config import load_config
from insightspike.core.episode import Episode
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.implementations.layers.cached_memory_manager import CachedMemoryManager
from insightspike.utils.config_utils import safe_get
from insightspike.monitoring.memory_monitor import MemoryMonitor


class TestConfigUtils:
    """Test safe config access utility"""
    
    def test_safe_get_dict_config(self):
        """Test safe_get with dict config"""
        config = {
            "processing": {
                "enable_learning": True,
                "nested": {
                    "value": 42
                }
            }
        }
        
        assert safe_get(config, "processing", "enable_learning") == True
        assert safe_get(config, "processing", "nested", "value") == 42
        assert safe_get(config, "processing", "missing", default="default") == "default"
        
    def test_safe_get_object_config(self):
        """Test safe_get with object config"""
        class MockConfig:
            def __init__(self):
                self.processing = type('obj', (object,), {
                    'enable_learning': True,
                    'nested': type('obj', (object,), {'value': 42})
                })
        
        config = MockConfig()
        assert safe_get(config, "processing", "enable_learning") == True
        assert safe_get(config, "processing", "nested", "value") == 42
        assert safe_get(config, "processing", "missing", default="default") == "default"


class TestEpisodeConfidenceField:
    """Test Episode confidence field standardization"""
    
    def test_confidence_field(self):
        """Test confidence field and aliases"""
        episode = Episode(
            text="Test",
            vec=np.zeros(384),
            confidence=0.8
        )
        
        # Primary field
        assert episode.confidence == 0.8
        
        # Backward compatibility aliases
        assert episode.c == 0.8
        assert episode.c_value == 0.8
        
    def test_episode_serialization(self):
        """Test episode serialization includes confidence"""
        episode = Episode(
            text="Test",
            vec=np.zeros(384),
            confidence=0.9
        )
        
        # Convert to dict
        ep_dict = episode.__dict__.copy()
        assert "confidence" in ep_dict
        assert ep_dict["confidence"] == 0.9


class TestCachedMemoryManager:
    """Test CachedMemoryManager fixes"""
    
    def test_get_all_episodes(self):
        """Test get_all_episodes returns all episodes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            datastore = DataStoreFactory.create("filesystem", base_path=tmpdir)
            manager = CachedMemoryManager(datastore, cache_size=2)
            
            # Add 5 episodes
            for i in range(5):
                manager.add_episode(f"Episode {i}", c_value=0.5)
                
            # Get all episodes
            all_episodes = manager.get_all_episodes()
            assert len(all_episodes) == 5
            
            # Get cached episodes (should be limited by cache size)
            cached_episodes = manager.get_cached_episodes()
            assert len(cached_episodes) <= 2
            
    def test_episodes_property(self):
        """Test episodes property returns all episodes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            datastore = DataStoreFactory.create("filesystem", base_path=tmpdir)
            manager = CachedMemoryManager(datastore, cache_size=2)
            
            # Add episodes
            for i in range(3):
                manager.add_episode(f"Episode {i}", c_value=0.5)
                
            # Property should return all episodes
            assert len(manager.episodes) == 3


class TestMemoryThresholds:
    """Test memory threshold adjustments"""
    
    def test_memory_thresholds(self):
        """Test updated memory thresholds"""
        monitor = MemoryMonitor()
        
        # Check thresholds
        assert monitor.warning_threshold_mb == 2048  # 2GB
        assert monitor.critical_threshold_mb == 4096  # 4GB


class TestDataStoreAppendBehavior:
    """Test DataStore append behavior fixes"""
    
    def test_filesystem_store_appends(self):
        """Test FileSystemDataStore appends instead of overwrites"""
        with tempfile.TemporaryDirectory() as tmpdir:
            datastore = DataStoreFactory.create("filesystem", base_path=tmpdir)
            
            # Save first batch
            episodes1 = [
                {"id": "1", "text": "Episode 1", "vec": [0.1] * 384, "confidence": 0.5}
            ]
            datastore.save_episodes(episodes1)
            
            # Save second batch
            episodes2 = [
                {"id": "2", "text": "Episode 2", "vec": [0.2] * 384, "confidence": 0.6}
            ]
            datastore.save_episodes(episodes2)
            
            # Load all episodes
            all_episodes = datastore.load_episodes()
            assert len(all_episodes) == 2
            assert any(ep["text"] == "Episode 1" for ep in all_episodes)
            assert any(ep["text"] == "Episode 2" for ep in all_episodes)


class TestMainAgentInitialization:
    """Test MainAgent initialization without warnings"""
    
    def test_agent_initializes_without_warnings(self, caplog):
        """Test agent initializes without excessive warnings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config = {
                "processing": {"enable_learning": False},
                "memory": {"max_retrieved_docs": 5},
                "l4_config": {"provider": "mock"},
                "graph": {"similarity_threshold": 0.7}
            }
            
            # Create datastore
            datastore = DataStoreFactory.create("filesystem", base_path=tmpdir)
            
            # Create agent
            with caplog.at_level("WARNING"):
                agent = MainAgent(config=config, datastore=datastore)
                agent.initialize()
                
            # Check no excessive warnings
            warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
            
            # These warnings should NOT appear
            unwanted_warnings = [
                "Memory usage warning",
                "Reduced cache size",
                "only cached episodes",
                "Advanced metrics not available",
                "Layer2 not using ScalableGraphManager"
            ]
            
            for warning in unwanted_warnings:
                assert not any(warning in msg for msg in warning_messages), \
                    f"Unwanted warning found: {warning}"


@pytest.mark.parametrize("config_format", ["dict", "object"])
def test_config_compatibility(config_format):
    """Test both dict and object config formats work"""
    with tempfile.TemporaryDirectory() as tmpdir:
        if config_format == "dict":
            config = {
                "processing": {"enable_learning": False},
                "memory": {"max_retrieved_docs": 5},
                "l4_config": {"provider": "mock"},
                "graph": {"similarity_threshold": 0.7}
            }
        else:
            # Create object-style config
            config = type('Config', (), {
                'processing': type('obj', (), {'enable_learning': False}),
                'memory': type('obj', (), {'max_retrieved_docs': 5}),
                'l4_config': type('obj', (), {'provider': 'mock'}),
                'graph': type('obj', (), {'similarity_threshold': 0.7})
            })()
            
        datastore = DataStoreFactory.create("filesystem", base_path=tmpdir)
        agent = MainAgent(config=config, datastore=datastore)
        
        assert agent.initialize()
        
        # Test basic operation
        agent.add_knowledge("Test knowledge")
        result = agent.process_question("Test?")
        
        assert hasattr(result, 'response') or isinstance(result, dict)