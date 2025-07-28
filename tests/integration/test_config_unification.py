"""
Test Configuration System Unification
====================================

Tests that the configuration system properly handles Pydantic-only configs
and provides appropriate warnings for legacy formats.
"""

import pytest
import warnings
from types import SimpleNamespace

from insightspike.config import InsightSpikeConfig, load_config
from insightspike.config.legacy_adapter import LegacyConfigAdapter
from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
from insightspike.implementations.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.implementations.graph.pyg_graph_builder import PyGGraphBuilder


class TestConfigUnification:
    """Test configuration system unification to Pydantic-only"""
    
    def test_legacy_adapter_dict_config(self):
        """Test LegacyConfigAdapter handles dict configs with deprecation warning"""
        dict_config = {
            "graph": {
                "spike_ged_threshold": -0.3,
                "similarity_threshold": 0.5
            },
            "embedding": {
                "dimension": 768
            }
        }
        
        with pytest.warns(DeprecationWarning, match="Dict-based configs are deprecated"):
            pydantic_config = LegacyConfigAdapter.ensure_pydantic(dict_config)
        
        assert isinstance(pydantic_config, InsightSpikeConfig)
        assert pydantic_config.graph.spike_ged_threshold == -0.3
        assert pydantic_config.graph.similarity_threshold == 0.5
        assert pydantic_config.embedding.dimension == 768
    
    def test_legacy_adapter_namespace_config(self):
        """Test LegacyConfigAdapter handles SimpleNamespace configs"""
        namespace_config = SimpleNamespace(
            graph=SimpleNamespace(
                spike_ged_threshold=-0.4,
                similarity_threshold=0.6
            ),
            embedding=SimpleNamespace(
                dimension=512
            )
        )
        
        with pytest.warns(DeprecationWarning, match="SimpleNamespace configs are deprecated"):
            pydantic_config = LegacyConfigAdapter.ensure_pydantic(namespace_config)
        
        assert isinstance(pydantic_config, InsightSpikeConfig)
        assert pydantic_config.graph.spike_ged_threshold == -0.4
        assert pydantic_config.graph.similarity_threshold == 0.6
        assert pydantic_config.embedding.dimension == 512
    
    def test_legacy_adapter_pydantic_passthrough(self):
        """Test LegacyConfigAdapter passes through Pydantic configs unchanged"""
        pydantic_config = InsightSpikeConfig()
        
        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = LegacyConfigAdapter.ensure_pydantic(pydantic_config)
        
        assert result is pydantic_config  # Same object
    
    def test_legacy_adapter_get_value(self):
        """Test LegacyConfigAdapter.get_value with different config types"""
        # Dict config
        dict_config = {"graph": {"similarity_threshold": 0.7}}
        assert LegacyConfigAdapter.get_value(dict_config, "graph.similarity_threshold") == 0.7
        assert LegacyConfigAdapter.get_value(dict_config, "graph.missing", "default") == "default"
        
        # Pydantic config
        pydantic_config = InsightSpikeConfig()
        assert LegacyConfigAdapter.get_value(
            pydantic_config, "graph.similarity_threshold"
        ) == pydantic_config.graph.similarity_threshold
        
        # SimpleNamespace
        ns_config = SimpleNamespace(graph=SimpleNamespace(similarity_threshold=0.8))
        assert LegacyConfigAdapter.get_value(ns_config, "graph.similarity_threshold") == 0.8
    
    def test_layer3_with_unified_config(self):
        """Test L3GraphReasoner works with unified config"""
        # Test with Pydantic config (no warning)
        pydantic_config = InsightSpikeConfig()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reasoner = L3GraphReasoner(pydantic_config)
        
        assert reasoner.config.graph.conflict_threshold == 0.5  # Default value
        
        # Test with dict config (should warn but work)
        dict_config = {"graph": {"conflict_threshold": 0.7}}
        with pytest.warns(DeprecationWarning):
            reasoner_dict = L3GraphReasoner(dict_config)
        
        assert reasoner_dict.config.graph.conflict_threshold == 0.7
    
    def test_scalable_graph_builder_unified(self):
        """Test ScalableGraphBuilder with unified config"""
        # Test with Pydantic config
        pydantic_config = InsightSpikeConfig()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder = ScalableGraphBuilder(pydantic_config)
        
        assert builder.similarity_threshold == pydantic_config.graph.similarity_threshold
        assert builder.dimension == pydantic_config.embedding.dimension
        
        # Test with no config (should use defaults)
        builder_default = ScalableGraphBuilder(None)
        assert isinstance(builder_default.config, InsightSpikeConfig)
    
    def test_pyg_graph_builder_unified(self):
        """Test PyGGraphBuilder with unified config"""
        # Test with Pydantic config
        pydantic_config = InsightSpikeConfig()
        pydantic_config.graph.similarity_threshold = 0.8
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder = PyGGraphBuilder(pydantic_config)
        
        assert builder.similarity_threshold == 0.8
        
        # Test with dict config (should warn)
        dict_config = {"graph": {"similarity_threshold": 0.9}}
        with pytest.warns(DeprecationWarning):
            builder_dict = PyGGraphBuilder(dict_config)
        
        assert builder_dict.similarity_threshold == 0.9
    
    def test_config_loading_presets(self):
        """Test loading configs with presets returns Pydantic objects"""
        # Load experiment preset
        config = load_config(preset="experiment")
        assert isinstance(config, InsightSpikeConfig)
        assert config.environment == "experiment"
        
        # Load with overrides
        config_custom = load_config(
            preset="production",
            overrides={"graph": {"similarity_threshold": 0.85}}
        )
        assert isinstance(config_custom, InsightSpikeConfig)
        assert config_custom.environment == "production"
        assert config_custom.graph.similarity_threshold == 0.85
    
    def test_components_reject_invalid_configs(self):
        """Test components properly reject invalid config types"""
        invalid_config = "not a config"
        
        with pytest.raises(ValueError, match="Unsupported config type"):
            LegacyConfigAdapter.ensure_pydantic(invalid_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])