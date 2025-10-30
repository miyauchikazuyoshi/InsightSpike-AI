"""Tests for Pydantic-based configuration system."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from insightspike.config.loader import ConfigLoader
from insightspike.config.models import (
    EmbeddingConfig,
    GraphConfig,
    InsightSpikeConfig,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    MonitoringConfig,
)
from insightspike.config.presets import ConfigPresets


class TestInsightSpikeConfig:
    """Test InsightSpikeConfig model validation and behavior."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InsightSpikeConfig()

        assert config.environment == "development"
        assert config.llm.provider == "local"
        assert config.llm.model == "distilgpt2"
        assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.memory.episodic_memory_capacity == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        config = InsightSpikeConfig(
            environment="production",
            llm=LLMConfig(
                provider="openai", model="gpt-4", temperature=0.5, max_tokens=1024
            ),
        )

        assert config.environment == "production"
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4"
        assert config.llm.temperature == 0.5
        assert config.llm.max_tokens == 1024

    def test_validation_errors(self):
        """Test validation errors for invalid values."""
        # Invalid LLM provider
        with pytest.raises(ValidationError):
            InsightSpikeConfig(llm=LLMConfig(provider="invalid_provider"))

        # Temperature out of range
        with pytest.raises(ValidationError):
            InsightSpikeConfig(llm=LLMConfig(temperature=3.0))  # Max is 2.0

        # Negative max_tokens
        with pytest.raises(ValidationError):
            InsightSpikeConfig(llm=LLMConfig(max_tokens=-1))

    def test_serialization(self):
        """Test config serialization to dict and JSON."""
        config = InsightSpikeConfig()

        # To dict
        config_dict = config.dict()
        assert isinstance(config_dict, dict)
        assert config_dict["environment"] == "development"
        assert "llm" in config_dict
        assert "embedding" in config_dict

        # To JSON
        config_json = config.json()
        assert isinstance(config_json, str)
        parsed = json.loads(config_json)
        assert parsed["environment"] == "development"

    def test_deserialization(self):
        """Test config deserialization from dict and JSON."""
        config_dict = {
            "environment": "experiment",
            "llm": {"provider": "anthropic", "model": "claude-2", "temperature": 0.7},
        }

        # From dict
        config = InsightSpikeConfig(**config_dict)
        assert config.environment == "experiment"
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-2"

        # From JSON
        config_json = json.dumps(config_dict)
        config2 = InsightSpikeConfig.parse_raw(config_json)
        assert config2.environment == "experiment"
        assert config2.llm.provider == "anthropic"


class TestConfigPresets:
    """Test configuration presets."""

    def test_development_preset(self):
        """Test development preset configuration."""
        config = ConfigPresets.development()

        assert config.environment == "development"
        assert config.llm.provider == "mock"
        assert config.embedding.dimension == 384
        assert config.graph.spike_ged_threshold == -0.5
        assert config.monitoring.enabled is False
        assert config.logging.level == "DEBUG"

    def test_experiment_preset(self):
        """Test experiment preset configuration."""
        config = ConfigPresets.experiment()

        assert config.environment == "experiment"
        assert config.llm.provider == "local"
        assert config.llm.model == "distilgpt2"
        assert config.memory.episodic_memory_capacity == 100
        assert config.monitoring.enabled is True
        assert config.logging.level == "INFO"

    def test_production_preset(self):
        """Test production preset configuration."""
        config = ConfigPresets.production()

        assert config.environment == "production"
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-3.5-turbo"
        assert config.memory.episodic_memory_capacity == 200
        assert config.monitoring.enabled is True
        assert config.monitoring.performance_tracking is True
        assert config.logging.level == "WARNING"

    def test_research_preset(self):
        """Test research preset configuration."""
        config = ConfigPresets.research()

        assert config.environment == "research"
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-2"
        assert config.memory.episodic_memory_capacity == 500
        assert config.monitoring.detailed_tracing is True
        assert config.logging.log_to_console is True

    def test_get_preset_by_name(self):
        """Test getting preset by name (backward compatibility)."""
        preset_dict = ConfigPresets.get_preset("development")

        assert isinstance(preset_dict, dict)
        assert preset_dict["environment"] == "development"
        assert preset_dict["llm"]["provider"] == "mock"

        # Test invalid preset name
        with pytest.raises(ValueError):
            ConfigPresets.get_preset("invalid_preset")


class TestConfigLoader:
    """Test configuration loading from files and environment."""

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
environment: production
llm:
  provider: openai
  model: gpt-4
  temperature: 0.3
memory:
  episodic_memory_capacity: 150
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load_from_file(yaml_path)

            assert config.environment == "production"
            assert config.llm.provider == "openai"
            assert config.llm.model == "gpt-4"
            assert config.memory.episodic_memory_capacity == 150
        finally:
            yaml_path.unlink()

    def test_load_from_json(self):
        """Test loading configuration from JSON file."""
        json_content = {
            "environment": "experiment",
            "llm": {"provider": "local", "model": "distilgpt2", "temperature": 0.5},
            "graph": {"spike_ged_threshold": -0.3, "spike_ig_threshold": 0.25},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_content, f)
            json_path = Path(f.name)

        try:
            loader = ConfigLoader()
            config = loader.load_from_file(json_path)

            assert config.environment == "experiment"
            assert config.llm.provider == "local"
            assert config.graph.spike_ged_threshold == -0.3
        finally:
            json_path.unlink()

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ["INSIGHTSPIKE_ENVIRONMENT"] = "testing"
        os.environ["INSIGHTSPIKE_LLM__PROVIDER"] = "anthropic"
        os.environ["INSIGHTSPIKE_LLM__MODEL"] = "claude-2"
        os.environ["INSIGHTSPIKE_LLM__TEMPERATURE"] = "0.8"
        os.environ["INSIGHTSPIKE_MEMORY__EPISODIC_MEMORY_CAPACITY"] = "125"

        try:
            loader = ConfigLoader()
            config = loader._apply_env_overrides(InsightSpikeConfig())

            assert config.environment == "testing"
            assert config.llm.provider == "anthropic"
            assert config.llm.model == "claude-2"
            assert config.llm.temperature == 0.8
            assert config.memory.episodic_memory_capacity == 125
        finally:
            # Clean up environment variables
            for key in list(os.environ.keys()):
                if key.startswith("INSIGHTSPIKE_"):
                    del os.environ[key]

    def test_load_with_preset_and_overrides(self):
        """Test loading with preset and file overrides."""
        json_content = {"llm": {"temperature": 0.9, "max_tokens": 2048}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_content, f)
            json_path = Path(f.name)

        try:
            loader = ConfigLoader()
            # Start with development preset
            config = ConfigPresets.development()

            # Apply file overrides
            file_config = loader.load_from_file(json_path)
            config_dict = config.dict()
            file_dict = file_config.dict()

            # Merge (file overrides preset)
            for key, value in file_dict.items():
                if value is not None:
                    if isinstance(value, dict) and key in config_dict:
                        config_dict[key].update(value)
                    else:
                        config_dict[key] = value

            final_config = InsightSpikeConfig(**config_dict)

            # Check preset values are preserved
            assert final_config.environment == "development"
            assert (
                final_config.llm.provider == "local"
            )  # default provider when not specified

            # Check overrides are applied
            assert final_config.llm.temperature == 0.9
            assert final_config.llm.max_tokens == 2048
        finally:
            json_path.unlink()


class TestGraphConfig:
    """Test GraphConfig validation."""

    def test_spike_thresholds(self):
        """Test spike threshold validation."""
        # Valid negative GED threshold
        config = GraphConfig(spike_ged_threshold=-0.5)
        assert config.spike_ged_threshold == -0.5

        # Valid positive thresholds
        config = GraphConfig(spike_ged_threshold=0.3, spike_ig_threshold=0.4)
        assert config.spike_ged_threshold == 0.3
        assert config.spike_ig_threshold == 0.4

        # Out of range
        with pytest.raises(ValidationError):
            GraphConfig(spike_ged_threshold=-2.0)  # Min is -1.0

        with pytest.raises(ValidationError):
            GraphConfig(spike_ig_threshold=1.5)  # Max is 1.0


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_paths_config_exists(self):
        """Test that PathsConfig is available for legacy code."""
        config = InsightSpikeConfig()

        assert hasattr(config, "paths")
        assert config.paths is not None
        assert hasattr(config.paths, "data_dir")
        assert str(config.paths.data_dir) == "data"
        assert hasattr(config.paths, "raw_dir")
        assert str(config.paths.raw_dir) == "data/raw"

    def test_legacy_fields_excluded_from_dict(self):
        """Test that legacy fields are excluded from serialization."""
        config = InsightSpikeConfig()
        config_dict = config.dict()

        # New fields should be present
        assert "environment" in config_dict
        assert "llm" in config_dict
        assert "embedding" in config_dict

        # Legacy fields should be excluded (if exclude=True)
        # Note: paths is not excluded for backward compatibility
        assert "paths" in config_dict
