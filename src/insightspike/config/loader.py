"""
Configuration Loading and Management
===================================

Handles loading configuration from various sources:
- YAML/JSON files
- Environment variables
- Command-line arguments
- Presets
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .models import InsightSpikeConfig
from .presets import ConfigPresets

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Unified configuration loader that handles all configuration sources"""

    ENV_PREFIX = "INSIGHTSPIKE_"
    CONFIG_PATH_ENV = "INSIGHTSPIKE_CONFIG_PATH"

    def __init__(self):
        self._config: Optional[InsightSpikeConfig] = None
        self._config_path: Optional[Path] = None

    def load(
        self,
        config_path: Optional[Union[str, Path]] = None,
        preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> InsightSpikeConfig:
        """
        Load configuration from various sources with priority:
        1. Command-line overrides (highest)
        2. Environment variables
        3. Config file (YAML/JSON)
        4. Preset
        5. Defaults (lowest)

        Args:
            config_path: Path to config file (YAML/JSON)
            preset: Preset name to use as base
            overrides: Dictionary of overrides

        Returns:
            Loaded configuration
        """
        # Start with defaults or preset
        if preset:
            config_dict = ConfigPresets.get_preset(preset)
        else:
            config_dict = {}

        # Load from file if specified
        file_config = self._load_from_file(config_path)
        if file_config:
            config_dict = self._deep_merge(config_dict, file_config)

        # Apply environment variables
        env_config = self._load_from_env()
        if env_config:
            config_dict = self._deep_merge(config_dict, env_config)

        # Apply explicit overrides
        if overrides:
            config_dict = self._deep_merge(config_dict, overrides)

        # Create and validate configuration
        self._config = InsightSpikeConfig(**config_dict)
        return self._config

    def _load_from_file(
        self, config_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        # Check explicit path first
        if config_path:
            path = Path(config_path)
        else:
            # Check environment variable
            env_path = os.getenv(self.CONFIG_PATH_ENV)
            if env_path:
                path = Path(env_path)
            else:
                # Check default locations
                for default_path in [
                    "config.yaml",
                    "config.json",
                    ".insightspike.yaml",
                ]:
                    if Path(default_path).exists():
                        path = Path(default_path)
                        break
                else:
                    return {}

        if not path.exists():
            logger.debug(f"Config file not found: {path}")
            return {}

        self._config_path = path
        logger.info(f"Loading config from: {path}")

        try:
            with open(path, "r") as f:
                if path.suffix in [".yaml", ".yml"]:
                    try:
                        import yaml

                        return yaml.safe_load(f) or {}
                    except ImportError:
                        logger.warning(
                            "PyYAML not installed. Cannot load YAML config files. Use JSON instead or install PyYAML."
                        )
                        return {}
                elif path.suffix == ".json":
                    return json.load(f)
                else:
                    # Try to detect format - try JSON first
                    content = f.read()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Try YAML if available
                        try:
                            import yaml

                            f.seek(0)
                            return yaml.safe_load(f) or {}
                        except ImportError:
                            logger.warning(
                                "Could not parse config file as JSON and PyYAML not available"
                            )
                            return {}
        except Exception as e:
            logger.error(f"Failed to load config file {path}: {e}")
            return {}

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}

        # Map of environment variables to config paths
        env_mappings = {
            # New format with double underscore for nested fields
            f"{self.ENV_PREFIX}LLM__PROVIDER": "llm.provider",
            f"{self.ENV_PREFIX}LLM__MODEL": "llm.model",
            f"{self.ENV_PREFIX}LLM__TEMPERATURE": "llm.temperature",
            f"{self.ENV_PREFIX}LLM__MAX_TOKENS": "llm.max_tokens",
            f"{self.ENV_PREFIX}MEMORY__EPISODIC_MEMORY_CAPACITY": "memory.episodic_memory_capacity",
            f"{self.ENV_PREFIX}MEMORY__MAX_RETRIEVED_DOCS": "memory.max_retrieved_docs",
            f"{self.ENV_PREFIX}ENVIRONMENT": "environment",
            f"{self.ENV_PREFIX}LOGGING__LEVEL": "logging.level",
            f"{self.ENV_PREFIX}LOGGING__FILE_PATH": "logging.file_path",
            # Legacy mappings for backward compatibility
            f"{self.ENV_PREFIX}MODEL_NAME": "embedding.model_name",
            f"{self.ENV_PREFIX}DATA_DIR": "paths.data_dir",
            f"{self.ENV_PREFIX}LOG_DIR": "paths.log_dir",
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert boolean strings
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                # Convert numeric strings
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)

                # Set nested value
                self._set_nested(config, config_path, value)

        return config

    def _deep_merge(
        self, base: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _set_nested(self, d: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation"""
        keys = path.split(".")
        current = d

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _is_float(self, value: str) -> bool:
        """Check if string is a float"""
        try:
            float(value)
            return "." in value
        except ValueError:
            return False

    def load_from_file(self, path: Union[str, Path]) -> InsightSpikeConfig:
        """Public method to load configuration from file"""
        config_dict = self._load_from_file(path)
        if not config_dict:
            # Return default config if file is empty
            config_dict = {}
        return InsightSpikeConfig(**config_dict)

    def _apply_env_overrides(self, config: InsightSpikeConfig) -> InsightSpikeConfig:
        """Apply environment variable overrides to existing config"""
        env_config = self._load_from_env()
        if env_config:
            config_dict = config.dict()
            config_dict = self._deep_merge(config_dict, env_config)
            return InsightSpikeConfig(**config_dict)
        return config

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file"""
        if not self._config:
            raise ValueError("No configuration loaded")

        save_path = Path(path) if path else self._config_path
        if not save_path:
            save_path = Path("config.yaml")

        config_dict = self._config.dict()

        with open(save_path, "w") as f:
            if save_path.suffix in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to: {save_path}")


# Global instance
_loader = ConfigLoader()


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> InsightSpikeConfig:
    """Load configuration using global loader"""
    return _loader.load(config_path, preset, overrides)


def get_config() -> InsightSpikeConfig:
    """Get current configuration or load defaults"""
    if _loader._config is None:
        _loader.load()
    return _loader._config
