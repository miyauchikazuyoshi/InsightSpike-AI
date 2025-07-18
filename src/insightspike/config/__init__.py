"""
Unified Configuration System for InsightSpike
============================================

Provides a clean interface for configuration management:
- Pydantic-based configuration models
- YAML/JSON file support
- Environment variable overrides
- Presets for common use cases
- Backward compatibility with legacy config
"""

from .loader import get_config, load_config
from .models import InsightSpikeConfig
from .presets import ConfigPresets

# Legacy imports for backward compatibility
from .simple_config import ConfigManager, SimpleConfig, create_config_file

__all__ = [
    # Primary interface
    "get_config",
    "load_config",
    "InsightSpikeConfig",
    "ConfigPresets",
    # Legacy compatibility
    "SimpleConfig",
    "ConfigManager",
    "create_config_file",
]
