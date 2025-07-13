"""
InsightSpike Configuration Module
================================

Provides simplified configuration management.
"""

from .simple_config import (
    ConfigManager,
    ConfigPresets,
    SimpleConfig,
    create_config_file,
)
from .simple_config import get_config as _get_simple_config


# Create a wrapper that returns legacy config for backward compatibility
def get_config(preset: str = "development", legacy: bool = True):
    """Get configuration with optional legacy format"""
    simple_config = _get_simple_config(preset)

    if legacy:
        # Return legacy config for backward compatibility
        manager = ConfigManager(simple_config)
        return manager.to_legacy_config()
    else:
        return simple_config


__all__ = [
    "SimpleConfig",
    "ConfigPresets",
    "ConfigManager",
    "get_config",
    "create_config_file",
]
