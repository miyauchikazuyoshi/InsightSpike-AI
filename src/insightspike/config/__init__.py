"""
InsightSpike Configuration Module
================================

Provides simplified configuration management.
"""

from .simple_config import (
    SimpleConfig,
    ConfigPresets,
    ConfigManager,
    get_config,
    create_config_file
)

__all__ = [
    "SimpleConfig",
    "ConfigPresets", 
    "ConfigManager",
    "get_config",
    "create_config_file"
]