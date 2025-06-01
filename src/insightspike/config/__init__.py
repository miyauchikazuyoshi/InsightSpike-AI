"""
InsightSpike-AI Configuration Management
=====================================

Unified configuration system - delegates to core.config for consistency.
"""

# Import the structured config system from core
from ..core.config import get_config, Config, get_legacy_config

# Re-export for backward compatibility
__all__ = ['get_config', 'Config', 'get_legacy_config']

# Legacy compatibility functions
def reload_config():
    """Reload configuration (no-op for compatibility)"""
    return get_config()

def set_config(config):
    """Set configuration (no-op for compatibility)"""
    pass

# Export common legacy constants for backward compatibility
_legacy = get_legacy_config()
ROOT_DIR = _legacy['ROOT_DIR']
DATA_DIR = _legacy['DATA_DIR'] 
LOG_DIR = _legacy['LOG_DIR']
EMBED_MODEL_NAME = _legacy['EMBED_MODEL_NAME']
