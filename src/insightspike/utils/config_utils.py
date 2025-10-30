"""
Configuration utility functions for safe access across dict/object configs
"""

from typing import Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def safe_get(config: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely get nested config values from dict or object configs.
    
    Args:
        config: Configuration (dict or object)
        *keys: Keys to traverse (e.g., 'processing', 'enable_learning')
        default: Default value if not found
        
    Returns:
        Config value or default
        
    Examples:
        >>> safe_get(config, 'processing', 'enable_learning', default=False)
        >>> safe_get(config, 'l4_config', 'provider', default='mock')
    """
    try:
        # Handle None config
        if config is None:
            return default
            
        current = config
        
        for key in keys:
            if current is None:
                return default
                
            # Try dict access first
            if isinstance(current, dict):
                current = current.get(key)
            # Try object attribute access
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
                
        return current if current is not None else default
        
    except Exception as e:
        logger.debug(f"Config access error for {'.'.join(keys)}: {e}")
        return default


def is_dict_config(config: Any) -> bool:
    """Check if config is dict-based."""
    return isinstance(config, dict)


def is_object_config(config: Any) -> bool:
    """Check if config is object-based (has attributes)."""
    return hasattr(config, '__dict__') and not isinstance(config, dict)


def normalize_to_dict(config: Any) -> dict:
    """
    Convert any config format to dict.
    
    Args:
        config: Configuration in any format
        
    Returns:
        Dictionary configuration
    """
    if config is None:
        return {}
        
    if isinstance(config, dict):
        return config
        
    # Convert object to dict
    if hasattr(config, 'dict') and callable(config.dict):
        # Pydantic model
        return config.dict()
    elif hasattr(config, '__dict__'):
        # Regular object - recursively convert
        result = {}
        for key, value in config.__dict__.items():
            if not key.startswith('_'):
                if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, tuple)):
                    result[key] = normalize_to_dict(value)
                else:
                    result[key] = value
        return result
    else:
        # Fallback
        return {}


def get_config_value(config: Any, path: str, default: Any = None) -> Any:
    """
    Get config value using dot notation.
    
    Args:
        config: Configuration
        path: Dot-separated path (e.g., 'processing.enable_learning')
        default: Default value
        
    Returns:
        Config value or default
    """
    keys = path.split('.')
    return safe_get(config, *keys, default=default)


def set_config_value(config: dict, path: str, value: Any) -> None:
    """
    Set config value using dot notation (dict configs only).
    
    Args:
        config: Dictionary configuration
        path: Dot-separated path
        value: Value to set
    """
    if not isinstance(config, dict):
        raise TypeError("set_config_value only works with dict configs")
        
    keys = path.split('.')
    current = config
    
    # Navigate to parent
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
        
    # Set value
    current[keys[-1]] = value


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep merge two dict configs.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    import copy
    
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
            
    return result