"""
Legacy Configuration Adapter
===========================

Handles conversion from legacy dict-based configs to Pydantic models.
This allows gradual migration while maintaining backward compatibility.
"""

import warnings
from typing import Any, Dict, Union

from .models import InsightSpikeConfig
from .normalizer import ConfigNormalizer


class LegacyConfigAdapter:
    """
    Adapter for converting legacy configurations to Pydantic models.
    
    Usage:
        # In any component that receives config
        config = LegacyConfigAdapter.ensure_pydantic(config)
    """
    
    @staticmethod
    def ensure_pydantic(config: Union[Dict[str, Any], InsightSpikeConfig, Any]) -> InsightSpikeConfig:
        """
        Ensure config is a Pydantic InsightSpikeConfig object.
        
        Args:
            config: Legacy dict config, Pydantic config, or other format
            
        Returns:
            InsightSpikeConfig object
        """
        # Already Pydantic
        if isinstance(config, InsightSpikeConfig):
            return config
        
        # Dict config (legacy)
        if isinstance(config, dict):
            warnings.warn(
                "Dict-based configs are deprecated. Please migrate to InsightSpikeConfig.",
                DeprecationWarning,
                stacklevel=2
            )
            return ConfigNormalizer.normalize(config)
        
        # SimpleNamespace or other object with __dict__
        if hasattr(config, '__dict__'):
            warnings.warn(
                f"{type(config).__name__} configs are deprecated. Please migrate to InsightSpikeConfig.",
                DeprecationWarning,
                stacklevel=2
            )
            # Convert to dict first, then normalize
            config_dict = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list)):
                    # Nested object, convert recursively
                    config_dict[key] = value.__dict__
                else:
                    config_dict[key] = value
            return ConfigNormalizer.normalize(config_dict)
        
        # Unknown format
        raise ValueError(f"Unsupported config type: {type(config)}")
    
    @staticmethod
    def get_value(config: Any, path: str, default: Any = None) -> Any:
        """
        Get a value from config using dot notation, handling both dict and object access.
        
        Args:
            config: Config object (dict, Pydantic, or SimpleNamespace)
            path: Dot-separated path (e.g., "graph.spike_ged_threshold")
            default: Default value if path not found
            
        Returns:
            Value at path or default
        """
        parts = path.split('.')
        current = config
        
        for part in parts:
            try:
                if isinstance(current, dict):
                    current = current.get(part, default)
                    if current is default:
                        return default
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return default
            except (AttributeError, KeyError, TypeError):
                return default
        
        return current
    
    @staticmethod
    def is_legacy(config: Any) -> bool:
        """Check if config is in legacy format."""
        return isinstance(config, dict) or (
            hasattr(config, '__dict__') and 
            not isinstance(config, InsightSpikeConfig)
        )