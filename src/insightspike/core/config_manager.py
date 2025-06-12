"""
Configuration Management System
==============================

Enhanced configuration system for InsightSpike-AI that supports:
- YAML/JSON external configuration files
- Command line argument overrides  
- Environment variable support
- Preset configurations
- Random seed management for reproducibility
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict, field

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigManager",
    "InsightSpikeConfig", 
    "load_config",
    "save_config",
    "get_config_manager",
    "apply_cli_overrides"
]


@dataclass
class GNNConfig:
    """Graph Neural Network configuration."""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"


@dataclass  
class MemoryConfig:
    """Memory system configuration."""
    vector_dim: int = 384
    max_items: int = 10000
    ivf_nlist: int = 100
    pq_m: int = 64


@dataclass
class InsightDetectionConfig:
    """Insight detection configuration."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'ged': 0.4, 'ig': 0.3, 'conflict': 0.3
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'ged_threshold': -0.5, 'ig_threshold': 0.2
    })


@dataclass
class AlgorithmConfig:
    """Algorithm-specific configuration."""
    ged: Dict[str, Any] = field(default_factory=lambda: {
        'optimization_level': 'standard',
        'timeout_seconds': 5.0,
        'node_cost': 1.0,
        'edge_cost': 1.0
    })
    ig: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'clustering',
        'k_clusters': 8,
        'min_samples': 2
    })


@dataclass
class ExperimentalConfig:
    """Experimental settings."""
    random_seed: Optional[int] = 42
    batch_size: int = 32
    learning_rate: float = 0.001
    verbose_logging: bool = False


@dataclass
class InsightSpikeConfig:
    """Complete InsightSpike-AI configuration."""
    model: GNNConfig = field(default_factory=GNNConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    insight_detection: InsightDetectionConfig = field(default_factory=InsightDetectionConfig)
    algorithms: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightSpikeConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclass creation
        model_data = data.get('model', {})
        memory_data = data.get('memory', {})
        insight_data = data.get('insight_detection', {})
        algorithm_data = data.get('algorithms', {})
        experimental_data = data.get('experimental', {})
        
        return cls(
            model=GNNConfig(**model_data),
            memory=MemoryConfig(**memory_data),
            insight_detection=InsightDetectionConfig(**insight_data),
            algorithms=AlgorithmConfig(**algorithm_data),
            experimental=ExperimentalConfig(**experimental_data)
        )


class ConfigManager:
    """Manages configuration loading, saving, and overrides."""
    
    def __init__(self):
        self._config = InsightSpikeConfig()
        self._config_file_path: Optional[Path] = None
        
    @property
    def config(self) -> InsightSpikeConfig:
        """Get current configuration."""
        return self._config
    
    def load_from_file(self, config_path: Union[str, Path]) -> InsightSpikeConfig:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is unsupported or invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        suffix = config_path.suffix.lower()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if suffix == '.yaml' or suffix == '.yml':
                    if not YAML_AVAILABLE:
                        raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
                    data = yaml.safe_load(f)
                elif suffix == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {suffix}")
            
            # Apply environment variable overrides
            data = self._apply_env_overrides(data)
            
            # Create configuration object
            self._config = InsightSpikeConfig.from_dict(data)
            self._config_file_path = config_path
            
            logger.info(f"Configuration loaded from {config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_to_file(self, config_path: Union[str, Path], 
                    format: str = 'yaml') -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self._config.to_dict()
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    if not YAML_AVAILABLE:
                        raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
                    yaml.safe_dump(data, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def load_from_env(self) -> InsightSpikeConfig:
        """Load configuration from environment variables."""
        # Check for config file path in environment
        config_path = os.getenv('INSIGHTSPIKE_CONFIG_PATH')
        if config_path and Path(config_path).exists():
            return self.load_from_file(config_path)
        
        # Apply environment overrides to default config
        data = self._config.to_dict()
        data = self._apply_env_overrides(data)
        self._config = InsightSpikeConfig.from_dict(data)
        
        logger.info("Configuration loaded from environment variables")
        return self._config
    
    def apply_cli_overrides(self, **kwargs) -> InsightSpikeConfig:
        """
        Apply command line argument overrides.
        
        Args:
            **kwargs: Command line arguments
            
        Returns:
            Updated configuration
        """
        # Map CLI arguments to config structure
        cli_mappings = {
            'ged_optimization': 'algorithms.ged.optimization_level',
            'ig_method': 'algorithms.ig.method',
            'random_seed': 'experimental.random_seed',
            'batch_size': 'experimental.batch_size',
            'learning_rate': 'experimental.learning_rate',
            'hidden_dim': 'model.gnn.hidden_dim',
            'num_layers': 'model.gnn.num_layers',
            'vector_dim': 'memory.vector_dim',
            'max_items': 'memory.max_items',
            'verbose': 'experimental.verbose_logging'
        }
        
        data = self._config.to_dict()
        
        for cli_arg, config_path in cli_mappings.items():
            if cli_arg in kwargs and kwargs[cli_arg] is not None:
                self._set_nested_value(data, config_path, kwargs[cli_arg])
                logger.debug(f"CLI override: {config_path} = {kwargs[cli_arg]}")
        
        self._config = InsightSpikeConfig.from_dict(data)
        return self._config
    
    def _apply_env_overrides(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration data."""
        env_mappings = {
            'INSIGHTSPIKE_RANDOM_SEED': 'experimental.random_seed',
            'INSIGHTSPIKE_LOG_LEVEL': 'experimental.verbose_logging',
            'INSIGHTSPIKE_GED_OPTIMIZATION': 'algorithms.ged.optimization_level',
            'INSIGHTSPIKE_IG_METHOD': 'algorithms.ig.method',
            'INSIGHTSPIKE_HIDDEN_DIM': 'model.gnn.hidden_dim',
            'INSIGHTSPIKE_VECTOR_DIM': 'memory.vector_dim'
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion
                if config_path.endswith('random_seed') or config_path.endswith('hidden_dim') or config_path.endswith('vector_dim'):
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {env_value}")
                        continue
                elif config_path.endswith('verbose_logging'):
                    env_value = env_value.lower() in ('true', '1', 'yes', 'debug')
                
                self._set_nested_value(data, config_path, env_value)
                logger.debug(f"Environment override: {config_path} = {env_value}")
        
        return data
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set a nested dictionary value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_preset_config(self, preset_name: str) -> InsightSpikeConfig:
        """
        Get a preset configuration.
        
        Args:
            preset_name: Name of preset configuration
            
        Returns:
            Preset configuration
        """
        presets = {
            'research_high_precision': {
                'insight_detection': {
                    'weights': {'ged': 0.5, 'ig': 0.4, 'conflict': 0.1},
                    'thresholds': {'ged_threshold': -0.3, 'ig_threshold': 0.3}
                },
                'algorithms': {
                    'ged': {'optimization_level': 'precise', 'timeout_seconds': 10.0},
                    'ig': {'method': 'clustering', 'k_clusters': 12}
                }
            },
            'production_balanced': {
                'insight_detection': {
                    'weights': {'ged': 0.33, 'ig': 0.33, 'conflict': 0.34},
                    'thresholds': {'ged_threshold': -0.5, 'ig_threshold': 0.2}
                },
                'algorithms': {
                    'ged': {'optimization_level': 'standard', 'timeout_seconds': 5.0},
                    'ig': {'method': 'clustering', 'k_clusters': 8}
                }
            },
            'real_time_fast': {
                'insight_detection': {
                    'weights': {'ged': 0.4, 'ig': 0.3, 'conflict': 0.3},
                    'thresholds': {'ged_threshold': -0.7, 'ig_threshold': 0.3}
                },
                'algorithms': {
                    'ged': {'optimization_level': 'fast', 'timeout_seconds': 1.0},
                    'ig': {'method': 'shannon', 'k_clusters': 4}
                }
            }
        }
        
        if preset_name not in presets:
            available = list(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        # Merge preset with default configuration
        default_data = self._config.to_dict()
        preset_data = presets[preset_name]
        
        # Deep merge
        merged_data = self._deep_merge(default_data, preset_data)
        
        return InsightSpikeConfig.from_dict(merged_data)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[Union[str, Path]] = None, 
               preset: Optional[str] = None,
               **cli_overrides) -> InsightSpikeConfig:
    """
    Load configuration with multiple sources.
    
    Args:
        config_path: Path to configuration file
        preset: Preset configuration name
        **cli_overrides: Command line overrides
        
    Returns:
        Loaded configuration
    """
    manager = get_config_manager()
    
    # Load from file if provided
    if config_path:
        manager.load_from_file(config_path)
    # Load from environment
    else:
        manager.load_from_env()
    
    # Apply preset if specified
    if preset:
        preset_config = manager.get_preset_config(preset)
        manager._config = preset_config
    
    # Apply CLI overrides
    if cli_overrides:
        manager.apply_cli_overrides(**cli_overrides)
    
    return manager.config


def save_config(config: InsightSpikeConfig, 
               config_path: Union[str, Path],
               format: str = 'yaml') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration
        format: File format ('yaml' or 'json')
    """
    manager = get_config_manager()
    manager._config = config
    manager.save_to_file(config_path, format)


def apply_cli_overrides(config: InsightSpikeConfig, **kwargs) -> InsightSpikeConfig:
    """
    Apply command line overrides to configuration.
    
    Args:
        config: Base configuration
        **kwargs: CLI arguments
        
    Returns:
        Updated configuration
    """
    manager = get_config_manager()
    manager._config = config
    return manager.apply_cli_overrides(**kwargs)
