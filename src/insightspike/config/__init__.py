"""
InsightSpike-AI Configuration Management
=====================================

Centralized configuration for all components with environment-specific overrides.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import os
import yaml


@dataclass
class PathConfig:
    """File and directory paths"""
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent)
    data_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.root_dir / "data"
        self.logs_dir = self.data_dir / "logs"
        self.cache_dir = self.data_dir / "cache"
        self.models_dir = self.data_dir / "models"
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.logs_dir, self.cache_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model and embedding configurations"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    llm_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    max_seq_length: int = 512


@dataclass
class MemoryConfig:
    """Memory and retrieval configurations"""
    top_k: int = 5
    similarity_threshold: float = 0.35
    c_value_init: float = 0.2
    c_value_decay: float = 0.95
    c_value_boost: float = 1.1
    max_episodes: int = 10000
    quantization_levels: int = 256


@dataclass
class GraphConfig:
    """Graph and GNN configurations"""
    edge_threshold: float = 0.4
    max_edges_per_node: int = 10
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    dropout_rate: float = 0.1


@dataclass
class SpikeConfig:
    """Eureka spike detection thresholds"""
    ged_threshold: float = 0.5
    ig_threshold: float = 0.2
    reward_weight: float = 0.2
    spike_cooldown: int = 3  # minimum loops between spikes


@dataclass
class LoopConfig:
    """Agent loop configurations"""
    max_loops: int = 10
    timeout_seconds: int = 300
    error_tolerance: int = 3
    adaptive_learning_rate: bool = True


@dataclass
class LoggingConfig:
    """Logging and monitoring configurations"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_wandb: bool = False
    wandb_project: str = "insightspike-ai"
    save_artifacts: bool = True


@dataclass
class Config:
    """Main configuration container"""
    paths: PathConfig = field(default_factory=PathConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    environment: str = "local"  # local, colab, production
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file"""
        if not config_path.exists():
            return cls()
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create config instance and update with YAML data
        config = cls()
        config._update_from_dict(data)
        return config
    
    @classmethod
    def from_environment(cls) -> "Config":
        """Load configuration with environment-specific settings"""
        config = cls()
        
        # Detect environment
        if "COLAB_GPU" in os.environ:
            config.environment = "colab"
            config.models.device = "cuda"
            config.models.batch_size = 16  # Conservative for Colab
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            config.environment = "production"
            config.models.device = "cuda"
        else:
            config.environment = "local"
            config.models.device = "cpu"
        
        # Load environment-specific config file
        env_config_path = config.paths.root_dir / "config" / f"{config.environment}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_data = yaml.safe_load(f)
            config._update_from_dict(env_data)
        
        return config
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in data.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def save_yaml(self, config_path: Path):
        """Save current configuration to YAML file"""
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            if field_name not in ['paths']:  # Skip paths as they're computed
                section = getattr(self, field_name)
                if hasattr(section, '__dataclass_fields__'):
                    config_dict[field_name] = {
                        f.name: getattr(section, f.name) 
                        for f in section.__dataclass_fields__.values()
                    }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def timestamp(self) -> str:
        """Generate timestamp string"""
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config.from_environment()
    return _config


def set_config(config: Config):
    """Set global configuration instance"""
    global _config
    _config = config


def reload_config():
    """Reload configuration from environment"""
    global _config
    _config = Config.from_environment()


# Backward compatibility
def load_config() -> Config:
    """Load configuration (backward compatibility)"""
    return get_config()


# Export main symbols
__all__ = [
    'Config', 'PathConfig', 'ModelConfig', 'MemoryConfig', 
    'GraphConfig', 'SpikeConfig', 'LoopConfig', 'LoggingConfig',
    'get_config', 'set_config', 'reload_config', 'load_config'
]
