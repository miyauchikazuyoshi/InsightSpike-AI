"""
Simplified Configuration System for InsightSpike
==============================================

A cleaner, more intuitive configuration system with presets and validation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from insightspike.utils.error_handler import ConfigurationError, get_logger

logger = get_logger("config")


@dataclass
class SimpleConfig:
    """Simplified configuration with flat structure and sensible defaults"""
    
    # Execution mode
    mode: str = "cpu"  # Options: "cpu", "gpu", "mps"
    safe_mode: bool = True  # Use mock LLM for testing
    debug: bool = False
    
    # Model settings
    embedding_model: str = "paraphrase-MiniLM-L6-v2"
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm_provider: str = "local"  # Options: "local", "openai", "anthropic"
    
    # Performance settings
    max_tokens: int = 256
    temperature: float = 0.3
    batch_size: int = 32
    
    # Memory and retrieval
    similarity_threshold: float = 0.25
    top_k_results: int = 15
    max_episodes: int = 10000
    
    # Spike detection
    spike_ged_threshold: float = 0.5
    spike_ig_threshold: float = 0.2
    spike_sensitivity: float = 1.0  # Multiplier for thresholds
    
    # File paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_dir: Path = field(default_factory=lambda: Path.home() / ".insightspike" / "logs")
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".insightspike" / "cache")
    
    # Advanced options
    use_scalable_graph: bool = True
    use_advanced_metrics: bool = True
    enable_conflict_detection: bool = True
    
    # GED/IG calculation method selection
    ged_algorithm: str = "advanced"  # "simple", "advanced", "networkx"
    ig_algorithm: str = "advanced"   # "simple", "advanced", "entropy"
    
    def __post_init__(self):
        """Initialize and validate configuration"""
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original values before adjustments
        self._original_spike_ged = self.spike_ged_threshold
        self._original_spike_ig = self.spike_ig_threshold
        
        # Adjust settings based on mode
        if self.mode == "gpu" and not self._check_gpu_available():
            logger.warning("GPU requested but not available, falling back to CPU")
            self.mode = "cpu"
            
        # Apply spike sensitivity
        self.spike_ged_threshold *= self.spike_sensitivity
        self.spike_ig_threshold *= self.spike_sensitivity
        
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Convert Path objects to strings
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result
        
    def save(self, path: Union[str, Path]):
        """Save configuration to file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimpleConfig":
        """Load configuration from file"""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Convert string paths back to Path objects
        for key in ['data_dir', 'log_dir', 'cache_dir']:
            if key in data and isinstance(data[key], str):
                data[key] = Path(data[key])
                
        # Remove internal fields that shouldn't be loaded
        data = {k: v for k, v in data.items() if not k.startswith('_')}
                
        return cls(**data)


class ConfigPresets:
    """Pre-configured settings for common use cases"""
    
    @staticmethod
    def development() -> SimpleConfig:
        """Development configuration - fast, safe, debug-friendly"""
        return SimpleConfig(
            mode="cpu",
            safe_mode=True,
            debug=True,
            max_tokens=128,
            batch_size=16,
            spike_sensitivity=2.0  # More sensitive for testing
        )
        
    @staticmethod
    def testing() -> SimpleConfig:
        """Testing configuration - isolated, reproducible"""
        return SimpleConfig(
            mode="cpu",
            safe_mode=True,
            debug=False,
            data_dir=Path("test_data"),
            log_dir=Path("test_logs"),
            cache_dir=Path("test_cache"),
            max_episodes=100
        )
        
    @staticmethod
    def production() -> SimpleConfig:
        """Production configuration - optimized for performance"""
        return SimpleConfig(
            mode="gpu",
            safe_mode=False,
            debug=False,
            max_tokens=512,
            batch_size=64,
            use_scalable_graph=True,
            use_advanced_metrics=True
        )
        
    @staticmethod
    def experiment() -> SimpleConfig:
        """Experiment configuration - real LLM, moderate performance"""
        return SimpleConfig(
            mode="cpu",
            safe_mode=False,
            debug=True,
            max_tokens=256,
            batch_size=32,
            spike_sensitivity=1.5
        )
        
    @staticmethod
    def cloud() -> SimpleConfig:
        """Cloud configuration - for API-based LLMs"""
        return SimpleConfig(
            mode="cpu",
            safe_mode=False,
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            max_tokens=1024,
            temperature=0.7
        )


class ConfigManager:
    """Manages configuration with environment variable support"""
    
    ENV_PREFIX = "INSIGHTSPIKE_"
    
    def __init__(self, config: Optional[SimpleConfig] = None):
        self.config = config or SimpleConfig()
        self._apply_env_overrides()
        
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        for key in dir(self.config):
            if key.startswith('_'):
                continue
                
            env_key = f"{self.ENV_PREFIX}{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                
                # Type conversion
                current_value = getattr(self.config, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, Path):
                    value = Path(value)
                    
                setattr(self.config, key, value)
                logger.debug(f"Override from env: {key} = {value}")
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return getattr(self.config, key, default)
        
    def set(self, key: str, value: Any):
        """Set configuration value"""
        if not hasattr(self.config, key):
            raise ConfigurationError(f"Unknown configuration key: {key}")
        setattr(self.config, key, value)
        
    def update(self, **kwargs):
        """Update multiple configuration values"""
        for key, value in kwargs.items():
            self.set(key, value)
            
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check required paths exist
        if not self.config.data_dir.exists():
            errors.append(f"Data directory not found: {self.config.data_dir}")
            
        # Check model availability
        if not self.config.safe_mode:
            if self.config.llm_provider == "local":
                # Check if model is available locally
                model_path = Path.home() / ".cache" / "huggingface" / "hub"
                model_id = self.config.llm_model.replace("/", "--")
                if not any(model_path.glob(f"*{model_id}*")):
                    errors.append(f"Local model not found: {self.config.llm_model}")
                    
        # Check GPU availability
        if self.config.mode == "gpu" and not self.config._check_gpu_available():
            errors.append("GPU mode requested but no GPU available")
            
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
            
        return True
        
    def to_legacy_config(self):
        """Convert to legacy Config object for backward compatibility"""
        from insightspike.core.config import (
            Config, LLMConfig, EmbeddingConfig, SpikeConfig,
            MemoryConfig, RetrievalConfig, ReasoningConfig, GraphConfig
        )
        
        legacy = Config()
        
        # Map simple config to legacy nested structure
        legacy.llm = LLMConfig(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            device=self.config.mode,
            safe_mode=self.config.safe_mode
        )
        
        legacy.embedding = EmbeddingConfig(
            model_name=self.config.embedding_model,
            device=self.config.mode,
            dimension=384  # Default dimension for MiniLM
        )
        
        legacy.spike = SpikeConfig(
            spike_ged=self.config.spike_ged_threshold,
            spike_ig=self.config.spike_ig_threshold
        )
        
        # Add memory config with all required fields
        legacy.memory = MemoryConfig(
            max_retrieved_docs=self.config.top_k_results,
            min_similarity=self.config.similarity_threshold,
            nlist=256,  # Default values from MemoryConfig
            pq_segments=16,
            c_value_gamma=0.5,
            c_value_min=0.0,
            c_value_max=1.0,
            merge_ged=0.4,
            split_ig=-0.15,
            prune_c=0.05,
            inactive_n=30,
            index_file="data/index.faiss"
        )
        
        # Add retrieval config
        legacy.retrieval = RetrievalConfig(
            similarity_threshold=self.config.similarity_threshold,
            top_k=self.config.top_k_results
        )
        
        # Add reasoning config
        legacy.reasoning = ReasoningConfig(
            similarity_threshold=self.config.similarity_threshold,
            spike_ged_threshold=self.config.spike_ged_threshold,
            spike_ig_threshold=self.config.spike_ig_threshold,
            use_scalable_graph=self.config.use_scalable_graph,
            use_advanced_metrics=self.config.use_advanced_metrics
        )
        
        # Add graph config
        legacy.graph = GraphConfig(
            spike_ged_threshold=self.config.spike_ged_threshold,
            spike_ig_threshold=self.config.spike_ig_threshold
        )
        
        # Set paths
        legacy.paths.data_dir = self.config.data_dir
        legacy.paths.log_dir = self.config.log_dir
        
        return legacy


# Convenience functions
def get_config(preset: str = "development") -> SimpleConfig:
    """Get configuration with preset"""
    preset_map = {
        "development": ConfigPresets.development,
        "testing": ConfigPresets.testing,
        "production": ConfigPresets.production,
        "experiment": ConfigPresets.experiment,
        "cloud": ConfigPresets.cloud
    }
    
    if preset not in preset_map:
        raise ConfigurationError(f"Unknown preset: {preset}")
        
    return preset_map[preset]()


def create_config_file(path: Union[str, Path], preset: str = "development"):
    """Create a configuration file with preset"""
    config = get_config(preset)
    config.save(path)
    print(f"Configuration file created: {path}")
    print(f"Preset: {preset}")
    print(f"Edit the file to customize settings")