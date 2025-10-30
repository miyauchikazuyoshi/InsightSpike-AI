"""Configuration system for geDIG-RAG v3 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class GeDIGConfig:
    """geDIG evaluation parameters."""
    k_coefficient: float = 0.5
    radius: int = 2
    
    # Update thresholds
    add_ig_threshold: float = 0.3
    add_ged_min: float = 0.1
    add_ged_max: float = 0.8
    merge_similarity: float = 0.8
    merge_ig_min: float = 0.05
    prune_usage_min: int = 2
    prune_isolation_threshold: int = 1
    prune_contradiction_threshold: float = -0.1
    
    # Decision thresholding for geDIG (F = ΔGED - k·ΔIG; smaller is better)
    # mode: 'fixed' uses threshold_value, 'percentile' uses threshold_percentile
    threshold_mode: str = "fixed"
    threshold_value: float = 0.0
    threshold_percentile: float = 20.0


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    embedding_model: str = "all-MiniLM-L6-v2"
    generation_model: str = "microsoft/DialoGPT-medium"
    nli_model: str = "microsoft/DialoGPT-medium"
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    hotpot_qa_size: int = 1000
    domain_qa_path: str = "data/input/domain_qa.json"
    use_cache: bool = True
    cache_dir: str = "data/processed/cache"


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    
    # Core parameters
    gedig: GeDIGConfig = field(default_factory=GeDIGConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Experiment parameters
    n_sessions: int = 5
    queries_per_session: int = 20
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    enable_baselines: List[str] = field(
        default_factory=lambda: ["static", "frequency", "cosine", "gedig"]
    )
    
    # Evaluation parameters
    evaluation_metrics: List[str] = field(
        default_factory=lambda: ["em_f1", "recall", "mrr", "bleu", "rouge"]
    )
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Output parameters
    output_dir: str = "results"
    save_detailed_logs: bool = True
    generate_figures: bool = True
    output_formats: List[str] = field(
        default_factory=lambda: ["json", "csv", "png", "pdf"]
    )
    
    # Computational parameters
    n_processes: int = 4
    gpu_enabled: bool = False
    memory_limit_gb: int = 8
    
    # Baseline-specific parameters
    frequency_threshold: int = 3
    time_threshold_hours: float = 24.0
    cosine_similarity_threshold: float = 0.7
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> ExperimentConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create nested config objects
        config = cls()
        
        if 'gedig' in config_dict:
            config.gedig = GeDIGConfig(**config_dict['gedig'])
        
        if 'models' in config_dict:
            config.models = ModelConfig(**config_dict['models'])
        
        if 'datasets' in config_dict:
            config.datasets = DatasetConfig(**config_dict['datasets'])
        
        # Update other parameters
        for key, value in config_dict.items():
            if key not in ['gedig', 'models', 'datasets'] and hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gedig': self.gedig.__dict__,
            'models': self.models.__dict__,
            'datasets': self.datasets.__dict__,
            'experiment': {
                'n_sessions': self.n_sessions,
                'queries_per_session': self.queries_per_session,
                'seeds': self.seeds,
                'enable_baselines': self.enable_baselines
            },
            'evaluation': {
                'metrics': self.evaluation_metrics,
                'top_k_values': self.top_k_values
            },
            'output': {
                'output_dir': self.output_dir,
                'save_detailed_logs': self.save_detailed_logs,
                'generate_figures': self.generate_figures,
                'output_formats': self.output_formats
            },
            'computational': {
                'n_processes': self.n_processes,
                'gpu_enabled': self.gpu_enabled,
                'memory_limit_gb': self.memory_limit_gb
            }
        }
    
    def save_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, indent=2)


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()
