"""
Unified Configuration Models for InsightSpike
============================================

Pydantic-based configuration models that match the YAML structure.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, validator


class CoreConfig(BaseModel):
    """Core language model settings"""
    model_name: str = Field(default="paraphrase-MiniLM-L6-v2", description="Embedding model")
    llm_provider: Literal["local", "openai", "anthropic", "mock", "clean"] = Field(default="local")
    llm_model: str = Field(default="distilgpt2")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    device: str = Field(default="cpu")
    use_gpu: bool = Field(default=False)
    safe_mode: bool = Field(default=False, description="Use mock LLM for testing")


class MemoryConfig(BaseModel):
    """Memory system configuration"""
    max_retrieved_docs: int = Field(default=15, ge=1)
    short_term_capacity: int = Field(default=10, ge=1)
    working_memory_capacity: int = Field(default=20, ge=1)
    episodic_memory_capacity: int = Field(default=60, ge=1)
    pattern_cache_capacity: int = Field(default=15, ge=1)


class RetrievalConfig(BaseModel):
    """Retrieval settings (Top-K configuration)"""
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    top_k: int = Field(default=15, ge=1)
    layer1_top_k: int = Field(default=20, ge=1)
    layer2_top_k: int = Field(default=15, ge=1)
    layer3_top_k: int = Field(default=12, ge=1)


class HybridWeightsConfig(BaseModel):
    """Hybrid algorithm weights"""
    structure: float = Field(default=0.4, ge=0.0, le=1.0)
    semantic: float = Field(default=0.4, ge=0.0, le=1.0)
    quality: float = Field(default=0.2, ge=0.0, le=1.0)
    
    @validator("quality")
    def validate_sum(cls, v, values):
        if "structure" in values and "semantic" in values:
            total = values["structure"] + values["semantic"] + v
            if not 0.99 <= total <= 1.01:  # Allow small floating point errors
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


class GraphConfig(BaseModel):
    """Graph processing & spike detection"""
    spike_ged_threshold: float = Field(default=0.5, ge=-1.0, le=1.0)  # Allow negative for delta values
    spike_ig_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)  # Add this field
    use_gnn: bool = Field(default=False)
    gnn_hidden_dim: int = Field(default=64, ge=1)
    ged_algorithm: Literal["simple", "advanced", "networkx", "hybrid"] = Field(default="hybrid")
    ig_algorithm: Literal["simple", "advanced", "entropy", "hybrid"] = Field(default="hybrid")
    hybrid_weights: HybridWeightsConfig = Field(default_factory=HybridWeightsConfig)


class ReasoningConfig(BaseModel):
    """Reasoning layer configuration"""
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    conflict_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    weight_ged: float = Field(default=1.0, ge=0.0)
    weight_ig: float = Field(default=1.0, ge=0.0)
    weight_conflict: float = Field(default=0.5, ge=0.0)
    
    # Episode integration thresholds
    episode_integration_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    episode_integration_content_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    episode_integration_c_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Episode management
    episode_merge_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    episode_split_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    episode_prune_threshold: float = Field(default=0.1, ge=0.0, le=1.0)


class SpikeConfig(BaseModel):
    """Eureka spike detection"""
    spike_ged: float = Field(default=0.5, ge=0.0)
    spike_ig: float = Field(default=0.2, ge=0.0)
    eta_spike: float = Field(default=0.2, ge=0.0, le=1.0)


class UnknownLearnerConfig(BaseModel):
    """Unknown learning system"""
    initial_confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    cleanup_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    confidence_boost: float = Field(default=0.05, ge=0.0, le=1.0)
    max_weak_edges: int = Field(default=1000, ge=1)
    cleanup_interval: int = Field(default=300, ge=1)


class PathsConfig(BaseModel):
    """File paths configuration"""
    data_dir: Path = Field(default=Path("data/raw"))
    log_dir: Path = Field(default=Path("data/logs"))
    index_file: Path = Field(default=Path("data/index.faiss"))
    graph_file: Path = Field(default=Path("data/graph_pyg.pt"))
    
    @validator("*", pre=True)
    def convert_to_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class ProcessingConfig(BaseModel):
    """Processing settings"""
    batch_size: int = Field(default=32, ge=1)
    max_workers: int = Field(default=4, ge=1)
    timeout_seconds: int = Field(default=300, ge=1)


class OutputConfig(BaseModel):
    """Output configuration"""
    default_format: Literal["text", "json", "markdown"] = Field(default="text")
    save_results: bool = Field(default=True)
    generate_visualizations: bool = Field(default=False)
    verbose: bool = Field(default=False)


class DataStoreConfig(BaseModel):
    """DataStore configuration"""
    type: str = Field(default="filesystem", description="DataStore type (filesystem, memory, postgresql, etc.)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for DataStore initialization")


# New Pydantic models for cleaner config structure
class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: Literal["local", "openai", "anthropic", "mock", "clean"] = Field(default="local")
    model: str = Field(default="distilgpt2", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    api_key: Optional[str] = Field(default=None, description="API key for external providers")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")


class EmbeddingConfig(BaseModel):
    """Embedding configuration"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = Field(default=384, ge=1)
    device: str = Field(default="cpu")


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = Field(default=False)
    performance_tracking: bool = Field(default=False)
    detailed_tracing: bool = Field(default=False)
    metrics_port: int = Field(default=9090)


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    file_path: str = Field(default="/Users/miyauchikazuyoshi/.insightspike/logs")
    log_to_console: bool = Field(default=False)
    max_size_mb: int = Field(default=50)
    backup_count: int = Field(default=3)


class InsightSpikeConfig(BaseModel):
    """Complete InsightSpike configuration - new structure"""
    # Top-level settings
    environment: Literal["development", "experiment", "production", "research", "test", "testing", "custom"] = Field(default="development")
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Keep the old fields for backward compatibility
    core: Optional[CoreConfig] = Field(default=None, exclude=True)
    retrieval: Optional[RetrievalConfig] = Field(default=None, exclude=True)
    reasoning: Optional[ReasoningConfig] = Field(default=None, exclude=True)
    spike: Optional[SpikeConfig] = Field(default=None, exclude=True)
    unknown_learner: Optional[UnknownLearnerConfig] = Field(default=None, exclude=True)
    paths: PathsConfig = Field(default_factory=PathsConfig)  # Keep this for backward compatibility
    processing: Optional[ProcessingConfig] = Field(default=None, exclude=True)
    output: Optional[OutputConfig] = Field(default=None, exclude=True)
    datastore: Optional[DataStoreConfig] = Field(default=None, exclude=True)
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"  # Prevent unknown fields
        
    def to_legacy_config(self) -> "Config":
        """Convert to legacy Config format for backward compatibility"""
        from .legacy_config import (
            Config, EmbeddingConfig, LLMConfig, MemoryConfig as LegacyMemoryConfig,
            RetrievalConfig as LegacyRetrievalConfig, GraphConfig as LegacyGraphConfig,
            ReasoningConfig as LegacyReasoningConfig, PathConfig, ProcessingConfig as LegacyProcessingConfig
        )
        
        return Config(
            embedding=EmbeddingConfig(
                model_name=self.core.model_name,
                dimension=384,  # Default for MiniLM
                device=self.core.device
            ),
            llm=LLMConfig(
                provider=self.core.llm_provider,
                model_name=self.core.llm_model,
                max_tokens=self.core.max_tokens,
                temperature=self.core.temperature,
                device=self.core.device
            ),
            memory=LegacyMemoryConfig(
                max_episodes=self.memory.episodic_memory_capacity,
                merge_threshold=self.reasoning.episode_merge_threshold,
                split_threshold=self.reasoning.episode_split_threshold,
                prune_threshold=self.reasoning.episode_prune_threshold
            ),
            retrieval=LegacyRetrievalConfig(
                top_k=self.retrieval.top_k,
                similarity_threshold=self.retrieval.similarity_threshold
            ),
            graph=LegacyGraphConfig(
                spike_threshold_ged=self.graph.spike_ged_threshold,
                spike_threshold_ig=self.graph.spike_ig_threshold,
                use_gpu=self.core.use_gpu
            ),
            reasoning=LegacyReasoningConfig(
                max_reasoning_steps=5,
                convergence_threshold=0.9,
                weight_ged=self.reasoning.weight_ged,
                weight_ig=self.reasoning.weight_ig
            ),
            paths=PathConfig(
                root_dir=Path("."),
                data_dir=self.paths.data_dir,
                log_dir=self.paths.log_dir,
                cache_dir=Path("data/cache")
            ),
            processing=LegacyProcessingConfig(
                batch_size=self.processing.batch_size,
                num_workers=self.processing.max_workers,
                use_multiprocessing=False,
                timeout=self.processing.timeout_seconds
            ),
            safe_mode=self.core.safe_mode
        )