"""
Unified Configuration Models for InsightSpike
============================================

Clean Pydantic-based configuration models without backward compatibility.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, validator


class MemoryConfig(BaseModel):
    """Memory system configuration"""

    max_retrieved_docs: int = Field(default=15, ge=1)
    short_term_capacity: int = Field(default=10, ge=1)
    working_memory_capacity: int = Field(default=20, ge=1)
    episodic_memory_capacity: int = Field(default=60, ge=1)
    pattern_cache_capacity: int = Field(default=15, ge=1)


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

    spike_ged_threshold: float = Field(default=0.5, ge=-1.0, le=1.0)
    spike_ig_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    use_gnn: bool = Field(default=False)
    gnn_hidden_dim: int = Field(default=64, ge=1)
    ged_algorithm: Literal["simple", "advanced", "networkx", "hybrid"] = Field(
        default="hybrid"
    )
    ig_algorithm: Literal["simple", "advanced", "entropy", "hybrid"] = Field(
        default="hybrid"
    )
    hybrid_weights: HybridWeightsConfig = Field(default_factory=HybridWeightsConfig)
    
    # Graph-based search configuration
    enable_graph_search: bool = Field(
        default=False,
        description="Enable multi-hop graph traversal for memory search"
    )
    hop_limit: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Maximum hops for graph traversal"
    )
    neighbor_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for neighbor inclusion"
    )
    path_decay: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Relevance decay per hop in graph traversal"
    )

    # geDIG formula parameters
    weight_ged: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for ΔGED in reward formula"
    )
    weight_ig: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for ΔIG in reward formula"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Temperature parameter for exploration-exploitation balance",
    )


class PathsConfig(BaseModel):
    """File system paths"""

    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    embeddings_dir: Path = Field(default=Path("data/embeddings"))
    cache_dir: Path = Field(default=Path("data/cache"))
    models_dir: Path = Field(default=Path("data/models"))
    logs_dir: Path = Field(default=Path("/Users/miyauchikazuyoshi/.insightspike/logs"))

    @validator("logs_dir", pre=True)
    def expand_home_path(cls, v):
        if isinstance(v, str) and v.startswith("~"):
            return Path(v).expanduser()
        return v


class ProcessingConfig(BaseModel):
    """Processing configuration"""

    batch_size: int = Field(default=32, ge=1)
    max_workers: int = Field(default=4, ge=1)
    chunk_size: int = Field(default=500, ge=50)
    overlap: int = Field(default=50, ge=0)
    min_chunk_size: int = Field(default=100, ge=10)
    
    # Layer1 bypass configuration
    enable_layer1_bypass: bool = Field(
        default=False,
        description="Enable fast path for known concepts with low uncertainty"
    )
    bypass_uncertainty_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Uncertainty threshold below which bypass is triggered"
    )
    bypass_known_ratio_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Ratio of known elements required for bypass"
    )
    
    # Insight auto-registration configuration
    enable_insight_registration: bool = Field(
        default=True,
        description="Enable automatic insight registration when spikes are detected"
    )
    enable_insight_search: bool = Field(
        default=True,
        description="Enable searching and using insights in memory retrieval"
    )
    max_insights_per_query: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum number of insights to retrieve per query"
    )
    
    # Prompt length optimization
    dynamic_doc_adjustment: bool = Field(
        default=True,
        description="Reduce document count when insights are present"
    )
    max_docs_with_insights: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum documents when insights are included"
    )
    insight_relevance_boost: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Relevance score boost for insights"
    )
    
    # Learning mechanism configuration
    enable_learning: bool = Field(
        default=False,
        description="Enable adaptive learning from patterns and rewards"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Learning rate for strategy optimization"
    )
    exploration_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=0.3,
        description="Initial exploration rate for strategy discovery"
    )


class LLMConfig(BaseModel):
    """Language Model configuration"""

    provider: Literal["local", "openai", "anthropic", "ollama", "mock"] = Field(
        default="local", description="LLM provider to use"
    )
    model: str = Field(
        default="distilgpt2", description="Model name (provider-specific)"
    )
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    api_key: Optional[str] = Field(default=None, exclude=True)
    api_base: Optional[str] = Field(default=None)
    organization: Optional[str] = Field(default=None, exclude=True)
    device: str = Field(default="cpu", description="Device for local models")
    load_in_8bit: bool = Field(default=False, description="8-bit quantization")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for the model"
    )
    
    # Prompt building configuration
    prompt_style: Literal["standard", "detailed", "minimal"] = Field(
        default="standard",
        description="Prompt style: detailed for large models, standard for medium, minimal for small"
    )
    max_context_docs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum documents in context (adjusted by prompt style)"
    )
    use_simple_prompt: bool = Field(
        default=False,
        description="Use simplified prompt format for lightweight models"
    )
    include_metadata: bool = Field(
        default=False,
        description="Include relevance scores and metadata in detailed mode"
    )


class OutputConfig(BaseModel):
    """Output configuration"""

    format: Literal["json", "markdown", "html"] = Field(default="json")
    include_reasoning: bool = Field(default=True)
    include_sources: bool = Field(default=True)
    max_sources: int = Field(
        default=5, ge=1, description="Maximum number of sources to include"
    )
    max_context_length: int = Field(
        default=2000,
        ge=100,
        description="Maximum number of characters to include in context",
    )
    max_documents: int = Field(
        default=10,
        ge=1,
        description="Maximum number of documents to include in context",
    )
    include_metadata: bool = Field(
        default=True, description="Include relevance scores and graph analysis"
    )


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
    """Complete InsightSpike configuration - clean structure without backward compatibility"""

    # Top-level settings
    environment: Literal[
        "development",
        "experiment",
        "production",
        "research",
        "test",
        "testing",
        "custom",
    ] = Field(default="development")
    pre_warm_models: bool = Field(
        default=True,
        description="Pre-warm LLM models on startup for faster experiment initialization",
    )

    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    class Config:
        """Pydantic configuration"""

        validate_assignment = True
        extra = "forbid"  # Prevent unknown fields
