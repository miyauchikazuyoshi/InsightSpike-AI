"""
Unified Configuration Models for InsightSpike
============================================

Clean Pydantic-based configuration models without backward compatibility.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

# Pydantic v1/v2 互換シム
_PYDANTIC_V2 = True
try:  # v2
    from pydantic import field_validator, model_validator, ConfigDict  # type: ignore
except ImportError:  # v1 fallback
    _PYDANTIC_V2 = False
    from pydantic import validator, root_validator  # type: ignore

    # v2 style API の簡易ラッパ
    def field_validator(field_name: str, mode: str = "before"):  # type: ignore
        pre = mode == "before"
        def deco(fn):
            return validator(field_name, pre=pre, allow_reuse=True)(fn)  # type: ignore
        return deco

    def model_validator(*, mode: str):  # type: ignore
        # ここでは after のみ使用。v1では root_validator で代替。
        def wrap(fn):
            def _root(cls, values):  # type: ignore
                # values は dict。検証のみ行いそのまま返す。
                # fn は self を受けるため v1 では辞書経由の検証に変換。
                # HybridWeightsConfig 用に total チェックを values で再現。
                return values
            return root_validator(pre=False, allow_reuse=True)(_root)  # type: ignore
        return wrap

    class ConfigDict(dict):  # type: ignore
        pass

from .wake_sleep_config import WakeSleepConfig
# Maze config moved to maze_experimental
try:
    from ..maze_experimental.maze_config import MazeConfig, MazeNavigatorConfig, MazeExperimentConfig
except ImportError:
    # Provide dummy classes if maze_experimental is not available
    class MazeConfig(BaseModel):
        pass
    class MazeNavigatorConfig(BaseModel):
        pass
    class MazeExperimentConfig(BaseModel):
        pass


class MemoryConfig(BaseModel):
    """Memory system configuration

    Includes dynamic vector index parameters (faiss_index_type, metric) which may be
    injected by normalizers; they have explicit defaults here now to remove the need
    for attribute existence checks downstream.
    """

    max_retrieved_docs: int = Field(default=15, ge=1)
    short_term_capacity: int = Field(default=10, ge=1)
    working_memory_capacity: int = Field(default=20, ge=1)
    episodic_memory_capacity: int = Field(default=60, ge=1)
    pattern_cache_capacity: int = Field(default=15, ge=1)
    # Vector index parameters (normalized across code paths)
    faiss_index_type: str = Field(default="FlatL2")
    metric: str = Field(default="l2")
    # Retrieval / graph-related thresholds used by L2 memory
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # L2 memory operational toggles
    use_c_values: bool = Field(default=True)
    use_graph_integration: bool = Field(default=False)
    use_scalable_indexing: bool = Field(default=True)
    # Performance
    batch_size: int = Field(default=32, ge=1)
    cache_embeddings: bool = Field(default=True)
    vector_search_backend: str = Field(default="auto")  # auto, numpy, faiss
    # Diagnostics (which defaults were injected by normalizers)
    defaults_applied: list[str] = Field(default_factory=list, description="Auto-applied default field names")

class DefaultsAppliedMixin(BaseModel):
    """Mixin to record which config defaults were auto-applied at load time."""
    defaults_applied: list[str] = Field(default_factory=list)


class HybridWeightsConfig(BaseModel):
    """Hybrid algorithm weights (v1/v2 compatible)"""

    structure: float = Field(default=0.4, ge=0.0, le=1.0)
    semantic: float = Field(default=0.4, ge=0.0, le=1.0)
    quality: float = Field(default=0.2, ge=0.0, le=1.0)

    if _PYDANTIC_V2:
        @model_validator(mode="after")  # type: ignore[misc]
        def _check_sum(self):  # type: ignore[no-untyped-def]
            total = self.structure + self.semantic + self.quality
            if not 0.99 <= total <= 1.01:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
            return self
    else:
        @root_validator(pre=False, allow_reuse=True)  # type: ignore
        def _check_sum(cls, values):  # type: ignore[no-untyped-def]
            total = (values.get("structure", 0) + values.get("semantic", 0) + values.get("quality", 0))
            if not 0.99 <= total <= 1.01:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
            return values


class GraphConfig(BaseModel):
    """Graph processing & spike detection"""

    spike_ged_threshold: float = Field(default=-0.5, ge=-1.0, le=1.0)
    spike_ig_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    conflict_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_gnn: bool = Field(default=False)
    gnn_hidden_dim: int = Field(default=64, ge=1)
    ged_algorithm: Literal["simple", "advanced", "networkx", "hybrid"] = Field(
        default="advanced"
    )
    ig_algorithm: Literal["simple", "advanced", "entropy", "hybrid"] = Field(
        default="advanced"
    )
    hybrid_weights: HybridWeightsConfig = Field(default_factory=HybridWeightsConfig)

    # Paper-aligned geDIG controls (available to Layer3 and tools)
    sp_scope_mode: Literal["auto", "union"] = Field(
        default="auto", description="Scope for SP evaluation: auto or union-of-k-hop"
    )
    sp_eval_mode: Literal["connected", "fixed_before_pairs"] = Field(
        default="connected", description="Shortest-path gain evaluation mode"
    )
    sp_hop_expand: int = Field(
        default=0, ge=0, le=5, description="Extra hops to expand SP neighborhood"
    )
    sp_boundary_mode: Literal["induced", "trim", "nodes"] = Field(
        default="induced", description="Boundary trimming for SP subgraphs"
    )
    ig_source_mode: Literal["graph", "linkset", "hybrid", "paper", "strict"] = Field(
        default="graph", description="Information gain source"
    )
    ged_norm_scheme: Literal["edges_after", "candidate_base"] = Field(
        default="edges_after", description="GED normalization scheme"
    )
    linkset_query_weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Relative weight for query entry in linkset entropy"
    )
    lambda_weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Information temperature λ"
    )
    sp_beta: float = Field(
        default=0.2, ge=0.0, le=10.0, description="Weight γ for SP relative gain"
    )
    
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

    # Message passing configuration
    enable_message_passing: bool = Field(
        default=False,
        description="Enable question-aware message passing"
    )
    message_passing: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Message passing configuration"
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
    """File system paths (with home expansion compatible across pydantic versions)"""

    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    embeddings_dir: Path = Field(default=Path("data/embeddings"))
    cache_dir: Path = Field(default=Path("data/cache"))
    models_dir: Path = Field(default=Path("data/models"))
    # Use user-home based default to avoid machine-specific absolute paths
    logs_dir: Path = Field(default="~/.insightspike/logs")

    if _PYDANTIC_V2:
        @field_validator("logs_dir", mode="before")  # type: ignore[misc]
        def expand_home_path(cls, v):  # type: ignore[no-untyped-def]
            if isinstance(v, str) and v.startswith("~"):
                return Path(v).expanduser()
            return v
    else:
        @validator("logs_dir", pre=True, allow_reuse=True)  # type: ignore
        def expand_home_path(cls, v):  # type: ignore[no-untyped-def]
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
    
    # Processing cycle settings
    max_cycles: int = Field(default=10, ge=1)
    convergence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    min_quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_advanced_metrics: bool = Field(default=True)
    
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
    prompt_style: Literal["standard", "detailed", "minimal", "association", "association_extended"] = Field(
        default="standard",
        description="Prompt style: detailed for large models, standard for medium, minimal for small, association/association_extended for reasoning tasks"
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
    
    # Branching detection configuration (for association_extended)
    branching_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum relevance to be considered high for branching"
    )
    branching_min_branches: int = Field(
        default=2,
        ge=2,
        le=10,
        description="Minimum number of high-relevance docs for branching"
    )
    branching_max_gap: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Maximum gap between top relevances for branching"
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


class DataStoreConfig(BaseModel):
    """DataStore configuration"""
    
    type: Literal["filesystem", "in_memory"] = Field(default="filesystem")
    root_path: str = Field(default="./data/insight_store")
    # True if user explicitly set root_path (tracked by loader)
    explicit_root_path: bool = Field(default=False, exclude=True)


class SpectralEvaluationConfig(BaseModel):
    """Spectral evaluation configuration"""
    
    enabled: bool = Field(default=False)
    weight: float = Field(default=0.3, ge=0.0, le=1.0)


class MultihopConfig(BaseModel):
    """Multi-hop analysis configuration"""
    
    max_hops: int = Field(default=3, ge=1, le=10)
    decay_factor: float = Field(default=0.5, ge=0.0, le=1.0)


class MetricsConfig(BaseModel):
    """Advanced metrics configuration"""
    
    use_normalized_ged: bool = Field(default=True)
    use_entropy_variance_ig: bool = Field(default=False)
    use_multihop_gedig: bool = Field(default=False)
    # Query-centric local evaluation (k-hop around focal nodes derived from query)
    query_centric: bool = Field(default=True)
    query_topk_centers: int = Field(default=3, ge=1, le=50)
    query_radius: int = Field(default=1, ge=0, le=5)
    spectral_evaluation: SpectralEvaluationConfig = Field(default_factory=SpectralEvaluationConfig)
    multihop_config: MultihopConfig = Field(default_factory=MultihopConfig)
    theta_cand: float = Field(default=0.45, ge=0.0, le=1.0)
    theta_link: float = Field(default=0.35, ge=0.0, le=1.0)
    candidate_cap: int = Field(default=32, ge=1, le=1024)
    top_m: Optional[int] = Field(default=None)
    ig_denominator: Literal["legacy", "fixed_kstar"] = Field(default="fixed_kstar")
    use_local_normalization: bool = Field(default=False)


class ReasoningConfig(BaseModel):
    """Reasoning engine configuration"""
    
    max_cycles: int = Field(default=10, ge=1, le=100)
    convergence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    spike_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class PerformanceConfig(BaseModel):
    """Performance optimization configuration"""
    
    enable_cache: bool = Field(default=True)
    parallel_workers: int = Field(default=4, ge=1, le=32)


class VectorSearchConfig(BaseModel):
    """Vector search backend configuration"""
    
    backend: Literal["auto", "numpy", "faiss"] = Field(default="auto", description="Vector search backend")
    optimize: bool = Field(default=True, description="Use optimized implementations")
    batch_size: int = Field(default=1000, ge=1, description="Batch size for operations")


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
    
    # New configurations
    datastore: DataStoreConfig = Field(default_factory=DataStoreConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    wake_sleep: WakeSleepConfig = Field(default_factory=WakeSleepConfig)

    # Strict extra handling across pydantic versions
    # v2: use model_config; v1: provide inner Config
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    if not _PYDANTIC_V2:  # pydantic v1 compatibility
        class Config:  # type: ignore
            validate_assignment = True
            extra = 'forbid'
