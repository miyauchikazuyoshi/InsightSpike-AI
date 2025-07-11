"""
Configuration system for InsightSpike-AI
=======================================

Structured configuration with proper object hierarchy for the new architecture.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""

    model_name: str = "paraphrase-MiniLM-L6-v2"  # More stable on older CPUs
    dimension: int = 384
    device: str = "cpu"


@dataclass
class LLMConfig:
    """Configuration for language models"""

    provider: str = "local"
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Chat特化版でRAG評価に最適
    max_tokens: int = 256
    temperature: float = 0.3  # RAG評価用に低温度で安定した応答
    device: str = "cpu"
    use_gpu: bool = False
    safe_mode: bool = False  # Use mock provider to avoid model loading issues
    
    # Direct generation mode settings
    use_direct_generation: bool = False  # Enable direct response generation
    direct_generation_threshold: float = 0.7  # Reasoning quality threshold
    fallback_to_llm: bool = True  # Fallback to LLM if direct generation fails
    
    # Layer 4 pipeline settings
    use_layer4_pipeline: bool = False  # Use new Layer 4 pipeline architecture
    enable_polish: bool = True  # Enable Layer 4.1 polish
    polish_threshold: float = 0.6  # Confidence threshold for polishing
    always_polish_below: float = 0.4  # Always polish below this confidence
    force_direct_generation: bool = False  # Force direct mode in pipeline


@dataclass
class RetrievalConfig:
    """Configuration for retrieval layers"""

    similarity_threshold: float = 0.25
    top_k: int = 15
    layer1_top_k: int = 20
    layer2_top_k: int = 15
    layer3_top_k: int = 12


@dataclass
class SpikeConfig:
    """Configuration for Eureka spike detection"""

    spike_ged: float = 0.5
    spike_ig: float = 0.2
    eta_spike: float = 0.2


@dataclass
class GraphConfig:
    """Configuration for graph processing and spike detection"""

    spike_ged_threshold: float = 0.5
    spike_ig_threshold: float = 0.2
    ged_algorithm: str = "advanced"  # Options: simple, advanced, networkx
    ig_algorithm: str = "advanced"   # Options: simple, advanced, entropy


@dataclass
class ReasoningConfig:
    """Configuration for reasoning layer"""

    similarity_threshold: float = 0.2
    spike_ged_threshold: float = 0.5
    spike_ig_threshold: float = 0.2
    conflict_threshold: float = 0.6
    use_gnn: bool = False
    weight_ged: float = 1.0
    weight_ig: float = 1.0
    weight_conflict: float = 0.5
    gnn_hidden_dim: int = 64
    graph_file: str = "data/graph_pyg.pt"
    
    # Episode management thresholds based on graph analysis
    episode_merge_threshold: float = 0.8  # High similarity threshold for merging
    episode_split_threshold: float = 0.3  # High conflict threshold for splitting  
    episode_prune_threshold: float = 0.1  # Low C-value threshold for pruning
    
    # Episode integration thresholds for new episodes
    episode_integration_similarity_threshold: float = 0.85  # Vector similarity threshold
    episode_integration_content_threshold: float = 0.4     # Content overlap threshold (lowered)
    episode_integration_c_threshold: float = 0.3           # C-value difference threshold
    
    # Scalable graph configuration
    use_scalable_graph: bool = True  # Enable FAISS-based scalable graph
    graph_top_k: int = 50  # Maximum neighbors per node
    graph_batch_size: int = 1000  # Batch size for graph operations
    conflict_split_threshold: int = 2  # Number of conflicts to trigger split
    use_advanced_metrics: bool = True  # Use advanced GED/IG algorithms
    
    # GED/IG Algorithm Selection (also available in GraphConfig)
    ged_algorithm: str = "advanced"  # Options: simple, advanced, networkx
    ig_algorithm: str = "advanced"   # Options: simple, advanced, entropy


@dataclass
class MemoryConfig:
    """Configuration for memory management"""

    merge_ged: float = 0.4
    split_ig: float = -0.15
    prune_c: float = 0.05
    inactive_n: int = 30
    max_retrieved_docs: int = 15
    min_similarity: float = 0.3
    nlist: int = 256
    pq_segments: int = 16
    c_value_gamma: float = 0.5
    c_value_min: float = 0.0
    c_value_max: float = 1.0
    index_file: str = "data/index.faiss"


@dataclass
class PathConfig:
    """Configuration for file paths"""

    root_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = None
    log_dir: Path = None
    index_file: Path = None
    graph_file: Path = None

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.root_dir / "data" / "raw"
        if self.log_dir is None:
            self.log_dir = self.root_dir / "data" / "logs"
        if self.index_file is None:
            self.index_file = self.root_dir / "data" / "index.faiss"
        if self.graph_file is None:
            self.graph_file = self.root_dir / "data" / "graph_pyg.pt"

        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ScalableGraphConfig:
    """Configuration for scalable graph operations"""
    
    enabled: bool = True  # Enable scalable graph features
    faiss_index_type: str = "IndexFlatIP"  # FAISS index type
    top_k_neighbors: int = 50  # Maximum neighbors per node
    batch_size: int = 1000  # Batch size for processing
    similarity_threshold: float = 0.2  # Minimum similarity for edge creation
    conflict_threshold: float = 0.8  # Similarity threshold for conflict detection
    
    # Graph importance calculation
    use_graph_importance: bool = True  # Use graph-based importance instead of C-values
    importance_decay_factor: float = 0.1  # Time decay for access-based importance
    pagerank_alpha: float = 0.85  # PageRank damping factor
    importance_cache_validity: int = 300  # Cache validity in seconds
    
    # Conflict-based splitting
    enable_conflict_split: bool = True  # Enable automatic conflict-based splitting
    min_conflicts_for_split: int = 2  # Minimum conflicts to trigger split
    split_text_min_length: int = 50  # Minimum text length for meaningful split
    
    # Performance optimization
    faiss_nprobe: int = 10  # Number of clusters to search (for IVF indices)
    faiss_nlist: int = 100  # Number of clusters (for IVF indices)
    incremental_update: bool = True  # Use incremental graph updates


@dataclass
class UnknownLearnerConfig:
    """Configuration for unknown information learning system"""

    initial_confidence: float = 0.1
    cleanup_threshold: float = 0.15
    confidence_boost: float = 0.05
    max_weak_edges: int = 1000
    cleanup_interval: int = 300  # seconds


@dataclass
class Config:
    """Main configuration object"""

    environment: str = "local"
    embedding: EmbeddingConfig = None
    llm: LLMConfig = None
    retrieval: RetrievalConfig = None
    spike: SpikeConfig = None
    graph: GraphConfig = None
    reasoning: ReasoningConfig = None
    memory: MemoryConfig = None
    paths: PathConfig = None
    scalable_graph: ScalableGraphConfig = None
    unknown_learner: UnknownLearnerConfig = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.spike is None:
            self.spike = SpikeConfig()
        if self.graph is None:
            self.graph = GraphConfig()
        if self.reasoning is None:
            self.reasoning = ReasoningConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.paths is None:
            self.paths = PathConfig()
        if self.scalable_graph is None:
            self.scalable_graph = ScalableGraphConfig()
        if self.unknown_learner is None:
            self.unknown_learner = UnknownLearnerConfig()
    
    @property
    def gnn(self):
        """Property for backward compatibility - GNN config is in reasoning"""
        return self.reasoning


def get_config() -> Config:
    """Get the default configuration"""
    return Config()


# Legacy compatibility - export individual values for backward compatibility
def get_legacy_config():
    """Get configuration values in legacy format"""
    config = get_config()
    return {
        "ROOT_DIR": config.paths.root_dir,
        "DATA_DIR": config.paths.data_dir,
        "LOG_DIR": config.paths.log_dir,
        "INDEX_FILE": config.paths.index_file,
        "GRAPH_FILE": config.paths.graph_file,
        "EMBED_MODEL_NAME": config.embedding.model_name,
        "LLM_NAME": config.llm.model_name,
        "SIM_THRESHOLD": config.retrieval.similarity_threshold,
        "TOP_K": config.retrieval.top_k,
        "LAYER1_TOP_K": config.retrieval.layer1_top_k,
        "LAYER2_TOP_K": config.retrieval.layer2_top_k,
        "LAYER3_TOP_K": config.retrieval.layer3_top_k,
        "SPIKE_GED": config.spike.spike_ged,
        "SPIKE_IG": config.spike.spike_ig,
        "ETA_SPIKE": config.spike.eta_spike,
        "MERGE_GED": config.memory.merge_ged,
        "SPLIT_IG": config.memory.split_ig,
        "PRUNE_C": config.memory.prune_c,
        "INACTIVE_N": config.memory.inactive_n,
    }
