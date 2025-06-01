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
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"

@dataclass
class LLMConfig:
    """Configuration for language models"""
    provider: str = "local"
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    max_tokens: int = 256
    temperature: float = 0.7
    device: str = "cpu"
    use_gpu: bool = False
    safe_mode: bool = False  # Use mock provider to avoid model loading issues

@dataclass
class RetrievalConfig:
    """Configuration for retrieval layers"""
    similarity_threshold: float = 0.35
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

@dataclass
class ReasoningConfig:
    """Configuration for reasoning layer"""
    similarity_threshold: float = 0.3
    spike_ged_threshold: float = 0.5
    spike_ig_threshold: float = 0.2
    conflict_threshold: float = 0.6
    use_gnn: bool = False
    weight_ged: float = 1.0
    weight_ig: float = 1.0
    weight_conflict: float = 0.5
    gnn_hidden_dim: int = 64
    graph_file: str = "data/graph_pyg.pt"

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
        if self.unknown_learner is None:
            self.unknown_learner = UnknownLearnerConfig()

def get_config() -> Config:
    """Get the default configuration"""
    return Config()

# Legacy compatibility - export individual values for backward compatibility
def get_legacy_config():
    """Get configuration values in legacy format"""
    config = get_config()
    return {
        'ROOT_DIR': config.paths.root_dir,
        'DATA_DIR': config.paths.data_dir,
        'LOG_DIR': config.paths.log_dir,
        'INDEX_FILE': config.paths.index_file,
        'GRAPH_FILE': config.paths.graph_file,
        'EMBED_MODEL_NAME': config.embedding.model_name,
        'LLM_NAME': config.llm.model_name,
        'SIM_THRESHOLD': config.retrieval.similarity_threshold,
        'TOP_K': config.retrieval.top_k,
        'LAYER1_TOP_K': config.retrieval.layer1_top_k,
        'LAYER2_TOP_K': config.retrieval.layer2_top_k,
        'LAYER3_TOP_K': config.retrieval.layer3_top_k,
        'SPIKE_GED': config.spike.spike_ged,
        'SPIKE_IG': config.spike.spike_ig,
        'ETA_SPIKE': config.spike.eta_spike,
        'MERGE_GED': config.memory.merge_ged,
        'SPLIT_IG': config.memory.split_ig,
        'PRUNE_C': config.memory.prune_c,
        'INACTIVE_N': config.memory.inactive_n,
    }
