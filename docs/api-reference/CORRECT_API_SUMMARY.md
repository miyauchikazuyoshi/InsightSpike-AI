# Correct API Summary for InsightSpike-AI

Based on the actual implementation, here are the correct constructors and methods:

## Quick Start Functions

```python
from insightspike.public import create_agent, quick_demo, InsightAppWrapper

# Create agent with minimal configuration
agent = create_agent(provider="openai")  # or "mock", "anthropic"

# Run interactive demo
quick_demo()
```

## Local App Wrapper (InsightAppWrapper)

```python
app = InsightAppWrapper(provider="mock")
app.learn("geDIG is a graph distance signal.")
answer = app.ask("What is geDIG?", max_cycles=1)
```

## 1. EnvironmentState (from insightspike.core.interfaces.generic_interfaces)

```python
@dataclass
class EnvironmentState:
    # Required parameters
    state_data: Union[np.ndarray, Dict[str, Any], List, Tuple]
    environment_type: str
    task_type: TaskType
    
    # Optional parameters
    state_shape: Optional[Tuple[int, ...]] = None
    state_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    step_count: int = 0
    episode_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
```

## 2. GenericReasoner (from insightspike.core.agents.generic_agent)

```python
class GenericReasoner(ReasonerInterface):
    # No constructor parameters - it's a simple implementation
    def __init__(self):
        pass  # No parameters needed
```

## 3. L2MemoryManager (from insightspike.implementations.layers.layer2_memory_manager)

```python
class L2MemoryManager(L2MemoryInterface):
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_model: Optional[Any] = None
    ):
    
    # New memory management methods:
    def age_episodes(self) -> int:
        """Apply time-based aging to episodes"""
        
    def enforce_size_limit(self) -> int:
        """Enforce maximum episode limit"""
        
    def merge_episodes(self, indices: List[int]) -> int:
        """Merge episodes, auto-detects similar pairs if indices not specified"""
```

## 4. L3GraphReasoner (from insightspike.implementations.layers.layer3_graph_reasoner)

```python
class L3GraphReasoner(L3GraphReasonerInterface):
    def __init__(self, config=None):
        # Internally calls super().__init__("layer3_graph_reasoner", config)
        # Sets layer_id = "layer3_graph_reasoner"
```

## 5. CLI Commands (from insightspike.cli.spike)

Main commands:
- `query` - Ask a question to the AI agent (replaces deprecated 'ask')
- `embed` - Load documents or text into memory
- `stats` - Show memory and agent statistics
- `config` - Configuration management (show, export, validate)
- `chat` - Interactive chat mode
- `discover` - Discover insights from knowledge base
- `bridge` - Bridge concepts across domains
- `graph` - Graph visualization and analysis

Removed/Deprecated commands:
- `ask` → use `query`
- 13 legacy experiment commands removed
- `test_safe` and other debug commands removed

## Key Differences from Test Assumptions:

1. **EnvironmentState**: Uses `@dataclass`, not a regular class constructor
2. **GenericReasoner**: Takes no parameters in constructor
3. **L2MemoryManager**: All parameters are optional
4. **L3GraphReasoner**: Only takes optional config parameter
5. **CLI**: No `rag` command exists, but there are many insight-related commands

## 6. Memory Configuration (new in July 2025)

```python
@dataclass
class MemoryConfig:
    # Core settings
    embedding_dim: int = 384
    max_episodes: int = 10000
    
    # Memory aging settings (NEW)
    enable_aging: bool = True
    aging_factor: float = 0.95  # Decay per day
    min_age_days: int = 7
    max_age_days: int = 90
    prune_on_overflow: bool = True
```

## 7. Integrated Vector-Graph Index (NEW - January 2025)

```python
from insightspike.index import IntegratedVectorGraphIndex, BackwardCompatibleWrapper

# Direct usage
index = IntegratedVectorGraphIndex(dimension=768)

# Add episode with pre-computed normalization
episode = {
    'vec': np.random.randn(768),
    'text': 'Sample text',
    'pos': (10, 20),  # Optional spatial position
    'c_value': 0.8
}
idx = index.add_episode(episode)

# Search with O(1) performance
indices, scores = index.search(query_vector, k=10)

# Backward compatible usage
datastore = BackwardCompatibleWrapper(index)
datastore.save_episodes(episodes)  # Works with existing API
results = datastore.search_vectors(query, k=10)
```

### Key Features:
- **O(1) vector search**: Pre-normalized vectors eliminate normalization bottleneck
- **Spatial indexing**: O(log n) position-based queries for navigation tasks  
- **Graph integration**: NetworkX graph with similarity-based edges
- **FAISS auto-switching**: Automatic optimization for large datasets
- **100% backward compatible**: Drop-in replacement for existing DataStore

## 8. Enhanced DataStore with Integrated Index

```python
from insightspike.implementations.datastore import EnhancedFileSystemDataStore

# Enable integrated index
datastore = EnhancedFileSystemDataStore(
    root_path="./data",
    use_integrated_index=True,
    migration_mode="shadow"  # shadow, partial, or full
)

# Existing code works without changes
episodes = datastore.load_episodes()
results = datastore.search_vectors(query, k=10)

# Monitor performance
from insightspike.monitoring import IndexMonitoringDecorator

monitored = IndexMonitoringDecorator(datastore.index)
metrics = monitored.get_metrics()
print(f"Avg search time: {metrics['search_time']['avg_ms']:.2f}ms")
```

## Notes:

- Import paths changed: `core.layers.*` → `implementations.layers.*`
- Configuration now uses Pydantic models, not legacy dict format
- Memory manager includes automatic aging and size management
- CLI simplified to essential commands only
- Deprecated methods removed (save_graph, load_graph, _detect_spike, etc.)

---

## 9. GeDIG Refactor (August 2025)

### Factory & Core (Linkset‑First)
```python
from insightspike.algorithms.gedig_factory import GeDIGFactory, dual_evaluate
from insightspike.algorithms.linkset_adapter import build_linkset_info

core = GeDIGFactory.create({'use_refactored_gedig': True})
ls = build_linkset_info(
    s_link=[{"index": 1, "similarity": 1.0}],
    candidate_pool=[],
    decision={"index": 1, "similarity": 1.0},
    query_vector=[1.0],
    base_mode="link",
)
result = core.calculate(g_prev=g1, g_now=g2, linkset_info=ls)

# Dual evaluation (regression monitoring)
legacy = GeDIGFactory.create({'use_refactored_gedig': False})
ref    = GeDIGFactory.create({'use_refactored_gedig': True})
ref_res, delta = dual_evaluate(legacy, ref, g_prev=g1, g_now=g2)
```

Note: Calling `calculate(...)` without `linkset_info` falls back to graph‑IG and now emits a one‑time deprecation warning. Prefer passing linkset info for paper‑aligned IG.

### Result Key Fields
`raw_ged`, `ged_value`, `structural_cost`, `structural_improvement` (alias = `-structural_cost`), `ig_raw`, `ig_z_score`, `hop0_reward`, `aggregate_reward`, `reward`, `hop_results` (opt), `version`.

### Logger
```python
from insightspike.algorithms.gedig_core import GeDIGLogger
logger = GeDIGLogger("logs/gedig/gedig_metrics.csv")
core.logger = logger  # attach
```

### Reward
```
R_hop0 = λ * Z(IG_raw) - μ * structural_cost
R_agg  = λ * Z(IG_raw) - μ * Σ_w (w_h * Cost_h)/Σ_w w_h
```

### Warmup
`λ=0` for initial `warmup_steps` to stabilize z-score statistics.

### Upcoming (Phase C)
- SpikeDetectionMode(AND/OR/THRESHOLD)
- Preset parameter bundles
- False positive monitor

### Day3 Additions

```python
from insightspike.algorithms.gedig_analysis import analyze_divergence
stats = analyze_divergence('divergence.csv', threshold=0.3)
print(stats.to_dict())

from insightspike.algorithms.gedig_core import GeDIGLogger
logger = GeDIGLogger('logs/gedig_metrics.csv', compress_on_rotate=True)

# Maze navigator spike outcome modes (maze config)
# spike_outcome_mode: 'mirror' | 'structural_positive' | 'spike_threshold'
```

Features:

- Divergence log統計: 平均Δ / 最大Δ / 閾値超過率.
- ロガー gzip 圧縮オプション `compress_on_rotate`.
- Spike ground-truth 擬似導出モード（評価指標改善用）。
