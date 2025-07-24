# Correct API Summary for InsightSpike-AI

Based on the actual implementation, here are the correct constructors and methods:

## Quick Start Functions

```python
from insightspike import create_agent, quick_demo

# Create agent with minimal configuration
agent = create_agent(provider="openai")  # or "mock", "anthropic"

# Run interactive demo
quick_demo()
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

## Notes:

- Import paths changed: `core.layers.*` → `implementations.layers.*`
- Configuration now uses Pydantic models, not legacy dict format
- Memory manager includes automatic aging and size management
- CLI simplified to essential commands only
- Deprecated methods removed (save_graph, load_graph, _detect_spike, etc.)