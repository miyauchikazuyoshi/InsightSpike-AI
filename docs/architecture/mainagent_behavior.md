# MainAgent Behavior Documentation

> **式の位置づけ（簡約式） / Formula Status (Simplified)**: この文書の数式は説明用の簡約式です。正準定義（Canonical）は `docs/gedig_spec.md` です。
>
> **Status**: Current public runtime behavior as of 2026-07-24.


## 🧠 Overview

MainAgent is the core orchestrator in InsightSpike that coordinates all 4 neurobiologically-inspired layers to process questions, manage memory, detect insights, and generate responses.

## Facade and service boundaries

`MainAgent` remains the public facade and owns the replaceable component
attributes `l1_error_monitor`, `l1_embedder`, `l2_memory`, `l3_graph`,
`l4_llm`, and `datastore`. The L1–L4 algorithms still run through the facade;
the following coordination-only responsibilities are delegated:

| Service | Responsibility |
|---|---|
| `AgentLifecycle` | Resolve and initialize the current L4 provider, load legacy memory when applicable, and reset L1 |
| `AgentPersistence` | Exact episode snapshots, graph state, legacy persistence, and index rebuilds |
| `CycleResultAggregator` | Select the highest-quality cycle and materialize result metadata after learning hooks |
| `AgentConfigAccess` | Read live config values and build the normalized scalar facade |

These services do not own copies of the public components. A test or
integration may replace a component attribute on `MainAgent`, and the next
operation receives that current instance.

### Lifecycle contract

- A newly resolved provider is assigned to `MainAgent.l4_llm` before later
  initialization steps run. If memory loading or monitor reset fails, a retry
  reuses the same provider rather than constructing another one.
- Repeated `initialize()` calls preserve the established observable behavior:
  an already initialized provider is not initialized twice, while applicable
  memory loading and monitor reset run again.
- A failed repeated initialization returns `False`; if the agent was already
  initialized, its prior `initialized` state remains unchanged.
- Provider-construction errors use one lightweight fallback implementation
  with the marker response `[fallback-llm-response]`.

### Live configuration and explicit overrides

`AgentConfigAccess` rebuilds `NormalizedConfig` from the current source config
when `_nc()` is read. Runtime changes such as adaptive graph thresholds are
therefore visible to later cycles and learning snapshots.

`_normalized_config` remains an internal compatibility patch point used by
existing integrations. When it is replaced (for example with
`dataclasses.replace(..., gedig_mode="ab")`), only fields that differ from the
last generated facade become explicit overrides. Other fields continue to
follow the live source config.

### Result aggregation contract

The result with the highest `reasoning_quality` is selected; equal scores keep
the first result. Learning hooks run against that selected result before a new
public `CycleResult` is materialized. Aggregated graph metadata includes cycle
count, convergence, optional history, and agent statistics. `query_id` starts
as `None` on the aggregate and is assigned by `QueryRecorder`.

## 🔄 Processing Cycle

### 1. **Question Processing Flow**
```python
def process_question(self, question: str, max_cycles: int = 5, verbose: bool = False) -> CycleResult
```

The agent processes questions through multiple reasoning cycles.

**NEW: Query Storage** ⚡:
- All queries are automatically saved with metadata
- Includes processing time, LLM provider, spike status
- Query ID returned in CycleResult for tracking

```
Question → L1 (Error Analysis) → L2 (Memory Search) → L3 (Graph Analysis) → L4 (Response Generation)
     ↑              ↓ (bypass)                                                        ↓
     ←──────────── ↓ ────────── Convergence Check (similarity > 0.9) ←──────────────←
                   ↓
                   └──────────────────────────────────────────→ L4 (Direct Response)
```

**NEW: Layer1 Bypass** (July 2024):
- Low-uncertainty queries can skip directly from L1 to L4
- Bypasses memory search and graph analysis for known facts
- 10x performance improvement for production systems

**Cycle Components:**
1. **Error State Analysis** (L1)
   - Detects unknown concepts
   - Calculates uncertainty (0.0-1.0)
   - Identifies knowledge gaps

2. **Memory Retrieval** (L2)
   - Searches episodic memory
   - Returns top-k relevant documents
   - Updates C-values based on usage
   - **NEW**: Includes relevant insights from registry
   - **NEW**: Optional graph-based multi-hop search

3. **Graph Reasoning** (L3)
   - Builds knowledge graph from documents
   - Calculates ΔGED and ΔIG
   - Detects conflicts and spikes
   - Provides reasoning quality score
   - **Default (2025-10)**: GeDIGCore（advanced）でクエリ中心の局所サブグラフ（上位kノードの半径r-hop）を評価
     - 既定値: `metrics.query_centric=true`, `metrics.query_topk_centers=3`, `metrics.query_radius=1`
     - エンジン: `graph.ged_algorithm="advanced"`, `graph.ig_algorithm="advanced"`
     - LITEモード時はL3をスキップ（軽量経路）
   - **NEW**: Auto-registers insights when spikes detected

4. **Response Synthesis** (L4)
   - Generates natural language response
   - Incorporates graph analysis insights
   - Produces confidence scores
   - **NEW**: Mode-aware prompt building based on model capacity

### 2. **Convergence Detection**
The agent stops cycling when:
- Text similarity between consecutive responses > 0.9
- Maximum cycles reached
- Critical error occurs

## 📊 Quality Calculation

```python
def _calculate_reasoning_quality() -> float
```

Quality is a weighted combination:
- **Error Score** (20%): 1.0 - uncertainty
- **Memory Score** (30%): min(1.0, retrieved_docs / 3)
- **Graph Score** (30%): From L3 reasoning quality
- **LLM Score** (20%): Response confidence

Final quality: 0.0 (poor) to 1.0 (excellent)

## 🚀 Spike Detection

InsightSpike moments are detected when:
- **ΔGED** ≤ -0.5 (significant structural change)
- **ΔIG** ≥ 0.2 (information gain)

These thresholds are configurable in `config.yaml`.

## 💾 Memory Management

### State persistence

```python
saved: bool = agent.save_state()
loaded: bool = agent.load_state()
```

With a configured `DataStore`, persistence replaces the complete episode
snapshot, saves graph id `main_graph` in namespace `agent_state`, and rebuilds
the memory index after load. Loading an empty snapshot clears existing
episodes. Without a `DataStore`, the legacy L2/L3 save and load paths remain
available. Both public methods return `bool`; callers should handle `False`.

### Episode Storage
```python
def add_document(
    self,
    text: str | Mapping[str, Any],
    c_value: float = 0.5,
    metadata: dict | None = None,
) -> bool

def add_knowledge(self, text: str, c_value: float = 0.5) -> Dict[str, Any]
def learn(self, text: str, c_value: float = 0.5) -> Dict[str, Any]
```

`add_document()` is the compact boolean API. It accepts either text or the
legacy `{"text", "metadata", "c_value"}` mapping and rejects empty text or a
non-numeric confidence without inserting an episode.

`add_knowledge()` returns the episode id and graph-update status. `learn()` is
its compatibility alias and additionally reports `episodes_added` plus an
`insights` list. `graph_updated` means that L3 explicitly confirmed a completed
update; the current compatibility `L3GraphReasoner.update_graph()` is a no-op,
so merely accepting that call does not produce a false positive.

### Reward System
Episodes involved in successful reasoning receive C-value boosts:
- Base boost: 5% of total reward
- Maximum boost: 0.1 per cycle
- Propagates to related episodes

## 📈 Statistics & Insights

### Statistics Tracking
```python
def get_stats() -> Dict[str, Any]
```
- Total reasoning cycles
- Memory statistics
- Average quality scores
- Reasoning history length

### Insight Management
```python
def get_insights(limit: int = 5) -> Dict[str, Any]
def search_insights(concept: str, limit: int = 10) -> List[Dict]
```
- Accesses InsightFactRegistry
- Reuses the registry owned by the agent; it does not open an unrelated
  registry per call
- Maps `InsightFact.text`, `generated_at`, `quality_score`, and
  `relationship_type` to the public answer/timestamp/importance/category view
- Supports concept-based search through
  `InsightFactRegistry.search_insights_by_concept()`

## 🧪 Experiments & Demos

### Built-in Experiments
```python
def run_experiment(experiment_type: str, episodes: int = 5) -> Dict
```

**Types:**
- `simple`: Basic Q&A functionality
- `insight`: Spike detection capability
- `math`: Mathematical reasoning

### Demo Mode
```python
def run_demo() -> List[Dict[str, Any]]
```
Showcases:
1. Knowledge storage
2. Retrieval accuracy
3. Insight detection
4. Complex reasoning

## ⚙️ Configuration

MainAgent now uses the new Pydantic-based configuration system:

```python
from insightspike.config import InsightSpikeConfig, load_config
from insightspike.config.presets import ConfigPresets

# Option 1: Load from config.yaml
config = load_config()

# Option 2: Use a preset model
config = ConfigPresets.development()

# Option 3: Create custom config
custom_config = InsightSpikeConfig(
    llm={"provider": "mock"},
    memory={"episodic_memory_capacity": 100},
)
```

Key configuration parameters:
- `llm.provider`: LLM provider (`mock`, `openai`, `anthropic`, etc.)
- `memory.episodic_memory_capacity`: Number of episodes to retain
- `graph.episode_merge_threshold`: When to merge similar episodes
- `graph.spike_ged_threshold`: Threshold for spike detection
- `graph.ged_algorithm` / `graph.ig_algorithm`: メトリクス実装の選択（既定: `advanced`）
- `metrics.query_centric`, `metrics.query_topk_centers`, `metrics.query_radius`: クエリ中心の局所評価を制御（既定: `true`, `3`, `1`）

## 🔍 Error Handling

MainAgent handles errors gracefully:
- Returns `CycleResult` with `success=False`
- Logs detailed error information
- Provides fallback responses
- Maintains agent state consistency

## 💡 Best Practices

1. **Initialize Once**: Create agent once and reuse
2. **Monitor Quality**: Check `reasoning_quality` scores
3. **Manage Memory**: Periodically save state with `save_state()`
4. **Configure Appropriately**: Use presets for different use cases
5. **Handle Spikes**: Pay attention to `spike_detected` flag

## 🔗 Integration Example

```python
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.factory import DataStoreFactory
from insightspike.config.presets import ConfigPresets

# Create dependencies
datastore = DataStoreFactory.create("in_memory")
config = ConfigPresets.development()

# Initialize
agent = MainAgent(config=config, datastore=datastore)
if not agent.initialize():
    raise Exception("Initialization failed")

# Add knowledge
result = agent.add_knowledge(
    "Neural networks are inspired by biological neurons."
)

# Process question
answer = agent.process_question(
    "How do neural networks relate to the brain?",
    max_cycles=3,
    verbose=True
)

# Check for insights
if answer.spike_detected:
    print("Insight discovered!")
    insights = agent.get_insights()
    
# Access query ID (NEW)
print(f"Query saved with ID: {answer.query_id}")
    
# Save state
if not agent.save_state():
    raise RuntimeError("State persistence failed")
```

## 🐛 Debugging

Enable verbose mode to see:
- Cycle-by-cycle processing
- Retrieved documents
- Graph analysis metrics
- Convergence progress
- Quality scores

Use `get_stats()` to monitor:
- Agent health
- Performance metrics
- Memory usage
