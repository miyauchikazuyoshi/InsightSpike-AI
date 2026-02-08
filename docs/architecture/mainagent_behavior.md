# MainAgent Behavior Documentation

> **式の位置づけ（簡約式） / Formula Status (Simplified)**: この文書の数式は説明用の簡約式です。正準定義（Canonical）は `docs/gedig_spec.md` です。


## 🧠 Overview

MainAgent is the core orchestrator in InsightSpike that coordinates all 4 neurobiologically-inspired layers to process questions, manage memory, detect insights, and generate responses.

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

### Episode Storage
```python
def add_episode_with_graph_update(self, text: str, source: str = "user") -> Dict
```

1. Creates episode with embedding
2. Performs graph analysis
3. Detects potential spikes
4. Updates memory rewards
5. Triggers automatic management:
   - **Merge**: Similar episodes (> 0.8 similarity)
   - **Split**: Conflicting episodes (> 0.3 conflict)
   - **Prune**: Low-value episodes (< 0.1 C-value)

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
- Returns categorized insights
- Supports concept-based search

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

# Option 2: Use a preset
preset_dict = ConfigPresets.get_preset("development")

# Option 3: Create custom config
from insightspike.config.models import CoreConfig, MemoryConfig
custom_config = InsightSpikeConfig(
    core=CoreConfig(llm_provider="clean"),
    memory=MemoryConfig(episodic_memory_capacity=100)
)
```

Key configuration parameters:
- `core.llm_provider`: LLM provider ("mock", "clean", "openai", etc.)
- `memory.episodic_memory_capacity`: Number of episodes to retain
- `reasoning.episode_merge_threshold`: When to merge similar episodes
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
from insightspike.core.base.datastore import DataStore
from insightspike.config.converter import ConfigConverter
from insightspike.config.presets import ConfigPresets

# Create dependencies
datastore = DataStore()  # Or use a specific implementation
preset_dict = ConfigPresets.get_preset("development")
config = ConfigConverter.preset_dict_to_legacy_config(preset_dict)

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
agent.save_state()
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
---
status: deprecated
note: This document describes the legacy MainAgent behavior. The recommended path is to use `ConfigurableAgent` (implementations/agents/configurable_agent.py) via the public API `insightspike.public.create_agent`. geDIG computations must go through the selector entrypoint.
updated: 2025-09-08
---
