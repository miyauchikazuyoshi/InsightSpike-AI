# InsightSpike Agent Types

## 🤖 Available Agents

### 1. **MainAgent** (`implementations/agents/main_agent.py`)
- **Purpose**: Core orchestrating agent that coordinates all 4 layers
- **Use Case**: Standard question-answering tasks with neurobiologically-inspired processing
- **Architecture**:
  - **L1 (Error Monitor)**: Uncertainty detection and error handling
  - **L2 (Memory Manager)**: Episode storage/retrieval with C-value rewards
  - **L3 (Graph Reasoner)**: Graph analysis and spike detection (optional)
  - **L4 (LLM Interface)**: Natural language generation
- **Core Methods**:
  - `process_question()`: Main reasoning cycle with convergence detection
  - `add_episode_with_graph_update()`: Store knowledge with graph analysis
  - `get_stats()`: Agent statistics and performance metrics
  - `get_insights()`: Retrieve discovered insights
  - `search_insights()`: Search for concept-related insights
  - `run_experiment()`: Execute predefined experiments
  - `run_demo()`: Interactive capability demonstration
- **Features**:
  - Multi-cycle reasoning with convergence detection
  - Automatic episode management (merge/split/prune)
  - Spike detection (ΔGED/ΔIG thresholds)
  - Memory reward updates based on reasoning quality
  - State persistence (save/load)

### 2. **ConfigurableAgent** (`implementations/agents/configurable_agent.py`)
- **Purpose**: Advanced Q&A agent with configurable features
- **Use Case**: When you need customizable behavior or advanced features
- **Operation Modes**:
  - **BASIC**: Like MainAgent
  - **ENHANCED**: Graph-aware memory
  - **QUERY_TRANSFORM**: Query evolution through graph
  - **ADVANCED**: Multi-hop reasoning
  - **OPTIMIZED**: Production features (caching, async)
  - **GRAPH_CENTRIC**: Pure graph-based (no C-values)
- **Features**:
  - All MainAgent features
  - Feature toggles
  - Query caching
  - Async processing
  - GPU acceleration
  - Query transformation

### 3. **GenericAgent** (`implementations/agents/generic_agent.py`)
- **Purpose**: Domain-agnostic agent for non-Q&A tasks
- **Use Case**: Games, simulations, custom environments
- **Features**:
  - Abstract interfaces
  - Q-learning based
  - Environment adapter pattern
  - No hardcoded layer dependencies

### 4. **DataStoreMainAgent** (`implementations/agents/datastore_agent.py`)
- **Purpose**: Scalable agent using DataStore backend for persistence
- **Use Case**: Production deployments with large knowledge bases
- **Architecture**: Similar to MainAgent but with DataStore integration
- **Core Methods**:
  - `process()`: Main processing pipeline with 5 phases:
    1. Store episode in DataStore
    2. Search for related episodes
    3. Detect insights/spikes
    4. Generate reasoning (if spike detected)
    5. Generate response (always provided)
  - `search()`: Search for relevant episodes
  - `get_stats()`: Get agent and DataStore statistics
  - `save_checkpoint()` / `load_checkpoint()`: State persistence
- **Key Differences from MainAgent**:
  - Uses DataStore for all persistence (not in-memory)
  - Working memory approach for active data (L2WorkingMemoryManager)
  - Lazy loading of data on demand
  - Better scalability for large datasets
  - Async-ready architecture with sync fallbacks
  - Separate embedding management
- **Features**:
  - Transaction-based persistence
  - Multi-user support with data isolation
  - Reduced memory footprint
  - Compatible with various DataStore backends (SQLite, PostgreSQL, etc.)
  - Vector similarity search via FAISS integration
  - Automatic response generation with LLM fallback
- **Configuration**:
  ```python
  from insightspike.config.models import InsightSpikeConfig
  
  config = InsightSpikeConfig(
      memory={'max_retrieved_docs': 15},  # Not 'search_k'
      llm={'provider': 'openai', 'model_name': 'gpt-3.5-turbo'}
  )

## 📊 Quick Comparison

| Feature | MainAgent | ConfigurableAgent | GenericAgent | DataStoreMainAgent |
|---------|-----------|-------------------|--------------|-------------------|
| Q&A Tasks | ✅ | ✅ | ❌ | ✅ |
| Fixed Architecture | ✅ | ❌ | ❌ | ✅ |
| Configurable | ❌ | ✅ | ✅ | ⚠️ |
| Multiple Modes | ❌ | ✅ | ❌ | ❌ |
| Caching | ❌ | ✅ | ❌ | ✅ |
| Async Processing | ❌ | ✅ | ❌ | ⚠️ |
| Custom Environments | ❌ | ❌ | ✅ | ❌ |
| Scalable Storage | ❌ | ❌ | ❌ | ✅ |
| Multi-User | ❌ | ❌ | ❌ | ✅ |
| Production Ready | ✅ | ✅ | ⚠️ | ✅ |

## 🚀 Which Agent to Use?

```python
# For simple Q&A tasks
from insightspike.implementations.agents import MainAgent
agent = MainAgent()

# For advanced Q&A with specific features
from insightspike.implementations.agents import ConfigurableAgent, AgentConfig, AgentMode
config = AgentConfig.from_mode(AgentMode.ENHANCED)
agent = ConfigurableAgent(config)

# For custom environments (games, etc.)
from insightspike.implementations.agents import GenericInsightSpikeAgent
agent = GenericInsightSpikeAgent(...)

# For scalable production deployments with DataStore
from insightspike.implementations.agents.datastore_agent import DataStoreMainAgent
from insightspike.implementations.datastore.sqlite_store import SQLiteDataStore

# Create DataStore backend
datastore = SQLiteDataStore("knowledge.db")  # or use PostgreSQL, etc.

# Initialize agent with configuration
from insightspike.config.models import InsightSpikeConfig
config = InsightSpikeConfig(
    memory={'max_retrieved_docs': 20},
    llm={'provider': 'openai'}  # or 'mock' for testing
)
agent = DataStoreMainAgent(datastore=datastore, config=config)
```

## 🔄 Migration Path

### Deprecated Agents
If you were using deprecated agents:
- `main_agent_enhanced.py` → Use `ConfigurableAgent` with `ENHANCED` mode
- `main_agent_optimized.py` → Use `ConfigurableAgent` with `OPTIMIZED` mode
- `main_agent_with_query_transform.py` → Use `ConfigurableAgent` with `QUERY_TRANSFORM` mode

### Migrating from MainAgent to DataStoreMainAgent

When to migrate:
- Your knowledge base exceeds 10,000 episodes
- You need multi-user support
- Memory usage is a concern
- You need transaction-based persistence

Migration steps:
```python
# Before (MainAgent)
from insightspike.implementations.agents import MainAgent
agent = MainAgent()
result = agent.process_question("What is consciousness?")

# After (DataStoreMainAgent)
from insightspike.implementations.agents.datastore_agent import DataStoreMainAgent
from insightspike.implementations.datastore.sqlite_store import SQLiteDataStore

datastore = SQLiteDataStore("knowledge.db")
agent = DataStoreMainAgent(datastore=datastore)
result = agent.process("What is consciousness?")  # Note: method name is 'process', not 'process_question'

# Access results
print(result['response'])  # DataStoreMainAgent returns dict, not object
print(result.get('has_spike', False))
```

Key differences:
- DataStoreMainAgent returns dictionaries, not objects
- Method names may differ (`process` vs `process_question`)
- Requires explicit DataStore initialization
- Better performance for large datasets