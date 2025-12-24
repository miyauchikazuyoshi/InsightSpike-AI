# InsightSpike Architecture Documentation

## 📚 Documentation Index

### Core Architecture
- **[Directory Structure](directory_structure.md)** - Clean architecture and package organization
- **[Layer Architecture](layer_architecture.md)** - 4-layer neurobiologically-inspired processing system
- **[Agent Types](agent_types.md)** - Available agent implementations and their use cases
- **[MainAgent Behavior](mainagent_behavior.md)** (deprecated) - Detailed behavior and processing flow of legacy MainAgent
- **[geDIG Selector & Core](gedig_selector_and_core.md)** - Canonical selector entry and refactored core (2025‑09)
- **[Configuration System](configuration.md)** - YAML-based configuration and settings management
- **[Data Management](data_management_strategy.md)** - DataStore abstraction and data handling
- **[Query Storage System](query_storage.md)** ⚡ **NEW** - Query persistence and analysis architecture

### Advanced Features
- **[Message Passing & Edge Re-evaluation](recent_features_2024_07.md)** - Question-aware graph enhancement
- **[Spectral GED Enhancement](spectral_ged_feature.md)** - Laplacian eigenvalue analysis for structural quality
- **[Advanced Metrics](advanced_metrics_2025_01.md)** - GeDIG, multi-hop reasoning, and quantum metrics
- **[Why InsightSpike is Advanced](why_insightspike_is_advanced.md)** - Key innovations and differentiators

### System Design
- **[Multi-User Design](multi_user_design.md)** - Architecture for multi-user scenarios
- **[Vector Search Backend](vector_search.md)** - High-performance NumPy-based vector similarity search

## 🧠 Quick Overview

InsightSpike implements a brain-inspired architecture with 4 processing layers:

1. **Layer 1 (Error Monitor)** - Cerebellum analog for error detection
2. **Layer 2 (Memory Manager)** - Hippocampus analog for episodic memory
3. **Layer 3 (Graph Reasoner)** - Prefrontal cortex analog for reasoning
4. **Layer 4 (Language Interface)** - Broca's/Wernicke's areas analog for language

## 🚀 Getting Started

### Basic Usage
```bash
# Add knowledge
spike embed ./documents/

# Ask questions
spike query "What is the main concept?"

# Interactive mode
spike chat

# View insights
spike insights
```

### For Developers
```python
from insightspike.public import create_agent

# Public API (recommended)
agent = create_agent(provider="mock")  # or "openai" (requires API key)
res = agent.process_question("Your question here")
print(res.get("response", getattr(res, "response", "")))
```

### Local Knowledge App (Streamlit)
The Streamlit UI in `apps/knowledge_app.py` uses `insightspike.public.InsightAppWrapper` for chat, ingest, and graph visualization.

Notes:
- Top‑level imports should use `insightspike.public` (CI enforced)
- geDIG calculations must go through `algorithms.gedig.selector.compute_gedig` (STRICT guard available)

## 📊 Architecture Highlights

- **Neurobiologically-inspired** design based on brain structures
- **Graph-based reasoning** with PyTorch Geometric for insight detection
- **Flexible vector search** - FAISS optional, NumPy backend available
- **DataStore abstraction** - Filesystem, SQLite, or custom backends
- **Local knowledge app** - Streamlit UI built on `InsightAppWrapper`
- **Message passing** - Question-aware graph enhancement
- **Query persistence** - Full history tracking with analysis capabilities
- **Production-ready** with caching, error handling, and monitoring

## 🔄 Recent Updates (August 2025)

### C-Value System & Weight Vectors ⚡ **NEW**
- **C-value (Confidence) System** - Episodes track confidence through selection and repetition
- **Weight Vector Management** - Task-specific dimension importance adjustment
- **Separation of Concerns** - Clean separation between evaluation and confidence updates
- **Memory Management** - Confidence-based pruning and experience tracking
- See details in [C-Value Refactoring](../development/c_value_refactoring_plan.md) and [Vector Weights](../development/vector_weight_complete_plan.md)

### Query Storage System
- **Automatic query persistence** - All queries saved with rich metadata
- **Graph integration** - Queries as nodes with edges to episodes
- **Analysis capabilities** - Spike rate tracking, provider performance
- **Multiple backends** - FileSystem, SQLite, Memory stores
- See details in [Query Storage System](query_storage.md)

## 🔄 Previous Updates (July 2025)

### Message Passing & Edge Re-evaluation
- **Question-aware message passing** - Propagates query relevance through graph
- **Dynamic edge re-evaluation** - Discovers new connections based on context
- **Configurable via YAML** - Enable/disable features independently
- Performance optimizations needed for large graphs (>20 nodes)

### FAISS Removal & Vector Index Abstraction
- **Removed hard dependency on FAISS** - Resolved segmentation fault issues
- **NumPy backend implementation** - Pure Python alternative
- **VectorIndexFactory** - Automatic backend selection
- See migration guide in [faiss_removal_complete.md](../development/done/faiss_removal_complete.md)

### DataStore Abstraction
- **Unified data access layer** - Consistent API for all data operations
- **Multiple backends** - Filesystem (default), SQLite, in-memory
- **Namespace support** - Isolate data by experiment or component
- Configuration: `datastore.root_path` in config.yaml

### Configuration System Updates
- **YAML-based configuration** - Central config.yaml file
- **Pydantic models** - Type-safe configuration validation
- **Environment-specific settings** - Development, testing, production modes
- **Backward compatibility** - Supports legacy dict-based configs

## 📈 Performance Considerations

### Current Bottlenecks
1. **Message Passing** - O(N²) complexity, exponential slowdown with graph size
2. **Graph Building** - Incremental updates can be expensive
3. **Memory Usage** - Graph state accumulates over time

### Optimizations
- Use `enable_message_passing: false` for better performance
- Limit `message_passing.iterations` to 1-2
- Consider `use_faiss: false` to avoid segmentation faults
- Regular cleanup of temporary files and caches

## 🛠️ Development Status

### Stable Features
- ✅ Core 4-layer architecture
- ✅ Basic graph reasoning and spike detection
- ✅ DataStore abstraction
- ✅ Vector search with NumPy backend
- ✅ Configuration system
- ✅ C-value confidence system
- ✅ Weight vector management
- ✅ Confidence-based memory management

### Experimental Features
- ⚠️ Message passing (performance issues)
- ⚠️ GNN integration (use_gnn flag)
- ⚠️ Advanced GED/IG algorithms
- ⚠️ Multi-hop graph search

### Known Issues
- Message passing performance degrades with graph size
- Some advanced metrics not fully implemented
- Circular import warnings in embedder module

## 📖 Further Reading

- **[Configuration Guide](configuration.md)** - Detailed configuration options
- **[Layer Architecture](layer_architecture.md)** - Deep dive into each layer
- **[Development Docs](../development/)** - Implementation plans and technical details
- **[Research Notes](../research/)** - Theoretical foundations and future directions

## 🔧 Maintenance

### Regular Tasks
```bash
# Clean up caches and temporary files
./cleanup_disk_space.sh

# Run regression tests
poetry run pytest tests/regression/

# Check configuration
poetry run python -m insightspike.config validate
```

### Monitoring
- Check `.mypy_cache` size (can grow to 200MB+)
- Monitor `~/Library/Caches/claude-cli-nodejs` (can exceed 1GB)
- Review data/ directory for accumulated experiments

---

*Last updated: August 2025*
