# InsightSpike Architecture Documentation

> **鮮度注記(2026-07-03 監査)**: src 本体は 2026-04 以降凍結中のため、コア文書群は実装と同期している。
> 分類の詳細と全体案内は [docs/MAP.md](../MAP.md) を参照。
> ⚠ = 軽微な乖離あり、❓ = 実装の現存が未確認(要検証)。

## 📚 Documentation Index

### Core Architecture(実装と同期)
- **[Unified Core Architecture](unified_core_architecture.md)** ⭐ - `src/gedig/` 統一 F-eval コア(3 実験ライン共有、2026-03)
- **[Directory Structure](directory_structure.md)** - Clean architecture and package organization
- **[Layer Architecture](layer_architecture.md)** ⚠ - 4-layer processing system(バリアント: layer2_compatibility/_working_memory、layer3/ サブディレクトリは未記載)
- **[geDIG Selector & Core](gedig_selector_and_core.md)** - Canonical selector entry and refactored core
- **[geDIG Refactor Overview](gedig_refactor_overview.md)** - 10 モジュール分割(2026-02)
- **[Configuration System](configuration.md)** - YAML-based configuration and settings management
- **[Data Management](data_management_strategy.md)** - DataStore abstraction and data handling
- **[Query Storage System](query_storage.md)** - Query persistence and analysis architecture
- **[MainAgent Behavior](mainagent_behavior.md)** (deprecated) - legacy MainAgent

### Experiment-Side Architecture(実験ディレクトリ側に実体)
- **[Graph-Persistent DG / Sleep](../../experiments/maze/graph_persistent_dg/SPEC.md)** - Wake-Sleep-Wake の設計仕様
  (報酬値は実装が正: novel +0.2 / revisit −0.4。`--sleep-propagate replay` の意味論は
  `experiments/maze/test/test_sleep_propagate_semantics.py` が正典)
- **Three-Layer Search (L0/L1/L2)** ❌ 専用設計文書なし - 実装は `experiments/maze/qhlib/`(hash_index /
  graph_walker / attention / search_engine)、概要は [maze README](../../experiments/maze/README.md)
- **QHub Query-Node 体系** ❌ 専用設計文書なし - [maze README](../../experiments/maze/README.md) のノード体系節が現状の一次情報源

### Advanced Features
- **[Spectral GED Enhancement](spectral_ged_feature.md)** ❓ - Laplacian eigenvalue analysis(実装現存の確認要)
- **[Flash-geDIG Spec](flash_gedig_spec.md)** ❓ - GPU-native differentiable F(実装現存の確認要)
- **[Advanced Metrics](advanced_metrics_2025_01.md)** ⚠ - GeDIG, multi-hop reasoning
- **[Why InsightSpike is Advanced](why_insightspike_is_advanced.md)** - Key innovations and differentiators

### System Design
- **[Multi-User Design](multi_user_design.md)** ❓ - Architecture for multi-user scenarios(実装対応未確認)
- **[Vector Search Backend](vector_search.md)** - High-performance NumPy-based vector similarity search

> アーカイブ済み(2026-03-19): agent_types.md、recent_features_2024_07.md、
> navigator_na_bt_refactor_plan.md → `docs/archive/architecture/`

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

## 🔄 Recent Updates (January 2026)

### ⚡ High-Velocity Architecture (Flash-geDIG)
*   **[Flash-geDIG Library](flash_gedig_spec.md)** - GPU-native, differentiable implementation of structural metrics.
    *   **Zero-Copy**: No CPU transfer, pure Tensor operations.
    *   **Approximation**: Matrix powers $A^k$ for SP, Soft Thresholding for EPC.
    *   **End-to-End**: Fully differentiable, usable as a loss function.
*   **[Neuro-Pruning Tool](../design/neuro_pruning_spec.md)** - Structural lobotomy for Transformers.
    *   **Diagnosis**: Measures "Structural Fitness" (F-score) of every attention head.
    *   **Action**: Prunes bottom $N$% low-structure heads ("Chaos removal").
    *   **Result**: 10% pruning achieved with <1% accuracy drop on BERT-base.

## 🔄 Previous Updates (August 2025)

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

*Last updated: July 2026 (index + freshness audit; body sections below the index are of August-2025 vintage — treat performance/maintenance notes as historical)*
