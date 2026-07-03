# InsightSpike-AI Diagrams

This directory contains Mermaid diagrams illustrating the InsightSpike-AI architecture and processes.

> **鮮度注記(2026-07-03 監査)**: 図は src 本体(2026-04 以降凍結)を描いたもの。
> 古くなった 4 図(MULTIHOP_CONCEPT / WORKFLOW_TREE / SYSTEM_DASHBOARD / WRITER_INJECTION)は
> `docs/archive/diagrams/` へ移動済み。

## 🧠 Architecture Diagrams

### Experiment-Side Mechanisms(2026-07 新規)
- **THREELAYER_SEARCH.mermaid / .svg** - ⚡ **NEW** L0 ハッシュ / L1 注意歩行 / L2 フォールバックのカスケード
  (設計文書: [threelayer_search_architecture.md](../architecture/threelayer_search_architecture.md))
- **WAKE_SLEEP_WAKE.mermaid / .svg** - ⚡ **NEW** Wake-Sleep-Wake と `--sleep-propagate` 3 variant
  (replay/on/off)、readout 経路、v1/v2 のエビデンス注記付き
  (設計: [graph_persistent_dg/SPEC.md](../../experiments/maze/graph_persistent_dg/SPEC.md)、勝敗: [prereg 台帳](../prereg/README.md))

### Core System Architecture
- **CURRENT_ARCHITECTURE.mermaid** - ⚡ **LATEST** Current simplified architecture with query storage and Streamlit app wrapper (2025-09)
- **CURRENT_ARCHITECTURE.svg** - Rendered export of the latest architecture diagram
- **QUERY_STORAGE_ARCHITECTURE.mermaid** - Query storage and analysis system
- **CONFIG_SYSTEM.mermaid** - Pydantic-based configuration system
- **CLI_ARCHITECTURE.mermaid** - CLI dependency injection with Typer Context
- **TECHNICAL_ARCHITECTURE.mermaid** - System architecture with scalable graph implementation (L2_Integration node fixed 2026-07)
- **DATA_FLOW_ARCHITECTURE.mermaid** - Data flow with query storage
- **GEDIG_SELECTOR_PIPELINE.mermaid** - Canonical selector→core pipeline (2026-02 updated)
- **GEDIG_REFACTOR_2025_08.mermaid** - Modular geDIG architecture post-refactoring (2026-02)

### Memory Management
- **INTRINSIC_MOTIVATION_FLOW.mermaid** - Intrinsic reward → episode management flow
- **EPISODE_INTEGRATION_MATRIX.mermaid** - Graph-centric integration matrix (C-value free)
- **EPISODE_MANAGEMENT_WORKFLOW.mermaid** - Graph-informed episode management with automatic splitting

### Insight Processing
- **INSIGHT_LIFECYCLE.mermaid** - Insight discovery and quality assessment lifecycle

## 🚀 Latest Updates (2026-02-01)

### geDIG Modular Refactoring Complete
1. **GEDIG_SELECTOR_PIPELINE.mermaid** ⚡ **UPDATED**
   - Updated to reflect modular package structure
   - Shows GeDIGConfig configuration layer
   - Illustrates module dependencies (linkset, multihop, spike, graph_utils)

2. **GEDIG_REFACTOR_2025_08.mermaid** ⚡ **UPDATED**
   - Renamed to reflect completed refactoring (2026-02)
   - Shows 10 extracted modules with line counts
   - Includes statistics: 2,159→779 lines (-64%), 54%→84% coverage

3. **Refactoring Statistics**
   - gedig_core.py: 2,159 → 779 lines (-64%)
   - Modules: 1 → 10 (+9)
   - Tests: 53 → 227 (+174)
   - Coverage: 54% → 84% (+30%)

4. **Note**: SVG files may need regeneration from updated mermaid sources

---

## 🚀 Previous Updates (2025-08-06)

### Integrated Vector-Graph Index Implementation
1. **Integrated Index Architecture** ⚡ **NEW**
   - Pre-normalized vectors eliminate O(n) search bottleneck
   - Dual storage: normalized vectors + norms
   - Spatial indexing for O(log n) position-based queries
   - 100% backward compatible with existing DataStore APIs

2. **Updated Diagrams**
   - DATA_FLOW_ARCHITECTURE.mermaid - Added integrated index structure
   - TECHNICAL_ARCHITECTURE.mermaid - Updated Layer 2 with integrated index
   - Shows vector normalization flow and spatial indexing

3. **Performance Improvements**
   - Vector search: O(n) → O(1) with pre-normalization
   - Spatial search: Added O(log n) position indexing
   - Memory: Efficient dual storage (vectors + norms)

### SP / geDIG 温度ノブの反映メモ（2025-12）
- RAG v3-lite で SP 評価スコープ・サンプリング、ΔH softmax 温度（`entropy_tau`）を YAML で切り替え可能に。  
- 既存の図は変更不要だが、SP Flow (`sp_flow.svg`) を参照するとスコープ/サンプリングの位置づけが把握しやすい。

## 🚀 Previous Updates (2025-07-28)

### Query Storage Feature Implementation
1. **QUERY_STORAGE_ARCHITECTURE.mermaid** ⚡ **NEW**
   - Complete query persistence system
   - Automatic saving in MainAgent and AdaptiveProcessor
   - Rich metadata tracking (processing time, cycles, quality)
   - Graph integration with query nodes and edges

2. **Updated Diagrams**
   - CURRENT_ARCHITECTURE.mermaid - Added query storage layer
   - DATA_FLOW_ARCHITECTURE.mermaid - Added query storage flow
   - WORKFLOW_TREE.mermaid - Added query persistence steps

3. **Removed Outdated Files**
   - THEORETICAL_DESIGN_FLAWS_ANALYSIS.md (old issues, now fixed)
   - PIPELINE_ISSUES_SUMMARY.md (old analysis)
   - Interface analysis CSV files (outdated)

## 🚀 Previous Updates (2025-01-18)

### Major Refactoring Complete
1. **CURRENT_ARCHITECTURE.mermaid** ⚡ **NEW**
   - Simplified architecture without Query Transformation
   - Clean separation of concerns with Composition Root pattern
   - Type-safe API with CycleResult dataclass
   - Dependency injection via Typer Context

2. **CONFIG_SYSTEM.mermaid** ⚡ **NEW**
   - New Pydantic-based configuration system
   - Environment variable overrides
   - Multiple presets (development, experiment, production)
   - Legacy config converter for backward compatibility

3. **CLI_ARCHITECTURE.mermaid** ⚡ **NEW**
   - Typer Context-based dependency injection
   - Composition Root pattern implementation
   - Agent caching per configuration
   - No global state design

4. **Updated Diagrams**
   - INTRINSIC_MOTIVATION_FLOW.mermaid - Removed C-value references, added dynamic importance
   - WORKFLOW_TREE.mermaid - Updated config system to Pydantic-based

5. **Removed Outdated Diagrams**
   - QUERY_TRANSFORMATION_ARCHITECTURE.mermaid (feature removed)
   - TECHNICAL_ARCHITECTURE_v2.mermaid (outdated)
   - DATA_FLOW_ARCHITECTURE_v2.mermaid (outdated)

## 🚀 Previous Updates (2025-07-06)

### Historical Updates
1. **TECHNICAL_ARCHITECTURE.mermaid**
   - Shows scalable graph implementation
   - ScalableGraphManager with O(n log n) performance
   - Updated data storage paths

2. **DATA_FLOW_ARCHITECTURE.mermaid**
   - Complete data directory structure
   - Experiment data management workflow
   - Backup and restore procedures
   - Data access patterns

### Phase 2 & 3 Implementation Updates

### Scalable Graph Architecture
The diagrams now reflect the **NP-hard GED optimization** solutions:

1. **Phase 2: Scalable Graph Construction**
   - FAISS-based approximate nearest neighbor search
   - O(n²) → O(n log n) complexity reduction
   - Configurable top-k neighbor selection

2. **Phase 3: Hierarchical Graph Management**
   - 3-layer hierarchy: Episodes → Clusters → Super-clusters
   - O(log n) search complexity
   - 100x+ compression for large datasets
   - Dynamic document addition without full rebuild

### Graph-Centric Memory Management
The diagrams now show the **C-value free** implementation:

- **Dynamic Importance Calculation**:
  - Graph degree (40%)
  - Access frequency (30%)
  - Time decay (30%)

- **Graph-Informed Integration**:
  - Base threshold: 0.85
  - Graph bonus: -0.1 if connected
  - Weight = graph_strength OR similarity

- **Automatic Splitting**:
  - Detects neighbor conflicts
  - Splits episodes to maintain coherence
  - Self-attention-like behavior

## 📊 Key Features Illustrated

- ✅ **4-Layer Neurobiological Architecture**
- ✅ **Clean Separation of Concerns (Composition Root)**
- ✅ **Type-safe API with CycleResult**
- ✅ **Pydantic-based Configuration System**
- ✅ **Dependency Injection via Typer Context**
- ✅ **Integrated Vector-Graph Index (O(1) search)** ⚡ **NEW**
- ✅ **Pre-normalized Vector Storage** ⚡ **NEW**
- ✅ **Spatial Indexing for Navigation** ⚡ **NEW**
- ✅ **Graph-Centric Episode Management**
- ✅ **Dynamic Importance from Graph Structure**
- ✅ **100K+ Episode Handling (<1ms search)** ⚡ **IMPROVED**
- ✅ **geDIG Algorithm with Scalable Implementation**
- ✅ **Query Storage and Analysis System**
- ✅ **Query-Episode Graph Relationships**
- ✅ **Spike Success Rate Tracking**
- ✅ **Provider Performance Analytics**

## 🔄 Performance at Scale

The diagrams now include performance metrics:

### Integrated Index Performance ⚡ **NEW**
| Dataset Size | Vector Search | Spatial Search | Memory Overhead |
|-------------|---------------|----------------|-----------------|
| 1,000       | 0.1ms (O(1))  | 0.2ms         | +4KB            |
| 10,000      | 0.1ms (O(1))  | 0.5ms         | +40KB           |
| 100,000     | 0.1ms (O(1))  | 1ms           | +400KB          |

### Legacy FAISS Performance (for comparison)
| Dataset Size | Build Time | Search Time | Compression |
|-------------|------------|-------------|-------------|
| 1,000       | 150ms      | 0.5ms       | 100x        |
| 10,000      | 1.5s       | 2ms         | 200x        |
| 100,000     | 15s        | 5ms         | 500x        |

## 🔧 Usage

These diagrams can be:
1. **Viewed on GitHub** - Automatic Mermaid rendering in markdown
2. **Rendered locally** - Using VS Code Mermaid extensions
3. **Exported** - To PNG/SVG for presentations
4. **Referenced** - In documentation and papers

---

**Note**: All diagrams have been updated to reflect the current implementation with scalable graph management and C-value free episode handling.

Linkset‑First: The geDIG IG component is now paper‑aligned (Linkset‑IG). When using Core directly in code snippets, prefer passing a `linkset_info` payload to avoid the deprecated graph‑IG fallback (which now emits a one‑time warning).
