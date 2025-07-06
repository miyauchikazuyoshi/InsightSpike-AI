# InsightSpike-AI Diagrams

This directory contains Mermaid diagrams illustrating the InsightSpike-AI architecture and processes.

## 🧠 Architecture Diagrams

### Core System Architecture
- **TECHNICAL_ARCHITECTURE.mermaid** - Overall system architecture with scalable graph implementation
- **TECHNICAL_ARCHITECTURE_v2.mermaid** - ⚡ **NEW** Updated architecture with L2EnhancedScalableMemory integration
- **DATA_FLOW_ARCHITECTURE.mermaid** - ⚡ **NEW** Data flow and directory structure (data/, experiments/)
- **WORKFLOW_TREE.mermaid** - Processing workflow and decision trees
- **SYSTEM_DASHBOARD.mermaid** - Real-time system metrics with hierarchical graph performance

### Memory Management
- **INTRINSIC_MOTIVATION_FLOW.mermaid** - Intrinsic reward → episode management flow
- **EPISODE_INTEGRATION_MATRIX.mermaid** - Graph-centric integration matrix (C-value free)
- **EPISODE_MANAGEMENT_WORKFLOW.mermaid** - Graph-informed episode management with automatic splitting

### Insight Processing
- **INSIGHT_LIFECYCLE.mermaid** - Insight discovery and quality assessment lifecycle

## 🚀 Latest Updates (2025-07-06)

### New Diagrams Added
1. **TECHNICAL_ARCHITECTURE_v2.mermaid**
   - Shows MainAgent using L2EnhancedScalableMemory
   - Highlights ScalableGraphManager with O(n log n) performance
   - Updated data storage paths (data/core/, data/db/, etc.)

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
- ✅ **Scalable FAISS-indexed Memory (O(n log n))**
- ✅ **Hierarchical Graph Structure (O(log n) search)**
- ✅ **Graph-Centric Episode Management (No C-values)**
- ✅ **Dynamic Importance from Graph Structure**
- ✅ **Graph-Informed Integration/Splitting**
- ✅ **100K+ Episode Handling (<5ms search)**
- ✅ **geDIG Algorithm with Scalable Implementation**

## 🔄 Performance at Scale

The diagrams now include performance metrics:

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