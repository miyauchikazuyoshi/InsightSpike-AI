# Development Documentation

This directory contains all development-related documentation for InsightSpike AI.

## Structure

### 📂 Active Development Plans
- **CI_TEST_UPDATE_PLAN.md** - Plan for updating CI configuration and tests
- **MULTIMODAL_EXPERIMENTS_ROADMAP.md** - Multi-year experimental roadmap for multimodal capabilities
- **gedig_layer1_roadmap.md** - Roadmap for evolving Layer1 to use geDIG-based structural search
- **layer3_improvements.md** - Improvement plan for Layer3 entropy calculations and structural measures
- **layer4_refactoring_plan.md** - Plan to refactor Layer4 architecture to properly reflect PromptBuilder's role

### 📂 done/ - Completed Work
Contains documentation for completed implementations, investigations, and features:

#### Implementations
- **C_VALUE_REMOVAL_IMPLEMENTATION.md** - Completed C-value removal and graph-centric memory transition
- **ENHANCED_EPISODE_MANAGEMENT_IMPL.md** - Completed enhanced episode management with graph integration
- **PHASE3_HIERARCHICAL_IMPLEMENTATION.md** - Completed Phase 3 hierarchical graph management (100K+ episodes)
- **query_transformation.md** - Completed Query Transformation feature documentation
- **GED_IG_ALGORITHMS.md** - Completed GED/IG algorithm configuration system

#### Investigations & Fixes
- **GRAPH_UPDATE_INVESTIGATION.md** - Completed investigation of graph_pyg.pt update issues
- **GRAPH_EDGE_GENERATION_FIX.md** - Completed fix for graph edge generation problems

#### Design Decisions
- **command_naming_rationale.md** - Rationale for CLI command naming (query/embed)

#### Reports
- **COMPRESSION_EFFICIENCY_REPORT.md** - Completed analysis of compression efficiency
- **PHASE3_RAG_PERFORMANCE_RESULTS.md** - Completed Phase 3 RAG performance benchmark results

## Development Status Overview

### 🚀 Currently Active
1. **geDIG Layer1** - Revolutionary approach to replace cosine similarity with graph-based retrieval
2. **Layer3 Improvements** - Fixing entropy calculations and improving structural measures
3. **Layer4 Refactoring** - Architectural improvements for clarity
4. **CI/Test Updates** - Improving development infrastructure
5. **Multimodal Experiments** - Long-term vision for multimodal AGI capabilities

### ✅ Recently Completed
1. **Query Transformation** - Human-like thinking through graph message passing
2. **Phase 3 Scalability** - Support for 100,000+ episodes
3. **Graph Management** - Robust edge generation and update mechanisms
4. **C-value Removal** - Simplified architecture with graph-centric approach

## Contributing

When adding new development documentation:
1. Place active plans and roadmaps in this directory
2. Move completed work to the `done/` folder
3. Update this README to reflect the current status