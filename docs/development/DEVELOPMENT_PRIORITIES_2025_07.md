# Development Priorities - Jury 2025

## Overview
This document outlines the development priorities for InsightSpike-AI following the major refactoring completed in January 2025.

## Status Summary

### ✅ Completed (Moved to `/done`)
- Memory manager deprecation
- DataStore migration
- Graph search implementation
- Hybrid episode splitting
- Adaptive loop implementation
- Code review fixes from July 2024
- geDIG implementation status
- Layer1 bypass mechanism

### 🚧 In Progress
- FAISS removal implementation
- Multi-dimensional edge implementation
- Quantum geDIG research

### 📋 Planned
- Insight episode message passing
- Old implementation cleanup
- Test coverage improvements

## Priority Order

### 🔴 Phase 1: FAISS Removal (2-4 weeks)
**Documents:**
- `1_faiss_removal_implementation_plan.md` ← **CURRENT FOCUS**
- `faiss_dependency_analysis.md`

**Tasks:**
1. Implement NumPy-based nearest neighbor search
2. Create vector index factory for backward compatibility
3. Performance benchmarking
4. Migration guide for existing users

**Why:** Simplifies installation, improves platform compatibility, and enables pure Python deployment.

### 🟡 Phase 2: Multi-Dimensional Edges (3-4 weeks)
**Documents:**
- `2_multi_dimensional_edge_implementation_plan.md`
- `multi_dimensional_edge_design.md`

**Tasks:**
1. Implement EdgeDimensions dataclass
2. Update graph builders to support multi-dimensional edges
3. Enhance search with dimension-specific queries
4. GNN integration for edge learning

**Why:** Enables richer relationship representation and more sophisticated reasoning.

### 🟢 Phase 3: Insight Episode Message Passing (4-5 weeks)
**Documents:**
- `3_insight_episode_message_passing_plan.md`

**Tasks:**
1. Implement InsightEpisode class
2. Create message passing mechanism
3. Implement episode splitting logic
4. Integration with existing graph system

**Why:** Allows dynamic evolution of insights and autonomous knowledge refinement.

### 🔵 Phase 4: Clean Up & Optimization (2-3 weeks)
**Documents:**
- `pending_improvements_2025_01.md`
- `test_coverage_improvement_plan.md`

**Tasks:**
1. Remove deprecated code
2. Improve test coverage to >90%
3. Performance optimization
4. Documentation updates

## Research & Future Work

### Quantum geDIG (Research Phase)
**Documents:**
- `/docs/research/classical_to_quantum_gedig_evolution_2025_01.md`
- `/docs/research/quantum_gedig_implementation_strategy_2025_01.md`
- `/docs/research/gnn_in_vector_space_perspective_2025_01.md`

**Status:** Theoretical framework established, waiting for Phase 1-3 completion before implementation.

## Technical Debt

### High Priority
- Episode restoration from DataStore (layer2_working_memory.py:395)
- Graph statistics retrieval (layer2_memory_manager.py:694)
- Pattern logger embeddings integration

### Medium Priority
- External API integration plan
- Multimodal experiment implementation
- Layer 4 refactoring

### Low Priority
- Animation updates
- CLI improvements
- License compatibility review

## Development Guidelines

### Code Quality Standards
- All new code must have tests
- Type hints required
- Documentation for public APIs
- No new dependencies without discussion

### Review Process
1. Feature branches for all changes
2. PR with tests and documentation
3. Performance benchmarks for critical paths
4. Update relevant diagrams

## Success Metrics

### Phase 1 (FAISS Removal)
- ✅ All tests pass without FAISS
- ✅ Performance degradation < 2x for datasets < 10k
- ✅ Zero installation failures

### Phase 2 (Multi-Dimensional Edges)
- ✅ Edge queries 10x more expressive
- ✅ Backward compatibility maintained
- ✅ New insights discovered in test data

### Phase 3 (Message Passing)
- ✅ Autonomous insight evolution working
- ✅ Episode splitting reduces redundancy by 30%
- ✅ Knowledge quality improves over time

### Phase 4 (Clean Up)
- ✅ Test coverage > 90%
- ✅ All TODOs addressed or documented
- ✅ Performance benchmarks established

## Notes

- Quantum geDIG remains in research phase until core improvements are stable
- External collaborations (paper reviews, etc.) are tracked separately
- Monthly progress reviews to adjust priorities