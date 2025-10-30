---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# Pending Improvements - January 2025

## Overview
This document consolidates all pending improvements, TODOs, and action items for the InsightSpike-AI project as of July 2025.

## Priority Legend
- 🔴 **Critical** - Blocking core functionality or paper requirements
- 🟡 **High** - Important for production readiness
- 🟢 **Medium** - Nice to have, performance improvements
- 🔵 **Low** - Future enhancements

## 1. Code Implementation TODOs

### 🔴 Critical Implementation Gaps

#### Episode Management
- **Location**: `src/insightspike/implementations/layers/layer2_working_memory.py:395`
- **Issue**: Episodes not being restored from persistent storage
- **TODO**: `# TODO: Load actual episodes from DataStore using IDs`
- **Impact**: Memory persistence not working properly

#### Graph Statistics
- **Location**: `src/insightspike/implementations/layers/layer2_memory_manager.py:694`
- **Issue**: Graph stats retrieval not implemented for ScalableGraphBuilder
- **TODO**: `# TODO: Implement graph stats retrieval for ScalableGraphBuilder`
- **Impact**: Can't get graph metrics for analysis

#### ErrorMonitor analyze_input Function (Fixed ✓)
- **Location**: `src/insightspike/implementations/layers/layer1_error_monitor.py`
- **Issue**: `analyze_uncertainty()` calls undefined `analyze_input()` function
- **TODO**: Fix NameError or implement the function
- **Impact**: Layer1 uncertainty analysis will crash if called
- **Status**: ✅ Function is already implemented in the same file

### 🟡 High Priority Features

#### Pattern Logger Embeddings
- **Location**: `src/insightspike/learning/pattern_logger.py:141,279`
- **Issue**: Embeddings integration incomplete
- **TODOs**: 
  - `# TODO: Use embeddings when available`
  - `# TODO: Implement when embeddings are accessible`
- **Impact**: Pattern matching less accurate without embeddings

## 2. Code Review Action Items (From July 2024 Reviews)

### 🔴 Critical Fixes

#### ~~Missing Components (DeepResearch Review)~~
**Update**: These were hallucinations in the original review:
- GraphMerger - Never existed
- EpisodePruner - Never existed
- split_large_node() - Never existed
- agent_retriever/episode_linker - Never existed

#### Method/Attribute Mismatches (Fixed ✓)
- [x] Method name mismatches (merge, split, prune) - Fixed with aliases
- [x] Attribute mismatch (scalable_graph vs graph_builder) - Fixed
- [x] Field name inconsistency (c_value vs c) - Fixed

#### Episode Attribute Naming Issue (Fixed ✓)
- **Issue**: Episode class uses `c` but some code references `c_value`
- **Location**: MainAgent graph rebuild fallback
- **Impact**: Always falls back to 0.5 confidence in graph analysis
- **Fix**: Added property `c_value` to Episode class - COMPLETED

### 🟡 Architecture Improvements

#### GraphBuilder Consolidation
- **Issue**: GraphBuilder and ScalableGraphBuilder have duplicate functionality
- **Action**: Unify into single implementation
- **Benefit**: Better maintainability

#### Legacy Code Cleanup
- **Issue**: CompatibleL2MemoryManager still used via MainAgent
- **Action**: Update MainAgent to use L2MemoryManager directly
- **Benefit**: Remove compatibility layer

### 🟢 Quality Improvements

#### Test Coverage
- **Current**: 22.82%
- **Target**: 40%
- **Needed**: Integration tests for merge/split/prune operations

#### Conflict Detection
- **Current**: Simple negation detection only
- **Needed**: Sophisticated contradiction detection algorithms

## 3. Paper Review Requirements

### 🔴 Statistical Validation

1. **Real LLM Testing**
   - Current: Only MockProvider tested
   - Needed: OpenAI/Anthropic provider validation
   - Action: Run comprehensive_gedig_evaluation with real LLMs

2. **Baseline Comparisons**
   - Needed: RAG, ConceptNet, GraphQA baselines
   - Action: Implement baseline framework

3. **Standard Benchmarks**
   - Needed: CommonsenseQA, ConceptNet QA, SciQ, ARC evaluation
   - Action: Create benchmark integration

### 🟡 Implementation Completeness

1. **Episode Merge Operations**
   - Status: Only split implemented, merge missing
   - Impact: Can't consolidate related episodes

2. **Memory Reorganization**
   - Status: Not implemented
   - Impact: Memory doesn't optimize over time

3. **Cost Function Alignment**
   - Issue: Parameters don't match paper specifications
   - Action: Align implementation with theoretical model

## 4. Experiment V2 Issues

### 🔴 Provider Problems

1. **LocalProvider Issues**
   - Status: Implemented but problematic
   - Issues: Slow init, high memory with TinyLlama
   - Recommendation: Use external APIs

2. **LLM Initialization**
   - Problem: Provider validation missing
   - Solution: Pre-validate before experiments

### 🟡 Experiment Improvements

1. **Progress Tracking**
   - Need: Progress bars for long experiments
   - Benefit: Better user experience

2. **Error Handling**
   - Need: Retry mechanism for LLM calls
   - Need: Better episode storage error handling

3. **Memory Efficiency**
   - Issue: High memory usage in large experiments
   - Solution: Streaming/batching implementation

## 5. Test Coverage Improvement Plan

### Phase 1: Quick Wins (Target: 30%)

- [ ] CLI command tests (experiment, demo)
- [ ] Utility function tests (text_utils)
- [ ] Module initialization tests

### Phase 2: Core Functions (Target: 35%)

- [ ] MainAgent method tests
- [ ] Config loader edge cases
- [ ] Basic algorithm tests

### Phase 3: Comprehensive (Target: 40%+)

- [ ] End-to-end integration tests
- [ ] Deep algorithm testing
- [ ] Error handling coverage

## 6. Architecture and Code Quality Issues (From GPT Review)

### 🔴 Critical Code Issues

1. **Undefined Function in ErrorMonitor** (Fixed ✓)
   - `analyze_uncertainty()` calls non-existent `analyze_input()`
   - Will cause NameError if Layer1 uncertainty is used
   - Currently dormant as MainAgent doesn't call it
   - **FIXED**: Function exists in the same file

2. **Episode Attribute Inconsistency** (Fixed ✓)
   - Episode class uses `c` but code references `c_value`
   - Causes fallback to default 0.5 in graph analysis
   - Affects confidence-weighted calculations
   - **FIXED**: Added c_value property to Episode class

### 🟡 Architecture Duplication

1. **Graph Building Logic**
   - GraphBuilder (Layer3) and ScalableGraphBuilder (Layer2) duplicate code
   - Different implementations for similar functionality
   - Risk of divergent behavior

2. **Memory Interface Fragmentation**
   - CompatibleL2MemoryManager adds complexity
   - Partial compatibility causes issues
   - Legacy support creates maintenance burden

### 🟢 Code Quality Issues

1. **Incomplete Feature Flags**
   - `use_importance_scoring` - logs "not implemented"
   - `use_hierarchical_graph` - referenced but unused
   - Misleading configuration options

2. **Unused Monitoring**
   - GraphOperationMonitor exists but not utilized
   - Performance tracking infrastructure unused

3. **FAISS Fallback Issues**
   - MockFaiss returns empty graphs
   - Silent degradation without proper FAISS
   - IVF index path hardcoded to False

## 7. Future Enhancements

### 🔵 Decoder Development (Future High Priority)
- geDIG Generative Grammar Decoder
- Concept Token Infrastructure
- Message Passing Implementation
- Bidirectional conceptual LLM

### 🔵 Layer 3 Improvements
- Priority 4: Proper Scaling - Normalization functions
- Priority 5: Information Thermodynamics Features
- Combined insight score calculation

## 8. Documentation Needs

### 🟡 High Priority Docs
1. **Implementation Status**
   - Clear feature completion matrix
   - Known limitations document

2. **Parameter Documentation**
   - Match paper specifications
   - Configuration guide updates

3. **Reproducibility Package**
   - Scripts for paper experiments
   - Data and results packaging

## 9. Immediate Action Items (Next Sprint)

### Week 1

1. [ ] Run v2 experiment with OpenAI/Anthropic
2. [ ] Fix episode storage persistence
3. [ ] Document LocalProvider limitations

### Week 2

1. [ ] Implement baseline RAG comparison
2. [ ] Add memory tracking to experiments
3. [ ] Create paper visualizations

### Week 3

1. [ ] GraphBuilder consolidation
2. [ ] Test coverage Phase 1
3. [ ] Standard benchmark integration

## Summary

The project has made significant progress with v2 implementing direct ΔGED/ΔIG calculations. GPT's assessment: **~60% complete** toward full geDIG vision.

### Completed ✓

- [x] Core geDIG metrics (ΔGED/ΔIG)
- [x] Basic episodic memory
- [x] Graph reasoning layer
- [x] Insight spike detection
- [x] Layer1 bypass mechanism
- [x] Auto-insight registration
- [x] Graph-based search (2-hop)
- [x] Learning mechanism framework
- [x] Method/attribute mismatch fixes
- [x] Configuration system unification

### Critical Gaps

- [ ] Episode merge operations
- [ ] Memory reorganization
- [ ] Episode persistence from DataStore
- [ ] Graph-enhanced retrieval (still vector-only)
- [ ] Concept-level graph construction
- [ ] Adaptive learning loops
- [x] ErrorMonitor analyze_input function - Already implemented
- [x] Episode c_value property - Fixed!

### Validation Needs

- [ ] Real LLM testing (OpenAI/Anthropic)
- [ ] Baseline comparisons (RAG)
- [ ] Standard benchmarks (CommonsenseQA, etc.)
- [ ] Integration testing for full cycles

### Quality Issues

- [ ] Low test coverage (23% → 40% target)
- [ ] Architecture duplication (GraphBuilder)
- [ ] Legacy code complexity (CompatibleL2MemoryManager)
- [ ] Incomplete feature stubs (use_importance_scoring, etc.)

The decoder breakthrough has shifted priorities, but these improvements remain important for a production-ready system.