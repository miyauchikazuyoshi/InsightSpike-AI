# InsightSpike Refactoring Plan (2025-01-27)

## Current Issues

### 1. Configuration System
- **Problem**: Pydantic models are too rigid for experimental use
  - Cannot add fields dynamically (e.g., `use_multihop_gedig`)
  - Runtime configuration changes are difficult
- **Impact**: Experiments require workarounds and legacy config conversions

### 2. Memory-LLM Integration
- **Problem**: Knowledge added via `add_knowledge()` isn't accessible to LLM
  - Memory search errors: "Found array with dim 3"
  - LLM responses: "I don't have enough information"
- **Impact**: Core functionality broken - cannot demonstrate insight detection

### 3. Error Handling
- **Problem**: Many warnings but system continues with degraded functionality
  - "Adjacency matrix not square" errors
  - "Memory search failed" errors
- **Impact**: Difficult to debug, unclear what's actually working

### 4. Provider Implementation
- **Problem**: Even with real Claude API, responses are generic
  - Knowledge retrieval not working
  - Context not being passed to LLM properly
- **Impact**: Cannot differentiate from MockProvider

## Proposed Solutions

### Phase 1: Create Minimal Working Core
1. Create a simplified agent that bypasses complex layers
2. Direct integration between memory store and LLM
3. Focus on core insight detection functionality

### Phase 2: Fix Configuration System
1. Add runtime config overlay system
2. Allow experimental parameters without modifying Pydantic models
3. Create builder pattern for complex configurations

### Phase 3: Improve Memory-LLM Pipeline
1. Fix memory search dimension issues
2. Ensure retrieved knowledge is properly formatted for LLM
3. Add logging to trace knowledge flow

### Phase 4: Better Error Handling
1. Add error recovery mechanisms
2. Clear error messages with actionable fixes
3. Validation at each pipeline stage

## Implementation Order

1. **Immediate**: Create `SimpleInsightAgent` for experiments
2. **Short-term**: Fix memory search issues
3. **Medium-term**: Improve configuration flexibility
4. **Long-term**: Full system refactoring

## Success Criteria

- Mathematical experiment shows actual insight detection
- Knowledge retrieval works with real LLM providers
- No critical errors during normal operation
- Configuration changes don't require code modifications