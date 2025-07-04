# InsightSpike-AI Test Suite

## Overview

This directory contains the comprehensive test suite for InsightSpike-AI, organized to match the current implementation including Phase 2 & 3 scalability improvements.

## Test Organization

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_layer2_memory_manager.py    # Graph-centric memory (C-value free)
│   ├── test_scalable_graph_builder.py   # Phase 2: O(n log n) implementation
│   ├── test_hierarchical_graph_builder.py # Phase 3: Hierarchical structure
│   ├── test_graph_centric_memory.py     # Dynamic importance calculation
│   └── ...                              # Other component tests
│
├── integration/               # Integration tests
│   ├── test_scalable_system.py          # Full system integration
│   ├── phase2_phase3/                   # Phase 2 & 3 specific tests
│   │   ├── test_phase3_minimal.py       # Minimal hierarchical test
│   │   ├── test_phase3_simple.py        # Simple performance display
│   │   ├── test_rag_performance_light.py # RAG performance comparison
│   │   └── ...                          # Other integration tests
│   └── ...                              # Other integration categories
│
└── conftest.py               # Shared fixtures and mocks (C-value free)
```

## Key Test Updates

### Phase 2 & 3 Implementation Tests

1. **Scalable Graph Builder** (`test_scalable_graph_builder.py`)
   - FAISS-based approximate nearest neighbor
   - O(n log n) complexity verification
   - Top-k neighbor selection

2. **Hierarchical Graph** (`test_hierarchical_graph_builder.py`)
   - 3-layer structure (Episodes → Clusters → Super-clusters)
   - O(log n) search complexity
   - Dynamic document addition

3. **Graph-Centric Memory** (`test_graph_centric_memory.py`)
   - No C-values in episodes
   - Dynamic importance from graph structure
   - Graph-informed integration/splitting

### Removed/Updated Components

- **C-value references**: All removed from mocks and tests
- **Old Layer2/3 tests**: Updated to match new implementations
- **Experiment files**: Moved to `experiments/archive/`

## Running Tests

### Quick Test
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific Phase 2/3 tests
pytest tests/integration/phase2_phase3/
```

### CI Integration
The GitHub Actions CI runs:
- Unit tests with mocking
- Integration tests for scalability
- Performance verification

### Test Configuration
Tests run with:
- `INSIGHTSPIKE_LITE_MODE=1` (avoid GPU requirements)
- Mock LLM providers
- Reduced dataset sizes for speed

## Test Coverage

### What's Tested
- ✅ Scalable graph construction (O(n log n))
- ✅ Hierarchical search (O(log n))
- ✅ Graph-centric episode management
- ✅ Dynamic importance calculation
- ✅ Integration without C-values
- ✅ Automatic splitting based on conflicts
- ✅ Memory optimization
- ✅ Multi-scale performance

### What's Not Tested (Yet)
- ⏳ 100K+ episode stress tests
- ⏳ GPU acceleration
- ⏳ Distributed processing
- ⏳ Multimodal integration

## Writing New Tests

When adding tests:
1. No C-value parameters in episodes
2. Use graph-centric memory manager
3. Test scalability characteristics
4. Verify O(log n) search complexity

Example:
```python
def test_new_feature(self):
    # Use GraphCentricMemoryManager, not old Memory class
    manager = GraphCentricMemoryManager(dim=384)
    
    # Add episodes without c_value
    idx = manager.add_episode(vector, text)  # No c_value!
    
    # Verify dynamic importance
    importance = manager.get_importance(idx)
    assert 0 <= importance <= 1
```

## Performance Benchmarks

Expected performance characteristics:

| Dataset Size | Build Time | Search Time | Memory |
|-------------|------------|-------------|---------|
| 1K docs     | <200ms     | <1ms        | ~5MB    |
| 10K docs    | <2s        | <5ms        | ~50MB   |
| 100K docs   | <20s       | <10ms       | ~500MB  |

These are achieved through:
- FAISS approximate search
- Hierarchical indexing
- Graph-based compression