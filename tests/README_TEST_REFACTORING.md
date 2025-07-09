# Test Refactoring Guide

This document explains the new test organization and best practices for writing maintainable tests.

## New Test Structure

```
tests/
├── conftest.py          # Pytest configuration and global fixtures
├── fixtures/            # Reusable test fixtures
│   └── graph_fixtures.py
├── factories/           # Mock object factories
│   └── mock_factory.py
├── helpers/             # Test helper utilities
│   └── test_helpers.py
├── unit/               # Unit tests
└── integration/        # Integration tests
```

## Key Improvements

### 1. Centralized Fixtures (fixtures/)
- **graph_fixtures.py**: Pre-defined graph structures for consistent testing
- Fixtures are automatically available in all tests via conftest.py
- Example usage:
```python
def test_my_graph_function(simple_graph, complex_graph):
    # simple_graph and complex_graph are automatically injected
    result = my_function(simple_graph)
    assert result is not None
```

### 2. Mock Factories (factories/)
- **GraphFactory**: Create customizable graphs
- **EmbeddingFactory**: Generate test embeddings
- **MemoryFactory**: Create mock memory components
- **ConfigFactory**: Generate test configurations
- Example usage:
```python
from tests.factories.mock_factory import graph_factory

def test_dynamic_graph():
    graph = graph_factory.create_simple_graph(num_nodes=10)
    # Use the dynamically created graph
```

### 3. Test Helpers (helpers/)
- **Assertion helpers**: `assert_graphs_equal`, `assert_embeddings_similar`
- **Creation helpers**: `create_test_embedding`, `create_mock_config_object`
- **Performance tracking**: `PerformanceMetrics` class
- Example usage:
```python
from tests.helpers.test_helpers import assert_graphs_similar

def test_graph_transformation():
    g1 = create_graph()
    g2 = transform_graph(g1)
    assert_graphs_similar(g1, g2, node_tolerance=2)
```

## Benefits

1. **Reduced Duplication**: Common test data is defined once
2. **Better Maintainability**: Changes to test data structure only require updates in one place
3. **Consistent Testing**: All tests use the same base fixtures
4. **Easier Test Writing**: Rich set of helpers and factories
5. **Decoupled from Implementation**: Tests focus on behavior, not implementation details

## Migration Guide

### Before (Old Style)
```python
def test_delta_ged():
    # Creating mock directly in test
    g1 = Mock()
    g1.nodes = Mock(return_value=[1, 2, 3])
    g1.edges = Mock(return_value=[(1, 2)])
    
    g2 = Mock()
    g2.nodes = Mock(return_value=[1, 2, 3, 4])
    g2.edges = Mock(return_value=[(1, 2), (2, 3)])
    
    result = delta_ged(g1, g2)
    assert result > 0
```

### After (New Style)
```python
def test_delta_ged(graph_pair_similar):
    # Using fixtures
    g1, g2 = graph_pair_similar
    
    result = delta_ged(g1, g2)
    assert result > 0
```

## Best Practices

1. **Use Fixtures for Static Test Data**
   - Pre-defined graphs, configurations, etc.
   - Data that doesn't change between tests

2. **Use Factories for Dynamic Test Data**
   - When you need customizable test objects
   - When test data depends on test parameters

3. **Use Helpers for Common Operations**
   - Complex assertions
   - Performance measurements
   - Data validation

4. **Keep Tests Focused**
   - Test one behavior per test function
   - Use descriptive test names
   - Avoid testing implementation details

## Test Markers

The following pytest markers are available:

- `@pytest.mark.slow`: For tests that take longer than 1 second
- `@pytest.mark.integration`: For integration tests
- `@pytest.mark.unit`: For unit tests (default)
- `@pytest.mark.requires_gpu`: For tests requiring GPU

Example:
```python
@pytest.mark.slow
def test_large_graph_processing(complex_graph):
    # This test takes time
    pass
```

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run without slow tests
pytest -m "not slow"

# Run with specific fixture
pytest -k "simple_graph"
```