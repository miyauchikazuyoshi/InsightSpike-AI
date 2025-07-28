# Test Coverage Improvement Plan

## Current Status (2024-07-18)
- **Current Coverage**: 22.82%
- **Target Coverage**: 40% (temporarily lowered to 20%)
- **Total Tests**: 43 (all passing)

## Coverage Analysis by Module

### High Coverage Modules (>70%)
- `config/constants.py`: 100%
- `config/legacy_config.py`: 99%
- `config/__init__.py`: 100%
- `config/loader.py`: 72%

### Medium Coverage Modules (30-70%)
- `cli/spike.py`: 56%
- `cli/__init__.py`: 60%
- `algorithms/__init__.py`: 56%
- `utils/__init__.py`: 56%

### Low Coverage Modules (<30%)
- `__main__.py`: 0%
- `algorithms/metrics_selector.py`: 0%
- `algorithms/pyg_adapter.py`: 0%
- `cli/commands/deps.py`: 0%
- `training/*`: 0%
- `visualization/dashboards.py`: 10%

## Improvement Strategy

### Phase 1: Quick Wins (Target: 30% coverage)
**Timeline**: 1-2 days
**Effort**: Low

1. **CLI Commands** (`cli/spike.py`)
   - Add tests for `experiment` command
   - Add tests for `demo` command
   - Add tests for error cases in `interactive` command
   - Expected coverage gain: +3-4%

2. **Utility Functions** (`utils/text_utils.py`)
   - Test `clean_text` function
   - Test `truncate_text` function
   - Expected coverage gain: +1%

3. **Simple Initialization Tests**
   - Test module imports and initialization
   - Test `__main__.py` basic functionality
   - Expected coverage gain: +2%

### Phase 2: Core Functionality (Target: 35% coverage)
**Timeline**: 3-5 days
**Effort**: Medium

1. **MainAgent Tests** (`implementations/agents/main_agent.py`)
   - Test `process_question` method
   - Test `add_knowledge` method
   - Test `get_stats` method
   - Expected coverage gain: +3-4%

2. **Config Loader Edge Cases** (`config/loader.py`)
   - Test error handling paths
   - Test preset merging logic
   - Test environment variable handling
   - Expected coverage gain: +1%

3. **Algorithm Basic Tests**
   - Test basic functionality of `metrics_selector.py`
   - Test adapter patterns in `pyg_adapter.py`
   - Expected coverage gain: +2%

### Phase 3: Comprehensive Coverage (Target: 40%+)
**Timeline**: 1-2 weeks
**Effort**: High

1. **Integration Tests**
   - End-to-end CLI workflow tests
   - Full agent initialization and processing
   - Expected coverage gain: +3%

2. **Algorithm Deep Tests**
   - Test entropy calculations
   - Test graph edit distance computations
   - Test information gain calculations
   - Expected coverage gain: +2%

3. **Error Handling and Edge Cases**
   - Test all error paths
   - Test boundary conditions
   - Test concurrent operations
   - Expected coverage gain: +2%

## Implementation Examples

### Example 1: CLI Command Test
```python
def test_experiment_command(runner, mock_factory, mock_agent):
    """Test the experiment command."""
    with patch.object(mock_factory, "get_agent", return_value=mock_agent):
        with patch("insightspike.cli.spike.ExperimentRunner") as MockRunner:
            mock_runner = Mock()
            mock_runner.run.return_value = {
                "type": "simple",
                "episodes": [{"question": "test", "success": True}]
            }
            MockRunner.return_value = mock_runner
            
            result = runner.invoke(
                app, ["experiment", "--name", "simple", "--episodes", "5"],
                obj=mock_factory
            )
            
            assert result.exit_code == 0
            assert "Experiment completed" in result.stdout
```

### Example 2: Utility Function Test
```python
def test_clean_text():
    """Test text cleaning function."""
    from insightspike.utils.text_utils import clean_text
    
    assert clean_text("  Hello  World  ") == "Hello World"
    assert clean_text("Line1\n\nLine2") == "Line1 Line2"
    assert clean_text("") == ""
```

## Testing Best Practices

1. **Focus on Business Logic**: Prioritize testing core functionality over boilerplate
2. **Use Mocks Wisely**: Mock external dependencies but test real logic
3. **Test Edge Cases**: Include boundary conditions and error scenarios
4. **Maintain Test Quality**: Well-structured, readable tests are easier to maintain

## Monitoring Progress

Track coverage improvements with:
```bash
# Generate detailed coverage report
poetry run pytest tests/unit/ --cov=src/insightspike --cov-report=html

# View coverage by module
poetry run pytest tests/unit/ --cov=src/insightspike --cov-report=term-missing
```

## Next Steps

1. Start with Phase 1 quick wins
2. Update CI coverage threshold incrementally (25% → 30% → 35% → 40%)
3. Document any modules that should be excluded from coverage
4. Consider adding integration tests as a separate test suite

## Notes

- Some modules (like `visualization/dashboards.py`) may not need high coverage if they're primarily UI code
- Training modules might be better tested with integration tests
- Focus on testing public APIs and critical business logic first