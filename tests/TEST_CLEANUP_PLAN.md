# Test Cleanup and CI Improvement Plan

## Test Directory Issues

### 1. Duplicate Files to Remove
```bash
# Remove duplicate _fixed files (keep the fixed versions)
rm tests/test_dependency_commands.py
rm tests/test_dependency_resolver.py
rm tests/test_platform_utils.py
rm tests/test_poetry_integration.py

# Rename fixed files to original names
mv tests/test_dependency_commands_fixed.py tests/test_dependency_commands.py
mv tests/test_dependency_resolver_fixed.py tests/test_dependency_resolver.py
mv tests/test_platform_utils_fixed.py tests/test_platform_utils.py
mv tests/test_poetry_integration_fixed.py tests/test_poetry_integration.py
```

### 2. Missing Tests for New Components
Need to add tests for:
- `ScalableGraphManager`
- `L2EnhancedScalableMemory`
- New data directory structure
- Experiment management workflow

### 3. Test Organization
Current structure is good:
- `unit/` - Unit tests for individual components
- `integration/` - Integration tests
- `validation/` - System validation tests
- `development/` - Development utilities

## CI Configuration Issues

### 1. Current CI is Good ✅
- Uses Poetry for dependency management
- Runs on Ubuntu with Python 3.10
- Has proper timeout (15 minutes)
- Uses CPU-only PyTorch for faster CI

### 2. Potential Improvements

#### Add test categories:
```yaml
- name: Run unit tests
  run: poetry run pytest tests/unit -v --tb=short

- name: Run integration tests
  run: poetry run pytest tests/integration -v --tb=short

- name: Run quick validation
  run: poetry run pytest tests/validation/test_quick_validation.py -v
```

#### Add code quality checks:
```yaml
- name: Code formatting check
  run: poetry run black --check src tests

- name: Import sorting check
  run: poetry run isort --check-only src tests

- name: Type checking
  run: poetry run mypy src --ignore-missing-imports
```

#### Add test coverage:
```yaml
- name: Run tests with coverage
  run: poetry run pytest --cov=src --cov-report=xml --cov-report=term

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Recommended Actions

1. **Immediate**: Clean up duplicate test files
2. **Short-term**: Add tests for new ScalableGraphManager components
3. **Medium-term**: Add code quality checks to CI
4. **Long-term**: Set up coverage tracking

## Test File Status

### Keep These (Core Tests):
- `test_core_integration.py`
- `test_mvp_integration.py`
- `test_reusability_comprehensive.py`
- `conftest.py` (pytest configuration)

### Clean Up These:
- Remove `_fixed` duplicates
- Update imports in remaining tests

### Add These:
- `test_scalable_graph_manager.py`
- `test_enhanced_memory.py`
- `test_data_management.py`