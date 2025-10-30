---
status: active
category: meta
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# CLI Test Coverage Analysis Report

## Summary

The project underwent a major CLI refactoring with:
- **Old CLI**: `insightspike` (legacy) - defined in `src/insightspike/cli/main.py`
- **New CLI**: `spike` (recommended) - defined in `src/insightspike/cli/improved_cli.py`

## Current State of CLI Commands

### New "spike" CLI Commands (improved_cli.py)
1. **query** (alias: q, ask) - Query the knowledge base
2. **embed** (alias: e, learn, l) - Embed documents into knowledge base
3. **config** - Manage configuration settings
4. **chat** (alias: c) - Interactive chat mode
5. **stats** - Show agent statistics
6. **experiment** - Run experiments
7. **demo** - Run interactive demo
8. **insights** - Show discovered insights
9. **insights-search** - Search for insights
10. **version** - Show version information

### Legacy "insightspike" CLI Commands (main.py)
1. **legacy-ask** - Deprecated ask command
2. **load_documents** - Load documents (without graph update)
3. **legacy-stats** - Deprecated stats command
4. **config_info** - Show current configuration
5. **deps** - Dependency management subcommands

## Test Coverage Issues

### 1. New "spike" CLI Tests
- **Location**: `tests/test_improved_cli.py`
- **Status**: ❌ NOT running in CI
- **Issue**: The test file exists but is not included in CI pipeline
- **Import Error**: Test file has import issues when running with pytest

### 2. Legacy "insightspike" CLI Tests
- **Location**: `tests/unit/test_cli_main.py`
- **Status**: ✅ Running in CI (as part of unit tests)
- **Coverage**: Tests for ask, load_documents, stats, config_info commands
- **Issues**: Several tests are skipped due to "CLI command routing issue"

### 3. Dependency CLI Tests
- **Location**: `tests/integration/test_deps_cli.py` and `tests/development/test_deps_cli_dev.py`
- **Status**: ✅ Running in CI
- **Coverage**: Tests for dependency management commands

## CI Configuration Issues

The CI pipeline (`/.github/workflows/ci.yml`) runs:
1. Unit tests: `poetry run pytest tests/unit/`
2. Integration tests: `poetry run pytest tests/integration/`

**Problem**: The new `tests/test_improved_cli.py` is in the root tests directory and is NOT collected by CI.

## Recommendations

### 1. Fix Import Issues in test_improved_cli.py
The test file needs to be moved to proper location and import issues fixed.

### 2. Update CI Pipeline
Add explicit test collection for the improved CLI:
```yaml
- name: Run CLI tests
  run: |
    echo "🧪 Running CLI tests..."
    poetry run pytest tests/test_improved_cli.py -v --tb=short
    echo "✅ CLI tests completed"
```

### 3. Add Missing Test Coverage
The following new CLI commands lack test coverage:
- `spike insights` command
- `spike insights-search` command  
- `spike demo` command
- Error handling for invalid commands
- Command aliases (q, c, e, l)

### 4. Fix Skipped Tests
Several legacy CLI tests are skipped with "CLI command routing issue". These should be investigated and fixed.

### 5. Test Both CLIs
Ensure both the legacy `insightspike` and new `spike` commands are tested since both are still available.

## Scripts Requiring Tests

Based on untracked files in git status:
- No Python scripts found in root directory requiring tests
- All core functionality appears to be properly organized in src/

## Conclusion

The new CLI refactoring introduced the `spike` command with many new features, but:
1. ❌ The tests for the new CLI are not running in CI
2. ❌ Some new commands lack test coverage
3. ⚠️ Import issues prevent the new CLI tests from running
4. ✅ Legacy CLI tests are running but have some skipped tests
5. ✅ Dependency management CLI is well tested

The main action item is to fix the test file location/imports and ensure CI runs all CLI tests.