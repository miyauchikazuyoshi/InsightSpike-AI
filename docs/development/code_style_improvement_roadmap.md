# Code Style Improvement Roadmap

## Overview
This document outlines the gradual improvement plan for code style and quality in the InsightSpike-AI project.

## Current Status (2025-01-13)
- ✅ Black formatting applied to all files
- ✅ isort import ordering fixed
- ⚠️ flake8 rules temporarily relaxed to pass CI
- ⚠️ mypy type checking temporarily disabled

## Improvement Phases

### Phase 1: Remove Unused Imports (Week 1)
**Goal**: Clean up all unused imports across the codebase

```bash
# Automated removal of unused imports
poetry run autoflake --remove-all-unused-imports --recursive src/ tests/

# Update CI configuration
# Remove F401 from ignore list in .github/workflows/ci.yml
```

**CI Configuration**:
```yaml
- run: poetry run flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,E501,F541,E722
```

### Phase 2: Fix f-string Issues (Week 2)
**Goal**: Fix all f-strings missing placeholders

```bash
# Automated f-string fixes
poetry run flynt src/ tests/

# Manual review for complex cases
```

**CI Configuration**:
```yaml
- run: poetry run flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,E501,E722
```

### Phase 3: Reduce Line Length (Week 3)
**Goal**: Gradually reduce maximum line length from 120 to 100 characters

```bash
# Check current violations
poetry run flake8 src/ tests/ --max-line-length=100 --select=E501

# Fix manually or with black configuration
```

**CI Configuration**:
```yaml
- run: poetry run flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,E722
```

### Phase 4: Fix Bare Except Statements (Week 4)
**Goal**: Replace all bare `except:` with specific exception types

```bash
# Find all bare except statements
grep -r "except:" src/ tests/

# Replace with specific exceptions like:
# except Exception as e:
# except (ValueError, TypeError) as e:
```

**CI Configuration**:
```yaml
- run: poetry run flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203
```

### Phase 5: Add Type Hints (Week 5-6)
**Goal**: Add comprehensive type hints and enable mypy

```bash
# Generate type stubs from runtime
poetry run monkeytype run -m pytest
poetry run monkeytype apply src.insightspike

# Fix mypy errors incrementally
poetry run mypy src/ --ignore-missing-imports
```

**CI Configuration**:
```yaml
- run: poetry run mypy src/ --ignore-missing-imports
```

### Phase 6: Full PEP8 Compliance (Week 7)
**Goal**: Achieve full PEP8 compliance with 79-character line limit

**CI Configuration**:
```yaml
- run: poetry run flake8 src/ tests/ --max-line-length=79
```

## Pre-commit Hook Setup

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=120"]
        additional_dependencies: [flake8-docstrings]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        args: ["--ignore-missing-imports"]
```

Install pre-commit:
```bash
poetry add --group dev pre-commit
poetry run pre-commit install
```

## Automation Tools

### Required Development Dependencies
```toml
[tool.poetry.group.dev.dependencies]
autoflake = "^2.0.0"
flynt = "^0.77"
monkeytype = "^23.3.0"
pre-commit = "^3.5.0"
```

### Useful Commands
```bash
# Check current code style violations
poetry run flake8 src/ tests/ --statistics

# Auto-format with black
poetry run black src/ tests/

# Sort imports
poetry run isort src/ tests/

# Remove unused imports
poetry run autoflake --remove-all-unused-imports --in-place --recursive src/

# Check type hints
poetry run mypy src/ --ignore-missing-imports
```

## Success Metrics
- [ ] 0 flake8 violations with standard configuration
- [ ] 100% of functions have type hints
- [ ] mypy passes without errors
- [ ] All CI checks pass without overrides
- [ ] Pre-commit hooks prevent style violations

## Timeline
- **Start Date**: 2025-01-20
- **Target Completion**: 2025-03-10
- **Review Frequency**: Weekly

## Notes
- Each phase should be completed in a single PR to maintain clean git history
- Run full test suite after each phase to ensure no regressions
- Document any exceptions or project-specific style decisions
- Consider team training on Python code style best practices