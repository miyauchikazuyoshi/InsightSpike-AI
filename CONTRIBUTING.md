# Contributing to InsightSpike-AI

Thank you for your interest in contributing to InsightSpike-AI! This guide will help you get started with contributing to our insight detection and graph-based learning library.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that promotes an inclusive and respectful environment for all contributors. Please read and follow our community guidelines:

- **Be respectful**: Treat all community members with courtesy and professionalism
- **Be inclusive**: Welcome contributors from all backgrounds and experience levels
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together to improve the project
- **Be patient**: Remember that everyone is learning and growing

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Git for version control
- Basic understanding of graph theory and machine learning concepts

### First-time Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/InsightSpike-AI.git
   cd InsightSpike-AI
   ```

3. **Set up the development environment**:
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   
   # Install PyTorch (CPU version for development)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # Install FAISS
   pip install faiss-cpu
   ```

4. **Verify your setup**:
   ```bash
   poetry run python -c "import insightspike; print('Setup successful!')"
   poetry run pytest tests/test_benchmarks.py -v
   ```

## Development Environment

### Using Poetry (Recommended)

```bash
# Activate the virtual environment
poetry shell

# Run tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_benchmarks.py -v

# Run linting
poetry run flake8 src/
poetry run black src/ --check
poetry run isort src/ --check-only
```

### Using Docker

```bash
# Development environment
docker-compose up insightspike-dev

# Run tests in container
docker-compose run insightspike-test

# Run benchmarks
docker-compose run insightspike-benchmark
```

### Environment Variables

Create a `.env` file for local development:
```env
INSIGHTSPIKE_MODE=development
PYTHONPATH=src:$PYTHONPATH
INSIGHTSPIKE_LOG_LEVEL=DEBUG
```

## Version Alignment — Pydantic v2 Migration Plan

We currently support Pydantic v1 in the core package. Some experimental submodules may use v2 APIs. To reduce confusion:

- Short‑term (v0.1.x): core remains on Pydantic v1; experiments that need v2 should vendor minimal adapters.
- Mid‑term (v0.2): introduce a tiny compat layer (e.g., `insightspike.utils.pydantic_compat`) exposing `model_dump`/`model_json` shims for v1.
- Migration rules:
  - Do not call `.dict()`/.json() directly in new code; use a compat helper (to be provided)
  - Avoid referencing `BaseModel.Config`; prefer field validators compatible with v2
  - Keep public config models stable (documented in README/Docs)
- When the core flips to v2, the compat shims will forward to native v2 and v1 paths will be deprecated for one minor.

## Stable Public API Surface

The following modules are considered stable (semantic‑versioned):
- `insightspike.public` (create_agent, load_config wrapper)
- CLI (`python -m insightspike.cli.spike`)

Internal modules may change between minors. If you need stability guarantees, open an issue to discuss promoting specific functions to the public surface.

## Coverage & Testing Targets

- Coverage threshold is enforced via coverage.py; target will increase over time:
  - v0.1: 35%
  - v0.2: 50% (focus: core metrics, datastore, gating)
- Use `make coverage` (added) to run with lightweight env flags.

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Address issues in existing functionality
2. **New Features**: Add new capabilities or algorithms
3. **Performance Improvements**: Optimize existing code
4. **Documentation**: Improve guides, examples, and API docs
5. **Tests**: Add or improve test coverage
6. **Examples**: Create demonstrations and tutorials

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our coding standards
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run the full test suite**
6. **Commit your changes** with descriptive messages
7. **Push and create a pull request**

### Coding Standards

#### Python Style Guide

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Maximum line length: 88 characters (Black formatter default)

#### Example Code Style

```python
from typing import List, Dict, Optional, Tuple
import numpy as np


class ExampleAlgorithm:
    """Example algorithm implementation.
    
    This class demonstrates our coding standards and documentation style.
    
    Args:
        parameter: Description of the parameter
        config: Optional configuration dictionary
        
    Attributes:
        processed_data: Stores processed results
        metadata: Algorithm metadata and statistics
    """
    
    def __init__(self, parameter: str, config: Optional[Dict] = None):
        self.parameter = parameter
        self.config = config or {}
        self.processed_data: List[np.ndarray] = []
        self.metadata: Dict[str, float] = {}
    
    def process_data(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process input data and return results with metadata.
        
        Args:
            input_data: Input array to process
            
        Returns:
            Tuple containing processed data and metadata dictionary
            
        Raises:
            ValueError: If input_data is empty or invalid
        """
        if input_data.size == 0:
            raise ValueError("Input data cannot be empty")
        
        # Process data
        result = self._internal_processing(input_data)
        
        # Generate metadata
        metadata = {
            'input_shape': input_data.shape,
            'processing_time': 0.0,  # Measure actual time
            'data_quality': self._assess_quality(result)
        }
        
        return result, metadata
    
    def _internal_processing(self, data: np.ndarray) -> np.ndarray:
        """Internal processing method (private)."""
        return data * 2  # Simplified example
    
    def _assess_quality(self, data: np.ndarray) -> float:
        """Assess data quality score."""
        return float(np.mean(data))
```

#### Algorithm Implementation Guidelines

- **Modularity**: Keep algorithms in separate, focused modules
- **Performance**: Consider computational complexity and memory usage
- **Robustness**: Handle edge cases and validate inputs
- **Documentation**: Explain algorithm choices and parameters
- **Testing**: Provide comprehensive test coverage

### Project Structure

```
src/insightspike/
├── algorithms/          # Core algorithms (GED, IG, etc.)
├── core/               # Main detection and processing logic
├── detection/          # Insight detection components
├── metrics/            # Evaluation and measurement tools
├── processing/         # Data processing utilities
├── training/           # Model training components
└── utils/              # General utilities

tests/
├── test_algorithms/    # Algorithm-specific tests
├── test_integration/   # Integration tests
└── test_benchmarks.py  # Performance benchmarks

benchmarks/             # Performance testing suite
docs/                   # Documentation
experiments/            # Research and experimental code
```

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Benchmark Tests**: Performance and scalability testing
4. **Regression Tests**: Ensure consistent behavior across versions

### Writing Tests

```python
import pytest
import numpy as np
from insightspike.algorithms.example import ExampleAlgorithm


class TestExampleAlgorithm:
    """Test suite for ExampleAlgorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.algorithm = ExampleAlgorithm("test_param")
        self.test_data = np.array([1, 2, 3, 4, 5])
    
    def test_process_data_basic(self):
        """Test basic data processing functionality."""
        result, metadata = self.algorithm.process_data(self.test_data)
        
        assert result.shape == self.test_data.shape
        assert 'input_shape' in metadata
        assert metadata['input_shape'] == self.test_data.shape
    
    def test_process_data_empty_input(self):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            self.algorithm.process_data(np.array([]))
    
    @pytest.mark.parametrize("input_size", [10, 100, 1000])
    def test_scalability(self, input_size):
        """Test algorithm scalability with different input sizes."""
        large_data = np.random.rand(input_size)
        result, metadata = self.algorithm.process_data(large_data)
        
        assert result.shape[0] == input_size
        assert metadata['processing_time'] < 1.0  # Performance requirement
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=insightspike --cov-report=html

# Run specific test categories
poetry run pytest tests/test_algorithms/ -v
poetry run pytest tests/test_benchmarks.py -v

# Run performance benchmarks
poetry run python benchmarks/performance_suite.py
```

## Documentation

### Documentation Standards

- **API Documentation**: Use docstrings with Google or NumPy style
- **Examples**: Include practical code examples
- **Tutorials**: Step-by-step guides for common use cases
- **Architecture**: Explain design decisions and trade-offs

### Building Documentation

```bash
# Install documentation dependencies
poetry install --with docs

# Build documentation locally
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: Run the full test suite
2. **Update documentation**: Include relevant doc changes
3. **Add changelog entry**: Document user-facing changes
4. **Rebase on main**: Ensure clean commit history

### PR Checklist

- [ ] Tests added or updated for new functionality
- [ ] Documentation updated (docstrings, guides, examples)
- [ ] Code follows project style guidelines
- [ ] Performance benchmarks pass (if applicable)
- [ ] Backward compatibility maintained (or breaking changes documented)
- [ ] Issue linked (if applicable)

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Benchmarks run successfully
- [ ] Manual testing completed

## Documentation
- [ ] Docstrings updated
- [ ] User guides updated
- [ ] Examples added/updated
- [ ] Changelog entry added

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests pass locally
- [ ] Documentation builds successfully
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests and linting
2. **Code review**: Maintainers review code quality and design
3. **Testing**: Verify functionality and performance
4. **Documentation review**: Ensure clarity and completeness
5. **Approval and merge**: Final approval and integration

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, discussions
- **GitHub Discussions**: Community questions and ideas
- **Pull Requests**: Code contributions and reviews

### Getting Help

1. **Check documentation**: Start with README and QUICK_START
2. **Search existing issues**: Your question might already be answered
3. **Create an issue**: Use appropriate issue template
4. **Join discussions**: Participate in community conversations

### Recognition

We value all contributions! Contributors are recognized through:
- **Contributors file**: Listed in project contributors
- **Release notes**: Acknowledged in version releases
- **Community mentions**: Highlighted in project communications

## Development Roadmap

See our [project roadmap](documentation/ARCHITECTURE_EVOLUTION_ROADMAP.md) for planned features and improvements.

### Areas for Contribution

Current priority areas:
- **Algorithm optimization**: Improve performance of core algorithms
- **Educational tools**: Develop learning and teaching utilities
- **Integration examples**: Create real-world application demos
- **Documentation**: Expand guides and tutorials
- **Testing**: Increase test coverage and robustness

---

Thank you for contributing to InsightSpike-AI! Your efforts help make this tool more valuable for researchers, educators, and developers working with insight detection and graph-based learning.
