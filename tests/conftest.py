"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import all fixtures to make them available
from tests.fixtures.graph_fixtures import *
from tests.factories.mock_factory import (
    graph_factory,
    embedding_factory,
    memory_factory,
    config_factory,
    llm_factory,
    document_factory,
)
from tests.helpers.test_helpers import (
    assert_graphs_equal,
    assert_graphs_similar,
    assert_embeddings_similar,
    create_test_embedding,
    create_mock_config_object,
    PerformanceMetrics,
)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    return create_mock_config_object()


@pytest.fixture
def performance_tracker():
    """Provide a performance metrics tracker."""
    return PerformanceMetrics()


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    import numpy as np
    import random

    np.random.seed(42)
    random.seed(42)
    yield
    # No cleanup needed


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


# Global test configuration
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "default_embedding_dim": 384,
        "default_timeout": 5.0,
        "max_test_samples": 100,
        "lite_mode": True,  # For CI environment
    }
