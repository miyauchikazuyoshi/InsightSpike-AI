"""Shared pytest fixtures and configuration for the test suite."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insightspike.config.models import InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
from insightspike.implementations.datastore.factory import DataStoreFactory


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


# Global fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-scoped temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def config_development():
    """Development configuration preset."""
    return ConfigPresets.development()


@pytest.fixture
def config_experiment():
    """Experiment configuration preset."""
    return ConfigPresets.experiment()


@pytest.fixture
def config_custom():
    """Custom test configuration."""
    return InsightSpikeConfig(
        environment="test",
        llm={"provider": "mock", "temperature": 0.5, "max_tokens": 256},
        memory={"max_episodes": 100, "max_retrieved_docs": 5},
        embedding={"model_name": "test-model", "dimension": 384},
        graph={"spike_ged_threshold": -0.5, "spike_ig_threshold": 0.3},
    )


@pytest.fixture
def mock_datastore():
    """Create a mock datastore for unit tests."""
    datastore = Mock(spec=DataStore)
    datastore.load_episodes.return_value = []
    datastore.save_episodes.return_value = True
    datastore.load_graph.return_value = None
    datastore.save_graph.return_value = True
    datastore.get_metadata.return_value = {}
    datastore.set_metadata.return_value = True
    return datastore


@pytest.fixture
def filesystem_datastore(tmp_path):
    """Create a real filesystem datastore for integration tests."""
    config = {"type": "filesystem", "params": {"base_path": str(tmp_path)}}
    return DataStoreFactory.create(config)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return {
        "doc1": np.random.randn(384).astype(np.float32),
        "doc2": np.random.randn(384).astype(np.float32),
        "doc3": np.random.randn(384).astype(np.float32),
        "doc4": np.random.randn(384).astype(np.float32),
        "doc5": np.random.randn(384).astype(np.float32),
    }


@pytest.fixture
def sample_documents(sample_embeddings):
    """Generate sample documents with embeddings."""
    return [
        {
            "text": "InsightSpike is an AI system for discovering insights.",
            "embedding": sample_embeddings["doc1"],
            "metadata": {"source": "intro", "importance": 0.9},
        },
        {
            "text": "It uses graph-based reasoning to detect patterns.",
            "embedding": sample_embeddings["doc2"],
            "metadata": {"source": "technical", "importance": 0.8},
        },
        {
            "text": "Multiple layers process information hierarchically.",
            "embedding": sample_embeddings["doc3"],
            "metadata": {"source": "architecture", "importance": 0.7},
        },
        {
            "text": "Emergence occurs when systems integrate.",
            "embedding": sample_embeddings["doc4"],
            "metadata": {"source": "theory", "importance": 0.85},
        },
        {
            "text": "The system learns from patterns in data.",
            "embedding": sample_embeddings["doc5"],
            "metadata": {"source": "ml", "importance": 0.75},
        },
    ]


@pytest.fixture
def knowledge_base():
    """Create a knowledge base for testing."""
    return [
        "System A operates independently with its own logic.",
        "System B operates independently with different logic.",
        "When A and B integrate, new properties emerge.",
        "This emergence is more than the sum of parts.",
        "Complex systems exhibit non-linear behavior.",
        "Feedback loops amplify small changes.",
        "Pattern recognition reveals hidden connections.",
        "Graph structures represent relationships effectively.",
        "Information gain measures the value of new knowledge.",
        "Spike detection identifies significant insights.",
    ]


# Test utilities
class TestHelpers:
    """Utility functions for tests."""

    @staticmethod
    def create_mock_episode(text, c_value=0.7, dimension=384):
        """Create a mock episode with embedding."""
        from insightspike.core.episode import Episode

        vec = np.random.randn(dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        return Episode(text=text, vec=vec, c=c_value)

    @staticmethod
    def create_mock_graph_data(num_nodes=5, num_edges=8):
        """Create mock graph data for testing."""
        try:
            import torch
            from torch_geometric.data import Data

            # Create node features
            x = torch.randn(num_nodes, 384)

            # Create random edges
            edge_list = []
            for _ in range(num_edges):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst:
                    edge_list.append([src, dst])

            if not edge_list:
                edge_list = [[0, 1], [1, 0]]  # Minimum edges

            edge_index = torch.tensor(edge_list, dtype=torch.long).t()

            return Data(x=x, edge_index=edge_index)
        except ImportError:
            # Return a mock object if PyTorch Geometric not available
            mock_graph = Mock()
            mock_graph.num_nodes = num_nodes
            mock_graph.edge_index = Mock()
            mock_graph.edge_index.size.return_value = (2, num_edges)
            return mock_graph

    @staticmethod
    def assert_config_valid(config):
        """Assert that a configuration is valid."""
        assert isinstance(config, InsightSpikeConfig)
        assert config.environment in [
            "development",
            "experiment",
            "production",
            "research",
            "test",
            "custom",
        ]
        assert config.llm.provider in ["local", "openai", "anthropic", "mock", "clean"]
        assert 0.0 <= config.llm.temperature <= 2.0
        assert config.memory.max_episodes > 0
        assert config.embedding.dimension > 0


@pytest.fixture
def test_helpers():
    """Provide test helper utilities."""
    return TestHelpers()


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Disable any external API calls
    monkeypatch.setenv("INSIGHTSPIKE_TEST_MODE", "1")

    # Use mock LLM by default
    monkeypatch.setenv("INSIGHTSPIKE_LLM__PROVIDER", "mock")

    # Disable logging to console during tests
    monkeypatch.setenv("INSIGHTSPIKE_LOGGING__LOG_TO_CONSOLE", "false")

    # Set a test-specific log directory
    monkeypatch.setenv("INSIGHTSPIKE_LOGGING__FILE_PATH", "/tmp/insightspike_test_logs")


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield

    # Clear any cached models or data
    from insightspike.processing.embedder import _model_cache

    _model_cache.clear()


# Performance tracking
@pytest.fixture
def benchmark(request):
    """Simple benchmark fixture for performance testing."""
    import time

    start_time = time.time()

    def get_duration():
        return time.time() - start_time

    request.node.benchmark = get_duration

    yield get_duration

    duration = get_duration()
    if duration > 5.0:  # Warn if test takes more than 5 seconds
        print(f"\nWarning: Test {request.node.name} took {duration:.2f} seconds")
