"""Mock factory for creating consistent test objects across the test suite."""

import numpy as np
import networkx as nx
from typing import Optional, List, Tuple, Dict, Any
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass


@dataclass
class MockEpisode:
    """Mock episode for memory testing."""

    text: str
    timestamp: float
    c_value: float = 0.5
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.random.randn(384)  # Default embedding size
        if self.metadata is None:
            self.metadata = {}


class GraphFactory:
    """Factory for creating various types of graphs for testing."""

    @staticmethod
    def create_simple_graph(
        num_nodes: int = 5, num_edges: Optional[int] = None
    ) -> nx.Graph:
        """Create a simple graph with specified nodes and edges."""
        g = nx.Graph()
        g.add_nodes_from(range(1, num_nodes + 1))

        if num_edges is None:
            # Create a path graph by default
            for i in range(1, num_nodes):
                g.add_edge(i, i + 1)
        else:
            # Add random edges
            import random

            possible_edges = [
                (i, j)
                for i in range(1, num_nodes + 1)
                for j in range(i + 1, num_nodes + 1)
            ]
            random.shuffle(possible_edges)
            g.add_edges_from(possible_edges[:num_edges])

        return g

    @staticmethod
    def create_graph_with_embeddings(
        num_nodes: int = 5, embedding_dim: int = 384
    ) -> nx.Graph:
        """Create a graph with node embeddings."""
        g = nx.Graph()

        for i in range(1, num_nodes + 1):
            embedding = np.random.randn(embedding_dim)
            g.add_node(i, embedding=embedding, text=f"Node {i}")

        # Add some edges
        for i in range(1, num_nodes):
            g.add_edge(i, i + 1)

        return g

    @staticmethod
    def create_pytorch_graph_mock(num_nodes: int = 5, feature_dim: int = 64):
        """Create a mock PyTorch geometric graph."""
        mock_graph = Mock()
        mock_graph.num_nodes = num_nodes
        mock_graph.x = np.random.randn(num_nodes, feature_dim)
        mock_graph.edge_index = Mock()
        mock_graph.edge_index.size = Mock(return_value=(2, num_nodes - 1))
        return mock_graph


class EmbeddingFactory:
    """Factory for creating embedding-related test objects."""

    @staticmethod
    def create_embedding_batch(num_samples: int = 10, dim: int = 384) -> np.ndarray:
        """Create a batch of embeddings."""
        return np.random.randn(num_samples, dim).astype(np.float32)

    @staticmethod
    def create_similar_embeddings(
        base_embedding: np.ndarray, num_variations: int = 5, noise_level: float = 0.1
    ) -> List[np.ndarray]:
        """Create embeddings similar to a base embedding."""
        embeddings = []
        for _ in range(num_variations):
            noise = np.random.randn(*base_embedding.shape) * noise_level
            embeddings.append(base_embedding + noise)
        return embeddings

    @staticmethod
    def create_mock_embedding_model():
        """Create a mock embedding model."""
        mock_model = Mock()
        mock_model.encode = Mock(
            side_effect=lambda texts: np.random.randn(
                len(texts) if hasattr(texts, "__len__") else 1, 384
            )
        )
        return mock_model


class MemoryFactory:
    """Factory for creating memory-related test objects."""

    @staticmethod
    def create_episodes(
        num_episodes: int = 5, embedding_dim: int = 384
    ) -> List[MockEpisode]:
        """Create a list of mock episodes."""
        episodes = []
        for i in range(num_episodes):
            episode = MockEpisode(
                text=f"Episode {i}: This is test content for episode {i}.",
                timestamp=float(i),
                c_value=np.random.rand(),
                embedding=np.random.randn(embedding_dim),
            )
            episodes.append(episode)
        return episodes

    @staticmethod
    def create_mock_memory_manager():
        """Create a mock memory manager."""
        mock_memory = Mock()
        mock_memory.episodes = []
        mock_memory.store_episode = Mock(
            side_effect=lambda ep: mock_memory.episodes.append(ep)
        )
        mock_memory.search_episodes = Mock(return_value=[])
        mock_memory.get_memory_stats = Mock(
            return_value={
                "episode_count": len(mock_memory.episodes),
                "total_capacity": 1000,
            }
        )
        return mock_memory


class ConfigFactory:
    """Factory for creating configuration objects for testing."""

    @staticmethod
    def create_test_config(
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a test configuration with optional overrides."""
        base_config = {
            "embedding": {
                "model_name": "test-model",
                "dimension": 384,
                "device": "cpu",
            },
            "llm": {
                "provider": "mock",
                "model_name": "mock-llm",
                "max_tokens": 256,
                "temperature": 0.3,
            },
            "retrieval": {"similarity_threshold": 0.25, "top_k": 5},
            "memory": {"max_episodes": 1000, "merge_threshold": 0.8},
            "reasoning": {"use_gnn": False, "spike_threshold": 0.5},
        }

        if overrides:
            # Deep merge overrides
            import copy

            config = copy.deepcopy(base_config)
            for key, value in overrides.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            return config

        return base_config


class LLMFactory:
    """Factory for creating LLM-related mocks."""

    @staticmethod
    def create_mock_llm_provider():
        """Create a mock LLM provider."""
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="This is a mock LLM response.")
        mock_llm.generate_batch = Mock(
            side_effect=lambda prompts: [
                f"Mock response for: {p[:30]}..." for p in prompts
            ]
        )
        return mock_llm

    @staticmethod
    def create_mock_llm_response(insight_detected: bool = False) -> str:
        """Create a mock LLM response with optional insight detection."""
        if insight_detected:
            return """Based on the analysis, I've discovered a key insight:
            The pattern shows a clear relationship between A and B.
            This is a eureka moment that simplifies our understanding."""
        else:
            return """The analysis shows standard patterns.
            No significant insights were detected in this iteration."""


class DocumentFactory:
    """Factory for creating document test data."""

    @staticmethod
    def create_test_documents(num_docs: int = 5) -> List[Dict[str, Any]]:
        """Create test documents."""
        docs = []
        topics = ["science", "technology", "history", "mathematics", "philosophy"]

        for i in range(num_docs):
            doc = {
                "id": f"doc_{i}",
                "text": f"This is document {i} about {topics[i % len(topics)]}. " * 10,
                "metadata": {
                    "topic": topics[i % len(topics)],
                    "timestamp": float(i),
                    "source": f"source_{i}",
                },
            }
            docs.append(doc)

        return docs


# Singleton instances for common factories
graph_factory = GraphFactory()
embedding_factory = EmbeddingFactory()
memory_factory = MemoryFactory()
config_factory = ConfigFactory()
llm_factory = LLMFactory()
document_factory = DocumentFactory()
