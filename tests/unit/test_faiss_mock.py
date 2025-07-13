"""Tests for FAISS mock implementation"""
import numpy as np
import pytest

from insightspike.utils.faiss_mock import (
    IndexFlatIP,
    IndexFlatL2,
    MockFaissIndex,
    get_faiss_index,
)


class TestMockFaissIndex:
    """Test mock FAISS index implementation."""

    def test_init(self):
        """Test index initialization."""
        index = MockFaissIndex(dimension=10)
        assert index.dimension == 10
        assert index.ntotal == 0
        assert index.is_trained is True
        assert index.vectors is None

    def test_add_single_vector(self):
        """Test adding single vector."""
        index = MockFaissIndex(dimension=5)
        vec = np.random.randn(5).astype(np.float32)

        index.add(vec)
        assert index.ntotal == 1
        assert index.vectors.shape == (1, 5)

    def test_add_multiple_vectors(self):
        """Test adding multiple vectors."""
        index = MockFaissIndex(dimension=5)
        vecs = np.random.randn(10, 5).astype(np.float32)

        index.add(vecs)
        assert index.ntotal == 10
        assert index.vectors.shape == (10, 5)

    def test_search_empty_index(self):
        """Test search on empty index."""
        index = MockFaissIndex(dimension=5)
        query = np.random.randn(5).astype(np.float32)

        distances, indices = index.search(query, k=5)
        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert np.all(indices[0] == -1)

    def test_search_single_query(self):
        """Test search with single query."""
        index = MockFaissIndex(dimension=5)

        # Add some vectors
        vecs = np.random.randn(10, 5).astype(np.float32)
        index.add(vecs)

        # Search
        query = vecs[0]  # Use first vector as query
        distances, indices = index.search(query, k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)
        assert indices[0, 0] == 0  # First result should be itself

    def test_search_multiple_queries(self):
        """Test search with multiple queries."""
        index = MockFaissIndex(dimension=5)

        # Add vectors
        vecs = np.random.randn(10, 5).astype(np.float32)
        index.add(vecs)

        # Search with multiple queries
        queries = np.random.randn(3, 5).astype(np.float32)
        distances, indices = index.search(queries, k=2)

        assert distances.shape == (3, 2)
        assert indices.shape == (3, 2)

    def test_reset(self):
        """Test index reset."""
        index = MockFaissIndex(dimension=5)
        index.add(np.random.randn(10, 5).astype(np.float32))

        assert index.ntotal == 10

        index.reset()
        assert index.ntotal == 0
        assert index.vectors is None


def test_index_flat_ip():
    """Test IndexFlatIP factory."""
    index = IndexFlatIP(dimension=10)
    assert isinstance(index, MockFaissIndex)
    assert index.dimension == 10
    assert index.index_type == "FlatIP"


def test_index_flat_l2():
    """Test IndexFlatL2 factory."""
    index = IndexFlatL2(dimension=10)
    assert isinstance(index, MockFaissIndex)
    assert index.dimension == 10
    assert index.index_type == "FlatL2"


def test_get_faiss_index_mock():
    """Test get_faiss_index with mock fallback."""
    # This will use mock since faiss import will fail in the function
    index = get_faiss_index(dimension=10, index_type="FlatIP")
    assert isinstance(index, MockFaissIndex) or hasattr(index, "add")
