"""
Tests for query-centric sphere search implementation.
"""

import pytest
import numpy as np
from insightspike.query.sphere_search import (
    SphereSearch, SimpleSphereSearch, NeighborNode
)


class TestSphereSearch:
    """Test sphere search functionality."""
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample node vectors for testing."""
        # Create vectors in 2D for easy visualization
        vectors = {
            'node_0': np.array([0.0, 0.0]),      # At origin
            'node_1': np.array([0.3, 0.0]),      # Close to origin
            'node_2': np.array([0.0, 0.5]),      # Medium distance
            'node_3': np.array([0.7, 0.7]),      # Far (distance ~0.99)
            'node_4': np.array([1.5, 0.0]),      # Very far
            'node_5': np.array([-0.2, 0.1]),     # Close, different direction
        }
        return vectors
    
    @pytest.fixture
    def searchers(self, sample_vectors):
        """Create both FAISS and simple searchers."""
        return {
            'faiss': SphereSearch(sample_vectors),
            'simple': SimpleSphereSearch(sample_vectors)
        }
    
    def test_sphere_search_basic(self, searchers):
        """Test basic sphere search functionality."""
        query = np.array([0.0, 0.0])
        radius = 0.6
        
        for name, searcher in searchers.items():
            neighbors = searcher.search_sphere(query, radius)
            
            # Should find nodes 0, 1, 2, 5 (within radius 0.6)
            assert len(neighbors) == 4, f"{name}: Wrong number of neighbors"
            
            # Check they are sorted by distance
            distances = [n.distance for n in neighbors]
            assert distances == sorted(distances), f"{name}: Not sorted"
            
            # Check specific nodes
            node_ids = [n.node_id for n in neighbors]
            assert 'node_0' in node_ids
            assert 'node_1' in node_ids
            assert 'node_2' in node_ids
            assert 'node_5' in node_ids
            assert 'node_3' not in node_ids  # Too far
            assert 'node_4' not in node_ids  # Too far
    
    def test_sphere_search_with_offset_query(self, searchers):
        """Test sphere search with query not at origin."""
        query = np.array([0.5, 0.5])
        radius = 0.5
        
        for name, searcher in searchers.items():
            neighbors = searcher.search_sphere(query, radius)
            
            # Check distances are measured from query point
            for neighbor in neighbors:
                actual_distance = np.linalg.norm(
                    neighbor.vector - query
                )
                assert abs(neighbor.distance - actual_distance) < 1e-6
                assert neighbor.distance < radius
    
    def test_donut_search(self, searchers):
        """Test donut search functionality."""
        query = np.array([0.0, 0.0])
        inner_radius = 0.3
        outer_radius = 0.8
        
        for name, searcher in searchers.items():
            neighbors = searcher.search_donut(
                query, inner_radius, outer_radius
            )
            
            # Should exclude node_0 (too close) and nodes 3,4 (too far)
            # Should include nodes 2, 5 (and maybe 1 if > 0.3)
            for neighbor in neighbors:
                assert neighbor.distance > inner_radius
                assert neighbor.distance < outer_radius
            
            node_ids = [n.node_id for n in neighbors]
            assert 'node_0' not in node_ids  # Too close
            assert 'node_2' in node_ids      # In donut
            
    def test_max_neighbors_limit(self, searchers):
        """Test limiting number of returned neighbors."""
        query = np.array([0.0, 0.0])
        radius = 2.0  # Large radius to include all
        max_neighbors = 3
        
        for name, searcher in searchers.items():
            neighbors = searcher.search_sphere(
                query, radius, max_neighbors=max_neighbors
            )
            
            assert len(neighbors) <= max_neighbors
            # Should return the closest ones
            assert neighbors[0].node_id == 'node_0'
    
    def test_empty_search(self, searchers):
        """Test search with no results."""
        query = np.array([10.0, 10.0])  # Far from all nodes
        radius = 0.1
        
        for name, searcher in searchers.items():
            neighbors = searcher.search_sphere(query, radius)
            assert len(neighbors) == 0
    
    def test_relative_positions(self, searchers):
        """Test that relative positions are correct."""
        query = np.array([0.1, 0.1])
        radius = 1.0
        
        for name, searcher in searchers.items():
            neighbors = searcher.search_sphere(query, radius)
            
            for neighbor in neighbors:
                # Check relative position calculation
                expected_relative = neighbor.vector - query
                np.testing.assert_array_almost_equal(
                    neighbor.relative_position,
                    expected_relative
                )
    
    def test_statistics(self, sample_vectors):
        """Test statistics calculation."""
        searcher = SimpleSphereSearch(sample_vectors)
        query = np.array([0.0, 0.0])
        neighbors = searcher.search_sphere(query, radius=0.6)
        
        stats = SphereSearch(sample_vectors).get_statistics(neighbors)
        
        assert stats['count'] == len(neighbors)
        assert stats['min_distance'] >= 0
        assert stats['max_distance'] < 0.6
        assert stats['mean_distance'] > 0


class TestHighDimensional:
    """Test with high-dimensional vectors."""
    
    def test_high_dim_search(self):
        """Test search in high dimensions."""
        dim = 768  # Typical embedding dimension
        n_nodes = 100
        
        # Create random vectors
        vectors = {
            f'node_{i}': np.random.randn(dim)
            for i in range(n_nodes)
        }
        
        # Add one close vector
        query = np.random.randn(dim)
        vectors['close_node'] = query + 0.1 * np.random.randn(dim)
        
        searcher = SimpleSphereSearch(vectors)
        neighbors = searcher.search_sphere(query, radius=0.5)
        
        # Should find at least the close node
        assert len(neighbors) > 0
        assert 'close_node' in [n.node_id for n in neighbors]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])