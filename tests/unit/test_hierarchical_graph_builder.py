"""
Tests for Hierarchical Graph Builder (Phase 3)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from insightspike.core.layers.hierarchical_graph_builder import HierarchicalGraphBuilder


class TestHierarchicalGraphBuilder:
    """Test the hierarchical graph builder implementation."""
    
    @pytest.fixture
    def builder(self):
        """Create a builder instance for testing."""
        return HierarchicalGraphBuilder(
            dimension=10,
            cluster_size=5,
            super_cluster_size=3,
            similarity_threshold=0.3,
            top_k=3
        )
    
    def test_initialization(self, builder):
        """Test builder initialization."""
        assert builder.dimension == 10
        assert builder.cluster_size == 5
        assert builder.super_cluster_size == 3
        assert len(builder.levels) == 3
        
    def test_build_hierarchical_graph(self, builder):
        """Test building a hierarchical graph."""
        # Create test documents
        documents = []
        for i in range(20):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            documents.append({
                'embedding': vec,
                'text': f'Document {i}',
                'metadata': {'id': i}
            })
        
        # Build hierarchy
        result = builder.build_hierarchical_graph(documents)
        
        # Verify structure
        assert 'nodes_per_level' in result
        assert len(result['nodes_per_level']) == 3
        assert result['nodes_per_level'][0] == 20  # All documents at level 0
        assert result['nodes_per_level'][1] > 0    # Some clusters at level 1
        assert result['nodes_per_level'][2] > 0    # At least one super-cluster
        assert result['compression_ratio'] > 1
        
    def test_hierarchical_search(self, builder):
        """Test hierarchical search functionality."""
        # Build a small hierarchy
        documents = []
        for i in range(10):
            vec = np.zeros(10, dtype=np.float32)
            vec[i % 10] = 1.0  # Different dimensions for different docs
            documents.append({
                'embedding': vec,
                'text': f'Document {i}',
                'metadata': {'id': i}
            })
        
        builder.build_hierarchical_graph(documents)
        
        # Search with a query similar to first document
        query = np.zeros(10, dtype=np.float32)
        query[0] = 1.0
        
        results = builder.search_hierarchical(query, k=3)
        
        # Should find the first document with highest similarity
        assert len(results) > 0
        assert results[0]['text'] == 'Document 0'
        assert results[0]['similarity'] > 0.9
        
    def test_compression_ratio(self, builder):
        """Test that compression ratio increases with data size."""
        # Small dataset
        small_docs = [{'embedding': np.random.randn(10).astype(np.float32), 
                      'text': f'Doc {i}'} for i in range(10)]
        result_small = builder.build_hierarchical_graph(small_docs)
        
        # Larger dataset
        builder_large = HierarchicalGraphBuilder(
            dimension=10,
            cluster_size=10,
            super_cluster_size=10
        )
        large_docs = [{'embedding': np.random.randn(10).astype(np.float32), 
                      'text': f'Doc {i}'} for i in range(100)]
        result_large = builder_large.build_hierarchical_graph(large_docs)
        
        # Larger dataset should have better compression
        assert result_large['compression_ratio'] > result_small['compression_ratio']
        
    def test_add_document(self, builder):
        """Test dynamic document addition."""
        # Initial documents
        documents = []
        for i in range(10):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            documents.append({
                'embedding': vec,
                'text': f'Document {i}'
            })
        
        builder.build_hierarchical_graph(documents)
        initial_count = builder.stats['total_nodes'][0]
        
        # Add new document
        new_doc = {
            'embedding': np.random.randn(10).astype(np.float32),
            'text': 'New document'
        }
        result = builder.add_document(new_doc)
        
        # Verify addition
        assert result['success']
        assert builder.stats['total_nodes'][0] == initial_count + 1
        assert result['level_0_idx'] == initial_count
        
    def test_statistics(self, builder):
        """Test statistics reporting."""
        documents = [{'embedding': np.random.randn(10).astype(np.float32), 
                     'text': f'Doc {i}'} for i in range(50)]
        builder.build_hierarchical_graph(documents)
        
        stats = builder.get_statistics()
        
        assert 'nodes_per_level' in stats
        assert 'compression_ratios' in stats
        assert len(stats['compression_ratios']) == 2  # Between 3 levels
        
        # If clusters exist, check size distribution
        if builder.levels[1].nodes:
            assert 'cluster_size_distribution' in stats
            assert stats['cluster_size_distribution']['mean'] > 0