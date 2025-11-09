"""
Test Phase 1 Re-modifications for PyG Standardization
====================================================

Tests the re-modifications needed after deciding to use PyG instead of NetworkX.
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.features.graph_reasoning import GraphAnalyzer
from insightspike.implementations.graph.pyg_graph_builder import PyGGraphBuilder
from insightspike.implementations.graph.graph_builder_adapter import GraphBuilderAdapter
from insightspike.core.episode import Episode


class TestPyGStandardization:
    """Test that all components now use PyG exclusively"""
    
    def test_graph_analyzer_pyg_only(self):
        """Test GraphAnalyzer only accepts PyG Data objects"""
        analyzer = GraphAnalyzer()
        
        # Create PyG graphs
        current = Data(
            x=torch.randn(5, 384),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            num_nodes=5
        )
        previous = Data(
            x=torch.randn(4, 384),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=4
        )
        
        # Simple metric functions for testing
        def mock_ged(g1, g2):
            return -0.5  # Negative for improvement
        
        def mock_ig(g1, g2):
            return 0.3  # Positive for information gain
        
        # This should work with PyG Data objects
        metrics = analyzer.calculate_metrics(current, previous, mock_ged, mock_ig)
        
        assert "delta_ged" in metrics
        assert "delta_ig" in metrics
        assert metrics["graph_size_current"] == 5
        assert metrics["graph_size_previous"] == 4
    
    def test_pyg_graph_builder(self):
        """Test PyGGraphBuilder creates PyG Data objects"""
        builder = PyGGraphBuilder()
        
        # Create test episodes
        episodes = [
            Episode(
                text=f"Episode {i}",
                vec=np.random.randn(384),
                c=0.5 + i * 0.1,
                timestamp=i,
                metadata={"index": i}
            )
            for i in range(3)
        ]
        
        # Build graph
        graph = builder.build_graph(episodes)
        
        # Check it's a PyG Data object
        assert isinstance(graph, Data)
        assert graph.num_nodes == 3
        assert graph.x.shape == (3, 384)
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'edge_attr')  # Multi-dimensional edge attributes
        assert hasattr(graph, 'episodes')
    
    def test_graph_builder_adapter_pyg_output(self):
        """Test GraphBuilderAdapter returns PyG Data objects"""
        # Test with ScalableGraphBuilder
        adapter_scalable = GraphBuilderAdapter(use_scalable=True)
        
        episodes = [
            Episode(
                text=f"Episode {i}",
                vec=np.random.randn(384),
                c=0.5,
                timestamp=i,
                metadata={}
            )
            for i in range(2)
        ]
        
        graph = adapter_scalable.build_graph(episodes)
        assert isinstance(graph, Data)
        
        # Test with PyGGraphBuilder
        adapter_pyg = GraphBuilderAdapter(use_scalable=False)
        graph2 = adapter_pyg.build_graph(episodes)
        assert isinstance(graph2, Data)
    
    def test_edge_attributes_support(self):
        """Test that PyG graphs support multi-dimensional edge attributes"""
        builder = PyGGraphBuilder()
        
        # Create episodes with high similarity
        base_vec = np.random.randn(384)
        episodes = [
            Episode(
                text=f"Similar episode {i}",
                vec=base_vec + np.random.randn(384) * 0.1,  # Small variations
                c=0.8,
                timestamp=i,
                metadata={}
            )
            for i in range(3)
        ]
        
        # Set low threshold to ensure edges are created
        builder.similarity_threshold = 0.5
        graph = builder.build_graph(episodes)
        
        # Check edge attributes exist
        assert hasattr(graph, 'edge_attr')
        if graph.edge_index.size(1) > 0:
            # Edge attributes should be present for each edge
            assert graph.edge_attr.shape[0] == graph.edge_index.size(1)
            # Currently 1D (just similarity), but ready for multi-dimensional
            assert graph.edge_attr.shape[1] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
