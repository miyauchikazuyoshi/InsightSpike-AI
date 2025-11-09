"""
Test PyG Algorithm Integration
==============================

Tests the integration of all PyG-based graph algorithms.
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.algorithms.graph_edit_distance import GraphEditDistance, compute_delta_ged
from insightspike.algorithms.graph_importance import GraphImportanceCalculator
from insightspike.implementations.memory.graph_memory_search import GraphMemorySearch
from insightspike.core.episode import Episode


class TestPyGAlgorithmsIntegration:
    """Test that all graph algorithms work together with PyG"""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample PyG graph for testing"""
        # Create node features (5 nodes with 384-dim features)
        x = torch.randn(5, 384)
        
        # Create edges (simple chain with some cross-connections)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 2],  # source nodes
            [1, 0, 2, 1, 3, 2, 4, 3, 2, 0]   # target nodes
        ], dtype=torch.long)
        
        # Create edge attributes (for future multi-dimensional features)
        edge_attr = torch.rand(10, 1)  # Currently just similarity scores
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=5)
    
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes matching the graph"""
        episodes = []
        for i in range(5):
            episodes.append(Episode(
                text=f"Episode {i}: Knowledge about topic {i}",
                vec=np.random.randn(384),
                c=0.5 + i * 0.1,
                timestamp=i,
                metadata={"topic": f"topic_{i}"}
            ))
        return episodes
    
    def test_graph_edit_distance_pyg(self, sample_graph):
        """Test GED calculation with PyG graphs"""
        ged_calc = GraphEditDistance()
        
        # Create a modified version of the graph (remove a node)
        modified_x = torch.randn(4, 384)
        modified_edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]
        ], dtype=torch.long)
        modified_graph = Data(x=modified_x, edge_index=modified_edge_index, num_nodes=4)
        
        # Calculate GED
        result = ged_calc.calculate(sample_graph, modified_graph)
        
        assert result.ged_value > 0  # Should detect difference
        assert result.graph1_size == 5
        assert result.graph2_size == 4
        assert not result.timeout_occurred
        
        # Test ΔGED calculation
        delta_ged = ged_calc.compute_delta_ged(sample_graph, modified_graph)
        assert delta_ged < 0  # Negative because graph got simpler
    
    def test_graph_importance_pyg(self, sample_graph):
        """Test graph importance calculation with PyG"""
        importance_calc = GraphImportanceCalculator()
        
        # Calculate importance for node 2 (most connected in our graph)
        scores = importance_calc.calculate_importance(sample_graph, node_idx=2)
        
        assert "degree" in scores
        assert "pagerank" in scores
        assert "combined" in scores
        assert scores["degree"] > 0  # Node 2 has connections
        
        # Get top-k important nodes
        top_nodes = importance_calc.get_top_k_important(sample_graph, k=3)
        assert len(top_nodes) == 3
        assert all(0 <= node_idx < 5 for node_idx, _ in top_nodes)
    
    def test_graph_memory_search_pyg(self, sample_graph, sample_episodes):
        """Test graph-based memory search with PyG"""
        search = GraphMemorySearch()
        
        # Create a query embedding
        query_embedding = np.random.randn(384)
        
        # Search without graph (baseline)
        results_no_graph = search.search_with_graph(
            query_embedding, sample_episodes, graph_data=None, k=3
        )
        assert len(results_no_graph) <= 3
        
        # Search with graph (multi-hop)
        results_with_graph = search.search_with_graph(
            query_embedding, sample_episodes, graph_data=sample_graph, k=3
        )
        assert len(results_with_graph) <= 3
        
        # Check that results have graph metadata
        if results_with_graph:
            result = results_with_graph[0]
            assert "hop" in result
            assert "path" in result
            assert isinstance(result["path"], list)
    
    def test_algorithm_pipeline_integration(self, sample_graph, sample_episodes):
        """Test full pipeline: GED → Importance → Memory Search"""
        # 1. Calculate GED between consecutive graph states
        ged_calc = GraphEditDistance()
        
        # Simulate graph evolution (add a node)
        new_x = torch.cat([sample_graph.x, torch.randn(1, 384)])
        new_edge_index = torch.cat([
            sample_graph.edge_index,
            torch.tensor([[4, 5], [5, 4]], dtype=torch.long)
        ], dim=1)
        evolved_graph = Data(x=new_x, edge_index=new_edge_index, num_nodes=6)
        
        delta_ged = ged_calc.compute_delta_ged(sample_graph, evolved_graph)
        
        # 2. Update importance scores based on graph changes
        importance_calc = GraphImportanceCalculator()
        
        if delta_ged < -0.5:  # Significant structural change
            importance_calc.invalidate_cache()
        
        importance_map = importance_calc.update_graph_importance(evolved_graph)
        assert len(importance_map) == 6  # All nodes have importance
        
        # 3. Use importance for memory search
        search = GraphMemorySearch()
        query_embedding = np.random.randn(384)
        
        # Add new episode for the new node
        new_episode = Episode(
            text="New insight discovered",
            vec=np.random.randn(384),
            c=0.9,  # High confidence due to insight
            timestamp=5,
            metadata={"type": "insight"}
        )
        all_episodes = sample_episodes + [new_episode]
        
        # Search with evolved graph
        results = search.search_with_graph(
            query_embedding, all_episodes, graph_data=evolved_graph, k=5
        )
        
        # Verify results are properly ranked
        assert len(results) <= 5
        if len(results) > 1:
            # Check that results are sorted by score
            scores = [r.get("graph_score", r["similarity"]) for r in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_edge_attribute_support(self):
        """Test that algorithms handle edge attributes properly"""
        # Create graph with multi-dimensional edge attributes
        x = torch.randn(3, 384)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        
        # Multi-dimensional edge attributes (semantic, structural, temporal)
        edge_attr = torch.stack([
            torch.tensor([0.8, 0.7, 0.9, 0.6]),  # semantic similarity
            torch.tensor([0.5, 0.5, 0.3, 0.4]),  # structural similarity
            torch.tensor([0.9, 0.9, 0.7, 0.8])   # temporal similarity
        ], dim=1)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
        
        # Test GED handles edge attributes
        ged_calc = GraphEditDistance()
        result = ged_calc.calculate(graph, graph)
        assert result.ged_value < 0.001  # Same graph (allowing for floating point precision)
        
        # Test importance calculator works with edge attributes
        importance_calc = GraphImportanceCalculator()
        scores = importance_calc.calculate_importance(graph, node_idx=1)
        assert scores["degree"] > 0
    
    def test_empty_graph_handling(self):
        """Test algorithms handle empty graphs gracefully"""
        empty_graph = Data(
            x=torch.empty(0, 384),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            num_nodes=0
        )
        
        # Test GED
        ged_calc = GraphEditDistance()
        single_node_graph = Data(
            x=torch.randn(1, 384),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            num_nodes=1
        )
        result = ged_calc.calculate(empty_graph, single_node_graph)
        assert result.ged_value == 1.0  # Cost of adding one node
        
        # Test importance (should return empty scores)
        importance_calc = GraphImportanceCalculator()
        scores = importance_calc.calculate_importance(empty_graph, node_idx=0)
        assert scores["combined"] == 0.0
        
        # Test memory search (should return empty results)
        search = GraphMemorySearch()
        results = search.search_with_graph(
            np.random.randn(384), [], graph_data=empty_graph, k=5
        )
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
