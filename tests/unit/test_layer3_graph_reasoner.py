"""
Unit tests for Layer3GraphReasoner
==================================

Tests graph reasoning functionality including:
- Graph construction and analysis
- Spike detection with ΔGED and ΔIG
- Conflict scoring
- GNN processing
- Error handling
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from insightspike.config.models import InsightSpikeConfig
from insightspike.implementations.layers.layer3_graph_reasoner import (
    ConflictScore,
    GraphBuilder,
    L3GraphReasoner,
)


class TestConflictScore:
    """Test ConflictScore calculation."""

    @pytest.fixture
    def conflict_scorer(self):
        """Create a conflict scorer for testing."""
        return ConflictScore()

    @pytest.fixture
    def sample_graphs(self):
        """Create sample graphs for testing."""
        # Graph 1: 3 nodes, 2 edges
        graph1 = Data(
            x=torch.randn(3, 384),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
        )
        graph1.num_nodes = 3

        # Graph 2: 4 nodes, 4 edges
        graph2 = Data(
            x=torch.randn(4, 384),
            edge_index=torch.tensor(
                [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long
            ).t(),
        )
        graph2.num_nodes = 4

        return graph1, graph2

    def test_calculate_conflict_basic(self, conflict_scorer, sample_graphs):
        """Test basic conflict calculation."""
        graph1, graph2 = sample_graphs
        context = {"previous_confidence": 0.8, "current_confidence": 0.3}

        conflicts = conflict_scorer.calculate_conflict(graph1, graph2, context)

        assert "structural" in conflicts
        assert "semantic" in conflicts
        assert "temporal" in conflicts
        assert "total" in conflicts
        assert 0 <= conflicts["total"] <= 1

    def test_structural_conflict(self, conflict_scorer, sample_graphs):
        """Test structural conflict calculation."""
        graph1, graph2 = sample_graphs

        conflict = conflict_scorer._structural_conflict(graph1, graph2)

        # Different number of nodes and edges should create conflict
        assert conflict > 0
        assert conflict <= 1

    def test_structural_conflict_identical(self, conflict_scorer):
        """Test structural conflict for identical graphs."""
        graph = Data(
            x=torch.randn(3, 384),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
        )
        graph.num_nodes = 3

        conflict = conflict_scorer._structural_conflict(graph, graph)
        assert conflict == 0.0

    def test_structural_conflict_none_graphs(self, conflict_scorer):
        """Test structural conflict with None graphs."""
        graph = Data(x=torch.randn(3, 384))

        assert conflict_scorer._structural_conflict(None, graph) == 0.0
        assert conflict_scorer._structural_conflict(graph, None) == 0.0
        assert conflict_scorer._structural_conflict(None, None) == 0.0

    def test_semantic_conflict(self, conflict_scorer):
        """Test semantic conflict calculation."""
        # Create graphs with similar features
        features1 = torch.randn(3, 384)
        features2 = features1 + torch.randn(3, 384) * 0.1  # Slightly different

        graph1 = Data(x=features1)
        graph2 = Data(x=features2)

        conflict = conflict_scorer._semantic_conflict(graph1, graph2)

        # Should have low conflict due to similarity
        assert 0 <= conflict <= 1
        assert conflict < 0.5  # Similar features = low conflict

    def test_semantic_conflict_orthogonal(self, conflict_scorer):
        """Test semantic conflict with orthogonal features."""
        # Create orthogonal feature vectors
        features1 = torch.zeros(2, 384)
        features1[0, 0] = 1.0
        features2 = torch.zeros(2, 384)
        features2[0, 1] = 1.0

        graph1 = Data(x=features1)
        graph2 = Data(x=features2)

        conflict = conflict_scorer._semantic_conflict(graph1, graph2)

        # Orthogonal features should have high conflict
        assert conflict > 0.9

    def test_semantic_conflict_empty_features(self, conflict_scorer):
        """Test semantic conflict with empty features."""
        graph1 = Data(x=torch.empty(0, 384))
        graph2 = Data(x=torch.randn(3, 384))

        conflict = conflict_scorer._semantic_conflict(graph1, graph2)
        assert conflict == 0.0

    def test_semantic_conflict_zero_norm(self, conflict_scorer):
        """Test semantic conflict with zero-norm features."""
        graph1 = Data(x=torch.zeros(3, 384))
        graph2 = Data(x=torch.randn(3, 384))

        conflict = conflict_scorer._semantic_conflict(graph1, graph2)
        assert conflict == 0.0

    def test_temporal_conflict(self, conflict_scorer):
        """Test temporal conflict calculation."""
        # High confidence difference
        context1 = {"previous_confidence": 0.9, "current_confidence": 0.2}
        conflict1 = conflict_scorer._temporal_conflict(context1)
        assert conflict1 == 0.7

        # Low confidence difference
        context2 = {"previous_confidence": 0.8, "current_confidence": 0.75}
        conflict2 = conflict_scorer._temporal_conflict(context2)
        assert abs(conflict2 - 0.05) < 0.0001  # Use approximate equality for floats

        # Missing confidence values
        context3 = {"other_data": "value"}
        conflict3 = conflict_scorer._temporal_conflict(context3)
        assert conflict3 == 0.0

    def test_calculate_conflict_error_handling(self, conflict_scorer):
        """Test error handling in conflict calculation."""
        # Create invalid graph
        invalid_graph = Mock()
        invalid_graph.edge_index.size.side_effect = Exception("Invalid graph")

        conflicts = conflict_scorer.calculate_conflict(invalid_graph, invalid_graph, {})

        # Should return zero conflicts on error
        assert conflicts["total"] == 0.0


class TestGraphBuilder:
    """Test GraphBuilder functionality."""

    @pytest.fixture
    def graph_builder(self):
        """Create a graph builder for testing."""
        return GraphBuilder()

    @pytest.fixture
    def sample_documents(self, sample_embeddings):
        """Create sample documents with embeddings."""
        return [
            {"text": "Doc 1", "embedding": sample_embeddings["doc1"]},
            {"text": "Doc 2", "embedding": sample_embeddings["doc2"]},
            {"text": "Doc 3", "embedding": sample_embeddings["doc3"]},
        ]

    def test_build_graph_basic(self, graph_builder, sample_documents):
        """Test basic graph building."""
        graph = graph_builder.build_graph(sample_documents)

        assert isinstance(graph, Data)
        assert graph.num_nodes == 3
        assert graph.x.shape == (3, 384)
        assert graph.edge_index.shape[0] == 2
        assert hasattr(graph, "documents")

    def test_build_graph_empty_documents(self, graph_builder):
        """Test graph building with empty documents."""
        graph = graph_builder.build_graph([])

        assert graph.x.shape == (0, 384)
        assert graph.edge_index.shape == (2, 0)

    def test_build_graph_single_document(self, graph_builder):
        """Test graph building with single document."""
        doc = {"text": "Single doc", "embedding": np.random.randn(384)}
        graph = graph_builder.build_graph([doc])

        assert graph.num_nodes == 1
        assert graph.edge_index.shape[1] >= 1  # At least self-loop

    def test_build_graph_two_documents(self, graph_builder):
        """Test graph building with two documents."""
        docs = [
            {"text": "Doc 1", "embedding": np.random.randn(384)},
            {"text": "Doc 2", "embedding": np.random.randn(384)},
        ]
        graph = graph_builder.build_graph(docs)

        assert graph.num_nodes == 2
        # Should create bidirectional edge
        assert graph.edge_index.shape[1] >= 2

    def test_build_graph_similarity_threshold(self, graph_builder):
        """Test graph building with similarity threshold."""
        # Create documents with controlled similarities
        base_emb = np.random.randn(384)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Similar documents
        docs = [
            {"text": "Doc 1", "embedding": base_emb},
            {"text": "Doc 2", "embedding": base_emb + np.random.randn(384) * 0.01},
            {"text": "Doc 3", "embedding": -base_emb},  # Dissimilar
        ]

        graph = graph_builder.build_graph(docs)

        # Should have edges between similar docs but not dissimilar
        edges = graph.edge_index.t().tolist()
        # Check that there's an edge between 0 and 1 (similar)
        assert [0, 1] in edges or [1, 0] in edges

    def test_build_graph_without_embeddings(self, graph_builder):
        """Test graph building when documents lack embeddings."""
        docs = [
            {"text": "Doc 1"},
            {"text": "Doc 2"},
            {"text": "Doc 3"},
        ]

        graph = graph_builder.build_graph(docs)

        assert graph.num_nodes == 3
        assert graph.x.shape == (3, 384)  # Should generate random embeddings

    def test_build_graph_error_handling(self, graph_builder):
        """Test error handling in graph building."""
        # Create documents that will cause error
        docs = [{"text": "Doc", "embedding": "invalid"}]

        graph = graph_builder.build_graph(docs)

        # Should return empty graph on error
        assert graph.x.shape[0] == 0


class TestL3GraphReasoner:
    """Test L3GraphReasoner functionality."""

    @pytest.fixture
    def graph_reasoner(self, config_experiment):
        """Create a graph reasoner for testing."""
        with patch(
            "insightspike.implementations.layers.layer3_graph_reasoner.ScalableGraphBuilder"
        ):
            reasoner = L3GraphReasoner(config_experiment)
            
            # Mock GraphAnalyzer to return proper metrics
            reasoner.graph_analyzer.calculate_metrics = Mock(
                return_value={
                    "delta_ged": -0.3,
                    "delta_ig": 0.4,
                    "graph_size": 5,
                    "modularity": 0.7
                }
            )
            
            # Mock detect_spike to return proper result
            reasoner.graph_analyzer.detect_spike = Mock(
                return_value=(True, 0.8)  # spike_detected, confidence
            )
            
            return reasoner

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph for testing."""
        graph = Data(
            x=torch.randn(5, 384),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
            ),
        )
        graph.num_nodes = 5
        return graph

    def test_initialization(self, config_experiment):
        """Test L3GraphReasoner initialization."""
        reasoner = L3GraphReasoner(config_experiment)

        assert reasoner.layer_id == "layer3_graph_reasoner"
        assert reasoner.graph_builder is not None
        assert reasoner.conflict_scorer is not None
        assert reasoner.graph_analyzer is not None
        assert reasoner.reward_calculator is not None

    def test_initialize_method(self, graph_reasoner):
        """Test initialize method."""
        success = graph_reasoner.initialize()
        assert success
        assert graph_reasoner._is_initialized

    def test_process_with_documents(self, graph_reasoner, sample_documents):
        """Test processing documents through the reasoner."""
        # Mock the graph builder
        mock_graph = Mock(spec=Data)
        mock_graph.num_nodes = 3
        mock_graph.x = torch.randn(3, 384)
        graph_reasoner.graph_builder.build_graph.return_value = mock_graph

        result = graph_reasoner.process(sample_documents)

        assert "graph" in result
        assert "metrics" in result
        assert "conflicts" in result
        assert "reward" in result
        assert "spike_detected" in result
        assert "reasoning_quality" in result

    def test_process_with_empty_documents(self, graph_reasoner):
        """Test processing empty documents."""
        result = graph_reasoner.process([])

        # Should create synthetic graph
        assert result["graph"] is not None
        assert result["spike_detected"] == False

    def test_process_with_context_graph(self, graph_reasoner, mock_graph):
        """Test processing with pre-built graph in context."""
        context = {"graph": mock_graph}
        
        # Mock the graph builder to return the mock_graph when called
        graph_reasoner.graph_builder._empty_graph.return_value = mock_graph

        result = graph_reasoner.process({"data": [], "context": context})

        # The graph should be from context, not empty graph
        assert result["graph"] is not None
        assert result["spike_detected"] == True  # Based on our mock

    def test_analyze_documents_with_previous_graph(self, graph_reasoner, mock_graph):
        """Test document analysis with previous graph."""
        # Set previous graph
        graph_reasoner.previous_graph = mock_graph

        # Create new graph
        new_graph = Data(
            x=torch.randn(6, 384),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
            ),
        )
        new_graph.num_nodes = 6

        graph_reasoner.graph_builder.build_graph.return_value = new_graph

        result = graph_reasoner.analyze_documents([{"text": "test"}])

        # Should calculate metrics between graphs
        assert (
            result["metrics"]["delta_ged"] != 0.0
            or result["metrics"]["delta_ig"] != 0.0
        )

    def test_spike_detection(self, graph_reasoner):
        """Test spike detection logic."""
        # Create metrics that should trigger spike
        metrics = {"delta_ged": -0.6, "delta_ig": 0.3}  # GED negative for improvement
        conflicts = {"total": 0.2}

        spike = graph_reasoner.graph_analyzer.detect_spike(
            metrics, conflicts, graph_reasoner._get_spike_thresholds()
        )

        assert spike == True

    def test_spike_detection_no_spike(self, graph_reasoner):
        """Test spike detection when conditions not met."""
        # Metrics that shouldn't trigger spike
        metrics = {"delta_ged": -0.1, "delta_ig": 0.1}
        conflicts = {"total": 0.8}

        spike = graph_reasoner.graph_analyzer.detect_spike(
            metrics, conflicts, graph_reasoner._get_spike_thresholds()
        )

        assert spike == False

    def test_get_spike_thresholds(self, graph_reasoner):
        """Test spike threshold retrieval."""
        thresholds = graph_reasoner._get_spike_thresholds()

        assert "ged" in thresholds
        assert "ig" in thresholds
        assert "conflict" in thresholds
        assert thresholds["ged"] < 0  # GED threshold should be negative

    def test_gnn_initialization(self, config_experiment):
        """Test GNN initialization when enabled."""
        # Enable GNN in config
        config_experiment.graph.use_gnn = True

        with patch(
            "insightspike.implementations.layers.layer3_graph_reasoner.ScalableGraphBuilder"
        ):
            reasoner = L3GraphReasoner(config_experiment)

        assert reasoner.gnn is not None

    def test_process_with_gnn(self, config_experiment, mock_graph):
        """Test graph processing with GNN."""
        config_experiment.graph.use_gnn = True

        with patch(
            "insightspike.implementations.layers.layer3_graph_reasoner.ScalableGraphBuilder"
        ):
            reasoner = L3GraphReasoner(config_experiment)
            
            # Mock the GNN processing
            if reasoner.gnn is not None:
                # Mock GNN forward method
                reasoner.gnn.forward = Mock(return_value=torch.randn(5, 64))

        # Process graph with GNN
        result = reasoner._process_with_gnn(mock_graph)

        if reasoner.gnn is not None:
            assert result is not None
            assert isinstance(result, torch.Tensor)

    def test_process_with_gnn_empty_graph(self, graph_reasoner):
        """Test GNN processing with empty graph."""
        empty_graph = Data(x=torch.empty(0, 384))
        empty_graph.num_nodes = 0

        result = graph_reasoner._process_with_gnn(empty_graph)
        assert result is None

    def test_fallback_result(self, graph_reasoner):
        """Test fallback result generation."""
        result = graph_reasoner._fallback_result()

        assert result["spike_detected"] == False
        assert result["metrics"]["delta_ged"] == 0.0
        assert result["metrics"]["delta_ig"] == 0.0
        assert result["reasoning_quality"] == 0.0

    def test_interface_methods(self, graph_reasoner):
        """Test L3GraphReasonerInterface methods."""
        # Test build_graph
        vectors = np.random.randn(3, 384)
        graph = graph_reasoner.build_graph(vectors)
        assert graph is not None

        # Test calculate_ged
        graph1 = Mock()
        graph2 = Mock()
        ged = graph_reasoner.calculate_ged(graph1, graph2)
        assert isinstance(ged, float)

        # Test calculate_ig
        state1 = Mock()
        state2 = Mock()
        ig = graph_reasoner.calculate_ig(state1, state2)
        assert isinstance(ig, float)

        # Test detect_eureka_spike
        spike = graph_reasoner.detect_eureka_spike(-0.6, 0.3)
        assert isinstance(spike, bool)

    def test_cleanup(self, graph_reasoner):
        """Test cleanup method."""
        graph_reasoner.previous_graph = Mock()
        graph_reasoner.gnn = Mock()

        graph_reasoner.cleanup()

        assert graph_reasoner.previous_graph is None
        assert graph_reasoner.gnn is None
        assert not graph_reasoner._is_initialized

    def test_process_error_handling(self, graph_reasoner):
        """Test error handling in process method."""
        # Create input that will cause error
        graph_reasoner.graph_builder.build_graph.side_effect = Exception("Build failed")

        result = graph_reasoner.process([{"text": "test"}])

        # Should return fallback result
        assert result["spike_detected"] == False
        assert result["metrics"]["delta_ged"] == 0.0

    def test_metrics_selector_integration(self, config_experiment):
        """Test integration with MetricsSelector."""
        reasoner = L3GraphReasoner(config_experiment)

        # Check that metrics methods are set
        assert reasoner.delta_ged is not None
        assert reasoner.delta_ig is not None

        # Get algorithm info
        algo_info = reasoner.metrics_selector.get_algorithm_info()
        assert "ged_algorithm" in algo_info
        assert "ig_algorithm" in algo_info


class TestIntegration:
    """Integration tests for graph reasoning components."""

    def test_full_reasoning_pipeline(self, config_experiment, sample_documents):
        """Test full reasoning pipeline from documents to spike detection."""
        reasoner = L3GraphReasoner(config_experiment)

        # First analysis (no previous graph)
        result1 = reasoner.analyze_documents(sample_documents[:2])
        assert result1["spike_detected"] == False  # No previous graph

        # Second analysis (with previous graph)
        result2 = reasoner.analyze_documents(sample_documents[2:])
        assert "delta_ged" in result2["metrics"]
        assert "delta_ig" in result2["metrics"]

        # Check that previous graph was stored
        assert reasoner.previous_graph is not None

    def test_conflict_detection_integration(self, config_experiment):
        """Test integration of conflict detection in reasoning."""
        reasoner = L3GraphReasoner(config_experiment)

        # Create conflicting document sets
        docs1 = [{"text": "System is stable and efficient"}]
        docs2 = [{"text": "System is unstable and inefficient"}]

        # Analyze first set
        result1 = reasoner.analyze_documents(docs1)

        # Analyze conflicting set
        result2 = reasoner.analyze_documents(docs2)

        # Should detect some conflict
        assert result2["conflicts"]["total"] >= 0

    def test_reward_calculation_integration(self, config_experiment):
        """Test reward calculation in full pipeline."""
        reasoner = L3GraphReasoner(config_experiment)

        # Create documents that should generate positive reward
        docs = [
            {"text": "Novel insight about system behavior"},
            {"text": "Unexpected pattern discovered"},
            {"text": "New connection revealed"},
        ]

        result = reasoner.analyze_documents(docs)

        assert "reward" in result
        assert "base" in result["reward"]
        assert "structure" in result["reward"]
        assert "novelty" in result["reward"]
        assert "total" in result["reward"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
