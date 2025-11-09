"""
Unit tests for GraphAnalyzer
============================

Tests graph analysis functionality including:
- Metrics calculation (ΔGED and ΔIG)
- Spike detection logic
- Quality assessment
- Graph format conversion
- Error handling
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.features.graph_reasoning.graph_analyzer import GraphAnalyzer


class TestGraphAnalyzerInitialization:
    """Test GraphAnalyzer initialization."""

    def test_init_default(self):
        """Test initialization with default config."""
        analyzer = GraphAnalyzer()
        assert analyzer.config == {}

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = {"graph": {"some_setting": "value"}}
        analyzer = GraphAnalyzer(config)
        assert analyzer.config == config


class TestMetricsCalculation:
    """Test metrics calculation functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a GraphAnalyzer instance."""
        return GraphAnalyzer()

    @pytest.fixture
    def sample_graphs(self):
        """Create sample PyTorch Geometric graphs."""
        # Graph 1: 3 nodes, simple chain
        graph1 = Data(
            x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        )
        graph1.num_nodes = 3

        # Graph 2: 4 nodes, cycle
        graph2 = Data(
            x=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float
            ),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]], dtype=torch.long
            ),
        )
        graph2.num_nodes = 4

        return graph1, graph2

    @pytest.fixture
    def mock_metrics_functions(self):
        """Create mock metric calculation functions."""
        mock_ged = Mock(return_value=-0.5)  # Negative for improvement
        mock_ig = Mock(return_value=0.3)
        return mock_ged, mock_ig

    def test_calculate_metrics_with_previous_graph(
        self, analyzer, sample_graphs, mock_metrics_functions
    ):
        """Test metrics calculation with previous graph."""
        graph1, graph2 = sample_graphs
        mock_ged, mock_ig = mock_metrics_functions

        metrics = analyzer.calculate_metrics(graph2, graph1, mock_ged, mock_ig)

        assert metrics["delta_ged"] == -0.5
        assert metrics["delta_ig"] == 0.3
        assert metrics["graph_size_current"] == 4
        assert metrics["graph_size_previous"] == 3

        # Check that metric functions were called
        assert mock_ged.called
        assert mock_ig.called

    def test_calculate_metrics_without_previous_graph(
        self, analyzer, sample_graphs, mock_metrics_functions
    ):
        """Test metrics calculation without previous graph."""
        _, graph2 = sample_graphs
        mock_ged, mock_ig = mock_metrics_functions

        metrics = analyzer.calculate_metrics(graph2, None, mock_ged, mock_ig)

        assert metrics["delta_ged"] == 0.0
        assert metrics["delta_ig"] == 0.0
        assert metrics["graph_size_current"] == 4
        assert metrics["graph_size_previous"] == 0

        # Metric functions should not be called
        assert not mock_ged.called
        assert not mock_ig.called

    def test_calculate_metrics_networkx_conversion(self, analyzer, sample_graphs):
        """Test NetworkX graph conversion during metrics calculation."""
        graph1, graph2 = sample_graphs

        # Mock delta functions that check NetworkX format
        def mock_ged(g1, g2):
            # Should receive NetworkX graphs
            assert hasattr(g1, "nodes")
            assert hasattr(g2, "nodes")
            return -0.4

        def mock_ig(vecs1, vecs2):
            # Should receive numpy arrays
            assert isinstance(vecs1, np.ndarray) or vecs1 is None
            assert isinstance(vecs2, np.ndarray) or vecs2 is None
            return 0.2

        metrics = analyzer.calculate_metrics(graph2, graph1, mock_ged, mock_ig)

        assert metrics["delta_ged"] == -0.4
        assert metrics["delta_ig"] == 0.2

    def test_calculate_metrics_with_no_features(self, analyzer):
        """Test metrics calculation when graphs have no features."""
        # Graphs without x attribute
        graph1 = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
        graph1.num_nodes = 2
        graph1.x = None

        graph2 = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
        graph2.num_nodes = 2
        graph2.x = None

        mock_ged = Mock(return_value=-0.3)
        mock_ig = Mock(return_value=0.0)

        metrics = analyzer.calculate_metrics(graph2, graph1, mock_ged, mock_ig)

        # IG should be 0 when no features
        assert metrics["delta_ig"] == 0.0

    def test_calculate_metrics_error_handling(self, analyzer, sample_graphs):
        """Test error handling in metrics calculation."""
        graph1, graph2 = sample_graphs

        # Mock functions that raise exceptions
        mock_ged = Mock(side_effect=Exception("GED calculation failed"))
        mock_ig = Mock(side_effect=Exception("IG calculation failed"))

        metrics = analyzer.calculate_metrics(graph2, graph1, mock_ged, mock_ig)

        # Should return zero metrics on error
        assert metrics["delta_ged"] == 0.0
        assert metrics["delta_ig"] == 0.0
        assert metrics["graph_size_current"] == 4
        assert metrics["graph_size_previous"] == 3

    def test_calculate_metrics_empty_current_graph(self, analyzer):
        """Test metrics with empty current graph."""
        empty_graph = Data(x=torch.empty(0, 384))
        empty_graph.num_nodes = 0

        mock_ged = Mock(return_value=0.0)
        mock_ig = Mock(return_value=0.0)

        metrics = analyzer.calculate_metrics(empty_graph, None, mock_ged, mock_ig)

        assert metrics["graph_size_current"] == 0


class TestSpikeDetection:
    """Test spike detection functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a GraphAnalyzer instance."""
        return GraphAnalyzer()

    @pytest.fixture
    def thresholds(self):
        """Standard spike detection thresholds."""
        return {
            "ged": -0.5,  # GED threshold (negative for improvement)
            "ig": 0.2,  # IG threshold
            "conflict": 0.5,  # Conflict threshold
        }

    def test_detect_spike_all_conditions_met(self, analyzer, thresholds):
        """Test spike detection when all conditions are met."""
        metrics = {
            "delta_ged": -0.6,  # Below threshold (good)
            "delta_ig": 0.3,  # Above threshold (good)
        }
        conflicts = {"total": 0.2}  # Below threshold (good)

        spike = analyzer.detect_spike(metrics, conflicts, thresholds)
        assert spike == True

    def test_detect_spike_ged_not_met(self, analyzer, thresholds):
        """Test spike detection when GED condition not met."""
        metrics = {
            "delta_ged": -0.3,  # Above threshold (not enough improvement)
            "delta_ig": 0.3,
        }
        conflicts = {"total": 0.2}

        spike = analyzer.detect_spike(metrics, conflicts, thresholds)
        assert spike == False

    def test_detect_spike_ig_not_met(self, analyzer, thresholds):
        """Test spike detection when IG condition not met."""
        metrics = {"delta_ged": -0.6, "delta_ig": 0.1}  # Below threshold
        conflicts = {"total": 0.2}

        spike = analyzer.detect_spike(metrics, conflicts, thresholds)
        assert spike == False

    def test_detect_spike_conflict_too_high(self, analyzer, thresholds):
        """Test spike detection when conflict is too high."""
        metrics = {"delta_ged": -0.6, "delta_ig": 0.3}
        conflicts = {"total": 0.7}  # Above threshold

        spike = analyzer.detect_spike(metrics, conflicts, thresholds)
        assert spike == False

    def test_detect_spike_edge_cases(self, analyzer, thresholds):
        """Test spike detection at exact thresholds."""
        # Exactly at thresholds
        metrics = {
            "delta_ged": -0.5,  # Exactly at threshold
            "delta_ig": 0.2,  # Exactly at threshold
        }
        conflicts = {"total": 0.5}  # Exactly at threshold

        spike = analyzer.detect_spike(metrics, conflicts, thresholds)
        assert spike == False  # Conflict must be below threshold

    def test_detect_spike_missing_metrics(self, analyzer, thresholds):
        """Test spike detection with missing metrics."""
        # Missing delta_ged
        metrics1 = {"delta_ig": 0.3}
        conflicts = {"total": 0.2}
        spike1 = analyzer.detect_spike(metrics1, conflicts, thresholds)
        assert spike1 == False

        # Missing delta_ig
        metrics2 = {"delta_ged": -0.6}
        spike2 = analyzer.detect_spike(metrics2, conflicts, thresholds)
        assert spike2 == False

        # Missing conflicts total
        metrics3 = {"delta_ged": -0.6, "delta_ig": 0.3}
        conflicts3 = {}
        spike3 = analyzer.detect_spike(metrics3, conflicts3, thresholds)
        assert spike3 == False

    def test_detect_spike_positive_ged(self, analyzer, thresholds):
        """Test spike detection with positive GED (degradation)."""
        metrics = {"delta_ged": 0.6, "delta_ig": 0.3}  # Positive means worse
        conflicts = {"total": 0.2}

        spike = analyzer.detect_spike(metrics, conflicts, thresholds)
        assert spike == False


class TestQualityAssessment:
    """Test quality assessment functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a GraphAnalyzer instance."""
        return GraphAnalyzer()

    def test_assess_quality_high_metrics_low_conflict(self, analyzer):
        """Test quality assessment with good metrics and low conflict."""
        metrics = {
            "delta_ged": -0.8,  # High improvement
            "delta_ig": 0.6,  # High information gain
        }
        conflicts = {"total": 0.1}  # Low conflict

        quality = analyzer.assess_quality(metrics, conflicts)

        # Expected: (0.8 + 0.6) / 2 - 0.1 = 0.7 - 0.1 = 0.6
        assert abs(quality - 0.6) < 0.001
        assert 0.0 <= quality <= 1.0

    def test_assess_quality_moderate_metrics(self, analyzer):
        """Test quality assessment with moderate metrics."""
        metrics = {"delta_ged": -0.4, "delta_ig": 0.3}
        conflicts = {"total": 0.2}

        quality = analyzer.assess_quality(metrics, conflicts)

        # Expected: (0.4 + 0.3) / 2 - 0.2 = 0.35 - 0.2 = 0.15
        assert abs(quality - 0.15) < 0.001

    def test_assess_quality_poor_metrics(self, analyzer):
        """Test quality assessment with poor metrics."""
        metrics = {
            "delta_ged": 0.2,  # Degradation
            "delta_ig": -0.1,  # Information loss
        }
        conflicts = {"total": 0.3}

        quality = analyzer.assess_quality(metrics, conflicts)

        # GED score = abs(0.2) = 0.2, IG score = -0.1 (negative)
        # Expected: (0.2 + (-0.1)) / 2 - 0.3 = 0.05 - 0.3 = -0.25
        # Clamped to 0
        assert quality == 0.0

    def test_assess_quality_clamping(self, analyzer):
        """Test quality clamping to [0, 1] range."""
        # Test upper bound
        metrics1 = {"delta_ged": -2.0, "delta_ig": 1.5}  # Very high improvement
        conflicts1 = {"total": 0.0}

        quality1 = analyzer.assess_quality(metrics1, conflicts1)
        assert quality1 == 1.0  # Clamped to 1

        # Test lower bound
        metrics2 = {"delta_ged": 0.5, "delta_ig": 0.0}
        conflicts2 = {"total": 1.0}  # Very high conflict

        quality2 = analyzer.assess_quality(metrics2, conflicts2)
        assert quality2 == 0.0  # Clamped to 0

    def test_assess_quality_missing_values(self, analyzer):
        """Test quality assessment with missing values."""
        # Missing delta_ged
        metrics1 = {"delta_ig": 0.5}
        conflicts = {"total": 0.1}
        quality1 = analyzer.assess_quality(metrics1, conflicts)
        # Expected: (0 + 0.5) / 2 - 0.1 = 0.15
        assert abs(quality1 - 0.15) < 0.001

        # Missing conflicts
        metrics2 = {"delta_ged": -0.6, "delta_ig": 0.4}
        conflicts2 = {}
        quality2 = analyzer.assess_quality(metrics2, conflicts2)
        # Expected: (0.6 + 0.4) / 2 - 0 = 0.5
        assert abs(quality2 - 0.5) < 0.001

    def test_assess_quality_zero_metrics(self, analyzer):
        """Test quality assessment with all zero metrics."""
        metrics = {"delta_ged": 0.0, "delta_ig": 0.0}
        conflicts = {"total": 0.0}

        quality = analyzer.assess_quality(metrics, conflicts)
        assert quality == 0.0


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.fixture
    def analyzer(self):
        """Create a GraphAnalyzer instance."""
        return GraphAnalyzer()

    def test_full_analysis_pipeline(self, analyzer):
        """Test full analysis pipeline from graphs to quality assessment."""
        # Create graphs
        graph1 = Data(
            x=torch.randn(5, 128),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
            ),
        )
        graph1.num_nodes = 5

        graph2 = Data(
            x=torch.randn(6, 128),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
            ),
        )
        graph2.num_nodes = 6

        # Mock metric functions
        mock_ged = Mock(return_value=-0.4)
        mock_ig = Mock(return_value=0.3)

        # Calculate metrics
        metrics = analyzer.calculate_metrics(graph2, graph1, mock_ged, mock_ig)

        # Create conflicts
        conflicts = {"total": 0.2}

        # Detect spike
        thresholds = {"ged": -0.5, "ig": 0.2, "conflict": 0.5}
        spike = analyzer.detect_spike(metrics, conflicts, thresholds)

        # Assess quality
        quality = analyzer.assess_quality(metrics, conflicts)

        # Verify results
        assert metrics["graph_size_current"] == 6
        assert metrics["graph_size_previous"] == 5
        # Spike detection depends on whether GED threshold is met
        # With mock_ged returning -0.4 and threshold -0.5, it won't spike
        assert spike == False  # GED threshold not met
        assert 0.0 <= quality <= 1.0

    def test_graph_format_edge_cases(self, analyzer):
        """Test handling of various graph format edge cases."""
        # Graph with no edges
        graph1 = Data(x=torch.randn(3, 64))
        graph1.num_nodes = 3
        graph1.edge_index = torch.empty(2, 0, dtype=torch.long)

        # Graph with self-loops only
        graph2 = Data(
            x=torch.randn(3, 64),
            edge_index=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        )
        graph2.num_nodes = 3

        mock_ged = Mock(return_value=-0.2)
        mock_ig = Mock(return_value=0.1)

        metrics = analyzer.calculate_metrics(graph2, graph1, mock_ged, mock_ig)

        assert metrics["delta_ged"] == -0.2
        assert metrics["delta_ig"] == 0.1

    def test_numerical_stability(self, analyzer):
        """Test numerical stability with extreme values."""
        # Test with very small quality differences
        metrics = {"delta_ged": -1e-10, "delta_ig": 1e-10}
        conflicts = {"total": 1e-10}

        quality = analyzer.assess_quality(metrics, conflicts)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

        # Test spike detection with extreme thresholds
        extreme_thresholds = {"ged": -1e10, "ig": 1e10, "conflict": 1e-10}

        spike = analyzer.detect_spike(metrics, conflicts, extreme_thresholds)
        assert isinstance(spike, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
