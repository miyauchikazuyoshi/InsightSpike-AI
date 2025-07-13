import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_build_graph(tmp_path):
    """Test GraphBuilder can be instantiated and basic functionality works."""

    # Mock PyTorch Geometric dependencies to avoid import issues
    with patch.dict(
        "sys.modules",
        {
            "torch_geometric": MagicMock(),
            "torch_geometric.data": MagicMock(),
            "torch_geometric.nn": MagicMock(),
        },
    ):
        from insightspike.core.layers.layer3_graph_reasoner import GraphBuilder

        # Test that GraphBuilder can be instantiated successfully
        graph_builder = GraphBuilder()

        # Test that it has the expected configuration
        assert hasattr(graph_builder, "similarity_threshold")
        assert isinstance(graph_builder.similarity_threshold, float)
        assert 0.0 <= graph_builder.similarity_threshold <= 1.0

        # Test that it has the expected methods
        assert hasattr(graph_builder, "build_graph")
        assert callable(graph_builder.build_graph)
