"""
GED Refactoring Test Cases
=========================

Tests for the improved GED/GEDIG implementation with desk-calculated expected values.
"""

import pytest
import numpy as np
import networkx as nx
from typing import Tuple, List

# Assuming these will be implemented
from insightspike.algorithms.graph_structure_analyzer import GraphStructureAnalyzer
from insightspike.metrics.improved_gedig_metrics import (
    ImprovedGEDIGCalculator,
    calculate_gedig_metrics
)


class TestGEDRefactoring:
    """Test cases with desk-calculated expected values."""
    
    def create_square_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        """
        Create a square graph (4 nodes, 4 edges).
        
        Structure:
        0 -- 1
        |    |
        3 -- 2
        
        Each node has degree 2.
        """
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        # Create embeddings in square pattern
        embeddings = np.array([
            [1.0, 1.0],    # Node 0: top-right
            [-1.0, 1.0],   # Node 1: top-left
            [-1.0, -1.0],  # Node 2: bottom-left
            [1.0, -1.0],   # Node 3: bottom-right
        ])
        
        # Normalize and pad to 384 dimensions
        embeddings = self._normalize_and_pad(embeddings)
        
        return G, embeddings
    
    def create_hub_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        """
        Create a hub graph (5 nodes, 8 edges).
        
        Structure:
        0 -- 1
        |\ /|
        | 4 |  (Node 4 is the hub)
        |/ \|
        3 -- 2
        
        Node 4 has degree 4, others have degree 3.
        """
        G = nx.Graph()
        # Square edges
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        # Hub connections
        G.add_edges_from([(4, 0), (4, 1), (4, 2), (4, 3)])
        
        # Create embeddings with hub at center
        embeddings = np.array([
            [1.0, 1.0],    # Node 0
            [-1.0, 1.0],   # Node 1
            [-1.0, -1.0],  # Node 2
            [1.0, -1.0],   # Node 3
            [0.0, 0.0],    # Node 4: hub at center
        ])
        
        embeddings = self._normalize_and_pad(embeddings)
        
        return G, embeddings
    
    def create_linear_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        """
        Create a linear chain graph (4 nodes, 3 edges).
        
        Structure: 0 -- 1 -- 2 -- 3
        """
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        # Linear embeddings
        embeddings = np.array([
            [-1.5, 0.0],   # Node 0
            [-0.5, 0.0],   # Node 1
            [0.5, 0.0],    # Node 2
            [1.5, 0.0],    # Node 3
        ])
        
        embeddings = self._normalize_and_pad(embeddings)
        
        return G, embeddings
    
    def create_star_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        """
        Create a star graph (4 nodes, 3 edges, node 1 is center).
        
        Structure:
            0
            |
        3 - 1 - 2
        """
        G = nx.Graph()
        G.add_edges_from([(1, 0), (1, 2), (1, 3)])
        
        # Star embeddings
        embeddings = np.array([
            [0.0, 1.0],    # Node 0: top
            [0.0, 0.0],    # Node 1: center
            [1.0, 0.0],    # Node 2: right
            [-1.0, 0.0],   # Node 3: left
        ])
        
        embeddings = self._normalize_and_pad(embeddings)
        
        return G, embeddings
    
    def _normalize_and_pad(self, embeddings: np.ndarray, dim: int = 384) -> np.ndarray:
        """Normalize embeddings and pad to specified dimensions."""
        n_nodes = embeddings.shape[0]
        padded = np.zeros((n_nodes, dim))
        
        # Copy original embeddings
        padded[:, :2] = embeddings
        
        # Add small noise to other dimensions
        padded[:, 2:10] = np.random.normal(0, 0.01, (n_nodes, 8))
        
        # Normalize each vector
        for i in range(n_nodes):
            norm = np.linalg.norm(padded[i])
            if norm > 0:
                padded[i] /= norm
                
        return padded
    
    def test_square_to_hub_transformation(self):
        """
        Test: Square → Hub transformation
        
        Expected values (desk calculation):
        - Nodes: 4 → 5 (+1)
        - Edges: 4 → 8 (+4)
        - GED = 5 (1 node insertion + 4 edge insertions)
        - Diameter: 2 → 2 (no change)
        - Average path length: 1.33 → 1.2 (improvement)
        - Global efficiency: ~0.67 → 0.8 (improvement)
        - Clustering: 0 → 0.5 (improvement)
        """
        # Create graphs
        square_graph, square_embeddings = self.create_square_graph()
        hub_graph, hub_embeddings = self.create_hub_graph()
        
        # Calculate structure metrics
        analyzer = GraphStructureAnalyzer()
        structure_result = analyzer.analyze_structural_change(square_graph, hub_graph)
        
        # Verify GED calculation
        assert structure_result["ged"] == pytest.approx(5.0, rel=0.1), \
            f"Expected GED=5, got {structure_result['ged']}"
        
        # Verify structural improvement (positive = better)
        assert structure_result["structural_improvement"] > 0.2, \
            "Hub formation should improve structure"
        
        # Verify hub formation detected
        assert structure_result["hub_formation"] > 0.5, \
            "Should detect strong hub formation"
        
        # Calculate full GEDIG metrics
        metrics = calculate_gedig_metrics(
            square_graph, hub_graph,
            square_embeddings, hub_embeddings
        )
        
        # Verify spike detection
        assert metrics.spike_detected == True, \
            "Square to hub should trigger spike"
        
        print(f"\nSquare→Hub Results:")
        print(f"  GED: {metrics.ged}")
        print(f"  Structural Improvement: {metrics.structural_improvement:.3f}")
        print(f"  IG: {metrics.ig:.3f}")
        print(f"  Insight Score: {metrics.insight_score:.3f}")
        print(f"  Spike: {metrics.spike_detected}")
    
    def test_linear_to_star_transformation(self):
        """
        Test: Linear chain → Star transformation
        
        Expected values:
        - Nodes: 4 → 4 (no change)
        - Edges: 3 → 3 (no change)
        - GED = 2 (reconnect 2 edges: remove 0-1, 2-3; add 1-0, 1-3)
        - Diameter: 3 → 2 (improvement)
        - Average path length: 2.0 → 1.5 (improvement)
        """
        linear_graph, linear_embeddings = self.create_linear_graph()
        star_graph, star_embeddings = self.create_star_graph()
        
        analyzer = GraphStructureAnalyzer()
        structure_result = analyzer.analyze_structural_change(linear_graph, star_graph)
        
        # GED should be 2 (edge reconnections)
        assert structure_result["ged"] == pytest.approx(2.0, rel=0.2), \
            f"Expected GED≈2, got {structure_result['ged']}"
        
        # Significant structural improvement
        assert structure_result["structural_improvement"] > 0.4, \
            "Linear to star should show major improvement"
        
        # Complexity reduction (diameter decreased)
        assert structure_result["complexity_reduction"] > 0.3, \
            "Should detect complexity reduction"
        
        print(f"\nLinear→Star Results:")
        print(f"  GED: {structure_result['ged']}")
        print(f"  Structural Improvement: {structure_result['structural_improvement']:.3f}")
        print(f"  Complexity Reduction: {structure_result['complexity_reduction']:.3f}")
    
    def test_disconnected_to_connected(self):
        """
        Test: Two disconnected pairs → Connected graph
        
        Initial: (0-1) (2-3) - two components
        Final: 0-1-2-3 - single component
        
        Expected:
        - GED = 1 (add edge 1-2)
        - Major efficiency improvement (infinite to finite path lengths)
        """
        # Create disconnected graph
        G1 = nx.Graph()
        G1.add_edges_from([(0, 1), (2, 3)])
        
        # Create connected graph
        G2 = nx.Graph()
        G2.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        analyzer = GraphStructureAnalyzer()
        result = analyzer.analyze_structural_change(G1, G2)
        
        assert result["ged"] == pytest.approx(1.0, rel=0.1)
        assert result["structural_improvement"] > 0.7, \
            "Connecting components should show massive improvement"
    
    def test_adding_redundant_edges(self):
        """
        Test: Adding redundant edges (no structural improvement)
        
        Square → Square with diagonal
        Should NOT trigger spike despite changes.
        """
        square_graph, square_embeddings = self.create_square_graph()
        
        # Add diagonal
        square_with_diagonal = square_graph.copy()
        square_with_diagonal.add_edge(0, 2)  # Add diagonal
        
        metrics = calculate_gedig_metrics(
            square_graph, square_with_diagonal,
            square_embeddings, square_embeddings
        )
        
        # GED = 1 (one edge added)
        assert metrics.ged == pytest.approx(1.0, rel=0.1)
        
        # Minimal improvement (already well-connected)
        assert metrics.structural_improvement < 0.2, \
            "Redundant edge shouldn't improve much"
        
        # Should NOT trigger spike
        assert metrics.spike_detected == False, \
            "Redundant changes shouldn't trigger spike"
    
    def test_legacy_compatibility(self):
        """Test backward compatibility with negative GED."""
        from insightspike.metrics.improved_gedig_metrics import compute_gedig_legacy
        
        # Test with old-style negative GED
        result = compute_gedig_legacy(
            delta_ged=-0.6,  # Old style: negative for improvement
            delta_ig=0.3
        )
        
        assert result["ged"] == 0.6  # Should be positive
        assert result["structural_improvement"] == 0.6
        assert result["spike_detected"] == True  # -0.6 < -0.5 and 0.3 > 0.2


class TestGEDCalculationDetails:
    """Detailed tests for GED calculation accuracy."""
    
    def test_exact_ged_small_graphs(self):
        """Test exact GED calculation for small graphs."""
        # Graph 1: Triangle
        G1 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        
        # Graph 2: Line
        G2 = nx.Graph([(0, 1), (1, 2)])
        
        # Expected GED = 1 (remove one edge)
        ged = nx.graph_edit_distance(G1, G2)
        assert ged == 1
    
    def test_node_addition_ged(self):
        """Test GED for node addition."""
        G1 = nx.Graph([(0, 1)])
        G2 = nx.Graph([(0, 1), (1, 2)])
        
        # Expected: 1 node + 1 edge = 2
        ged = nx.graph_edit_distance(G1, G2)
        assert ged == 2
    
    def test_efficiency_metrics(self):
        """Test graph efficiency calculations."""
        # Perfect star (most efficient)
        star = nx.star_graph(4)  # 5 nodes, center + 4 leaves
        star_eff = nx.global_efficiency(star)
        
        # Linear chain (least efficient)
        path = nx.path_graph(5)  # 5 nodes in line
        path_eff = nx.global_efficiency(path)
        
        # Star should be more efficient
        assert star_eff > path_eff
        
        # Verify expected values
        assert star_eff == pytest.approx(0.7, rel=0.1)
        assert path_eff == pytest.approx(0.4, rel=0.1)


if __name__ == "__main__":
    # Run specific test for debugging
    test = TestGEDRefactoring()
    test.test_square_to_hub_transformation()
    test.test_linear_to_star_transformation()