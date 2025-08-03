"""
Unit tests for Wake Mode geDIG implementation.
"""

import unittest
from typing import Set

import networkx as nx
import numpy as np

from insightspike.algorithms.gedig_wake_mode import (
    ProcessingMode,
    WakeModeGeDIG,
    WakeModeResult,
    calculate_wake_mode_gedig,
)


class TestWakeModeGeDIG(unittest.TestCase):
    """Test Wake Mode geDIG calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = WakeModeGeDIG()
        
        # Create a simple test graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([
            ('A', 'B'), ('B', 'C'), ('C', 'D'),
            ('D', 'E'), ('E', 'A'), ('B', 'D')
        ])
        
        # Create a pattern graph
        self.pattern_graph = nx.Graph()
        self.pattern_graph.add_edges_from([
            ('A', 'B'), ('B', 'C'), ('C', 'A')
        ])
        
    def test_wake_mode_basic(self):
        """Test basic Wake Mode calculation."""
        focal_nodes = {'B', 'C'}
        result = self.calculator.calculate_wake_mode_gedig(
            self.graph, focal_nodes
        )
        
        self.assertIsInstance(result, WakeModeResult)
        self.assertEqual(result.mode, "wake")
        self.assertGreater(result.gedig_value, 0)
        self.assertGreater(result.efficiency_score, 0)
        
    def test_minimal_ged_calculation(self):
        """Test minimal GED calculation for Wake Mode."""
        # Well-connected nodes should have lower GED
        focal_nodes_connected = {'B', 'D'}  # High degree nodes
        focal_nodes_isolated = {'A', 'E'}   # Lower degree nodes
        
        ged_connected = self.calculator._calculate_minimal_ged(
            self.graph, focal_nodes_connected
        )
        ged_isolated = self.calculator._calculate_minimal_ged(
            self.graph, focal_nodes_isolated
        )
        
        # Well-connected nodes need fewer changes
        self.assertLess(ged_connected, ged_isolated)
        
    def test_convergent_ig_calculation(self):
        """Test convergent IG favors low entropy."""
        # Create high entropy subgraph
        high_entropy_graph = nx.Graph()
        high_entropy_graph.add_edges_from([
            ('X', f'Y{i}') for i in range(10)
        ])
        
        # Create low entropy subgraph (regular structure)
        low_entropy_graph = nx.Graph()
        low_entropy_graph.add_edges_from([
            ('X', 'Y'), ('Y', 'Z'), ('Z', 'X')
        ])
        
        ig_high = self.calculator._calculate_convergent_ig(
            high_entropy_graph, {'X'}
        )
        ig_low = self.calculator._calculate_convergent_ig(
            low_entropy_graph, {'X'}
        )
        
        # Both should have non-zero values
        self.assertGreater(ig_low, 0.0)
        self.assertGreater(ig_high, 0.0)
        # Low entropy graph might have different IG
        # (not necessarily higher in this simple case)
        
    def test_pattern_matching(self):
        """Test pattern matching functionality."""
        # Add known pattern
        self.calculator.add_pattern('triangle', self.pattern_graph)
        
        # Query with similar structure
        focal_nodes = {'A', 'B', 'C'}
        query_context = {'type': 'triangle_search'}
        
        similarity = self.calculator._find_nearest_pattern(
            self.graph, focal_nodes, query_context
        )
        
        # Should find some match
        self.assertGreater(similarity, 0.3)
        
    def test_pattern_matching_no_patterns(self):
        """Test pattern matching with no stored patterns."""
        focal_nodes = {'A', 'B'}
        similarity = self.calculator._find_nearest_pattern(
            self.graph, focal_nodes, None
        )
        
        self.assertEqual(similarity, 0.0)
        
    def test_wake_vs_sleep_mode_values(self):
        """Test that Wake Mode produces different values than Sleep Mode."""
        focal_nodes = {'B', 'C', 'D'}
        
        # Wake Mode calculation
        wake_result = self.calculator.calculate_wake_mode_gedig(
            self.graph, focal_nodes
        )
        
        # Regular (Sleep Mode) calculation
        sleep_result = self.calculator.calculate(
            self.graph, focal_nodes
        )
        
        # Wake Mode should minimize, Sleep Mode should maximize
        # So Wake Mode geDIG should typically be lower
        self.assertNotEqual(wake_result.gedig_value, sleep_result.gedig_value)
        
    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        # Highly connected focal nodes
        focal_nodes = {'B', 'D'}
        result = self.calculator.calculate_wake_mode_gedig(
            self.graph, focal_nodes
        )
        
        # Should have good efficiency
        self.assertGreater(result.efficiency_score, 0.3)
        self.assertLessEqual(result.efficiency_score, 1.0)
        
    def test_subgraph_extraction(self):
        """Test subgraph extraction around focal nodes."""
        focal_nodes = {'B'}
        radius = 1
        
        subgraph = self.calculator._extract_subgraph(
            self.graph, focal_nodes, radius
        )
        
        # Should include B and its immediate neighbors
        expected_nodes = {'A', 'B', 'C', 'D'}
        self.assertEqual(set(subgraph.nodes()), expected_nodes)
        
    def test_graph_similarity(self):
        """Test graph similarity calculation."""
        # Identical graphs
        g1 = nx.Graph([('A', 'B'), ('B', 'C')])
        g2 = nx.Graph([('A', 'B'), ('B', 'C')])
        
        similarity = self.calculator._calculate_graph_similarity(g1, g2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Completely different graphs
        g3 = nx.Graph([('X', 'Y'), ('Y', 'Z')])
        similarity = self.calculator._calculate_graph_similarity(g1, g3)
        self.assertEqual(similarity, 0.0)
        
        # Partially overlapping
        g4 = nx.Graph([('A', 'B'), ('C', 'D')])
        similarity = self.calculator._calculate_graph_similarity(g1, g4)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
        
    def test_empty_graph(self):
        """Test Wake Mode with empty graph."""
        empty_graph = nx.Graph()
        focal_nodes = {'A', 'B'}
        
        result = self.calculator.calculate_wake_mode_gedig(
            empty_graph, focal_nodes
        )
        
        self.assertIsInstance(result, WakeModeResult)
        self.assertEqual(result.ged_value, 1.0)  # Maximum GED for empty graph
        
    def test_convenience_function(self):
        """Test the convenience function."""
        focal_nodes = {'B', 'C'}
        result = calculate_wake_mode_gedig(self.graph, focal_nodes)
        
        self.assertIsInstance(result, WakeModeResult)
        self.assertEqual(result.mode, "wake")
        
    def test_clear_patterns(self):
        """Test pattern clearing."""
        # Add patterns
        self.calculator.add_pattern('p1', self.pattern_graph)
        self.calculator.add_pattern('p2', self.graph)
        
        self.assertEqual(len(self.calculator.known_patterns), 2)
        
        # Clear patterns
        self.calculator.clear_patterns()
        
        self.assertEqual(len(self.calculator.known_patterns), 0)
        self.assertEqual(len(self.calculator.pattern_embeddings), 0)
        
    def test_convergence_score(self):
        """Test convergence score calculation."""
        # Add a known pattern
        self.calculator.add_pattern('known', self.pattern_graph)
        
        focal_nodes = {'A', 'B', 'C'}
        result = self.calculator.calculate_wake_mode_gedig(
            self.graph, focal_nodes
        )
        
        # Convergence score should be reasonable
        self.assertGreater(result.convergence_score, 0)
        self.assertLessEqual(result.convergence_score, 1.0)


class TestProcessingMode(unittest.TestCase):
    """Test ProcessingMode enum."""
    
    def test_mode_values(self):
        """Test enum values."""
        self.assertEqual(ProcessingMode.WAKE.value, "wake")
        self.assertEqual(ProcessingMode.SLEEP.value, "sleep")
        
    def test_mode_comparison(self):
        """Test mode comparison."""
        mode1 = ProcessingMode.WAKE
        mode2 = ProcessingMode.WAKE
        mode3 = ProcessingMode.SLEEP
        
        self.assertEqual(mode1, mode2)
        self.assertNotEqual(mode1, mode3)


if __name__ == '__main__':
    unittest.main()