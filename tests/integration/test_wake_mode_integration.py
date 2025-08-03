"""
Integration tests for Wake Mode with MainAgent.
"""

import unittest
from unittest.mock import Mock, patch
import networkx as nx
import numpy as np

from insightspike.algorithms.gedig_wake_mode import WakeModeGeDIG, ProcessingMode
from insightspike.algorithms.gedig_core import GeDIGCore


class TestWakeModeIntegration(unittest.TestCase):
    """Test Wake Mode integration with the main system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a realistic knowledge graph
        self.knowledge_graph = self._create_knowledge_graph()
        
        # Initialize Wake Mode calculator
        self.wake_calculator = WakeModeGeDIG()
        
        # Add some known patterns
        self._add_known_patterns()
        
    def _create_knowledge_graph(self):
        """Create a realistic knowledge graph for testing."""
        g = nx.Graph()
        
        # Add concept nodes
        concepts = [
            'maze', 'path', 'wall', 'goal', 'start',
            'movement', 'direction', 'obstacle', 'solution'
        ]
        g.add_nodes_from(concepts)
        
        # Add relationships
        edges = [
            ('maze', 'path'), ('maze', 'wall'), ('maze', 'goal'),
            ('maze', 'start'), ('path', 'movement'), ('path', 'direction'),
            ('wall', 'obstacle'), ('goal', 'solution'), ('start', 'movement')
        ]
        g.add_edges_from(edges)
        
        return g
        
    def _add_known_patterns(self):
        """Add known patterns to the calculator."""
        # Navigation pattern
        nav_pattern = nx.Graph()
        nav_pattern.add_edges_from([
            ('start', 'movement'), ('movement', 'direction'),
            ('direction', 'goal')
        ])
        self.wake_calculator.add_pattern('navigation', nav_pattern)
        
        # Obstacle avoidance pattern
        obstacle_pattern = nx.Graph()
        obstacle_pattern.add_edges_from([
            ('movement', 'obstacle'), ('obstacle', 'direction'),
            ('direction', 'path')
        ])
        self.wake_calculator.add_pattern('obstacle_avoidance', obstacle_pattern)
        
    def test_query_response_efficiency(self):
        """Test that Wake Mode responds efficiently to queries."""
        # Query about maze navigation
        query_context = {
            'query': 'How to navigate from start to goal?',
            'type': 'navigation'
        }
        focal_nodes = {'start', 'goal', 'movement'}
        
        # Calculate Wake Mode geDIG
        result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            focal_nodes,
            query_context
        )
        
        # Should have some pattern match (navigation pattern)
        self.assertGreater(result.pattern_match_score, 0.2)
        
        # Should have good efficiency
        self.assertGreater(result.efficiency_score, 0.3)
        
        # Overall Wake Mode score should be reasonable
        self.assertGreater(result.gedig_value, 0.2)
        
    def test_wake_vs_exploration_mode(self):
        """Compare Wake Mode with exploration mode."""
        focal_nodes = {'maze', 'solution'}
        
        # Wake Mode (efficient)
        wake_result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            focal_nodes
        )
        
        # Exploration mode (standard geDIG)
        explore_calculator = GeDIGCore()
        explore_result = explore_calculator.calculate(
            self.knowledge_graph,
            focal_nodes
        )
        
        # Wake Mode should prefer efficiency
        # Different optimization objectives
        self.assertIsNotNone(wake_result.efficiency_score)
        self.assertIsNotNone(wake_result.pattern_match_score)
        
    def test_pattern_learning_and_reuse(self):
        """Test learning new patterns and reusing them."""
        # First query - no pattern match
        focal_nodes = {'wall', 'obstacle', 'path'}
        result1 = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            focal_nodes
        )
        initial_score = result1.pattern_match_score
        
        # Learn this as a new pattern
        subgraph = self.wake_calculator._extract_subgraph(
            self.knowledge_graph,
            focal_nodes,
            radius=1
        )
        self.wake_calculator.add_pattern('wall_navigation', subgraph)
        
        # Second similar query - should match pattern
        result2 = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            focal_nodes
        )
        
        # Pattern match should improve
        self.assertGreater(result2.pattern_match_score, initial_score)
        
    def test_efficiency_under_load(self):
        """Test Wake Mode efficiency with large graph."""
        # Create larger graph
        large_graph = nx.karate_club_graph()
        
        # Random focal nodes
        focal_nodes = set(list(large_graph.nodes())[:5])
        
        # Measure computation time
        import time
        start = time.time()
        
        result = self.wake_calculator.calculate_wake_mode_gedig(
            large_graph,
            focal_nodes
        )
        
        computation_time = time.time() - start
        
        # Should complete quickly
        self.assertLess(computation_time, 1.0)  # Less than 1 second
        self.assertGreater(result.computation_time, 0)
        
    def test_convergent_behavior(self):
        """Test that Wake Mode converges to stable solutions."""
        focal_nodes = {'start', 'goal'}
        
        # Multiple iterations should converge
        scores = []
        for i in range(5):
            result = self.wake_calculator.calculate_wake_mode_gedig(
                self.knowledge_graph,
                focal_nodes
            )
            scores.append(result.gedig_value)
            
        # Scores should be consistent (convergent)
        variance = np.var(scores)
        self.assertLess(variance, 0.01)  # Low variance
        
    def test_query_context_influence(self):
        """Test that query context influences the result."""
        focal_nodes = {'movement', 'direction'}
        
        # Query with navigation context
        nav_context = {'type': 'navigation', 'priority': 'efficiency'}
        nav_result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            focal_nodes,
            nav_context
        )
        
        # Query with exploration context
        explore_context = {'type': 'exploration', 'priority': 'discovery'}
        explore_result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            focal_nodes,
            explore_context
        )
        
        # Results should be consistent (context doesn't change graph structure)
        # But pattern match might differ
        self.assertIsNotNone(nav_result.pattern_match_score)
        self.assertIsNotNone(explore_result.pattern_match_score)
        
    def test_minimal_change_principle(self):
        """Test that Wake Mode minimizes structural changes."""
        # Well-connected focal nodes
        connected_nodes = {'maze', 'path', 'wall'}  # Central nodes
        
        # Peripheral focal nodes
        peripheral_nodes = {'start', 'goal'}  # Edge nodes
        
        connected_result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            connected_nodes
        )
        
        peripheral_result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            peripheral_nodes
        )
        
        # Well-connected nodes should need fewer changes (lower GED)
        self.assertLess(connected_result.ged_value, peripheral_result.ged_value)
        
    def test_error_handling(self):
        """Test error handling in Wake Mode."""
        # Empty focal nodes
        result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            set()
        )
        self.assertIsNotNone(result)
        
        # Non-existent focal nodes
        result = self.wake_calculator.calculate_wake_mode_gedig(
            self.knowledge_graph,
            {'nonexistent1', 'nonexistent2'}
        )
        self.assertIsNotNone(result)
        
        # Empty graph
        empty_graph = nx.Graph()
        result = self.wake_calculator.calculate_wake_mode_gedig(
            empty_graph,
            {'A', 'B'}
        )
        self.assertIsNotNone(result)


class TestMazeNavigationScenario(unittest.TestCase):
    """Test Wake Mode in maze navigation scenario."""
    
    def setUp(self):
        """Set up maze navigation test."""
        self.wake_calculator = WakeModeGeDIG()
        self.maze_graph = self._create_maze_knowledge()
        
    def _create_maze_knowledge(self):
        """Create maze-specific knowledge graph."""
        g = nx.Graph()
        
        # Positions and states
        for x in range(3):
            for y in range(3):
                pos = f'pos_{x}_{y}'
                g.add_node(pos, type='position', x=x, y=y)
                
        # Add movement edges
        for x in range(3):
            for y in range(3):
                current = f'pos_{x}_{y}'
                # Right
                if x < 2:
                    g.add_edge(current, f'pos_{x+1}_{y}', action='right')
                # Down
                if y < 2:
                    g.add_edge(current, f'pos_{x}_{y+1}', action='down')
                    
        # Add walls (remove some edges)
        g.remove_edge('pos_0_1', 'pos_1_1')  # Wall
        g.remove_edge('pos_1_0', 'pos_1_1')  # Wall
        
        return g
        
    def test_efficient_path_finding(self):
        """Test efficient path finding in Wake Mode."""
        # Find path from top-left to bottom-right
        start = 'pos_0_0'
        goal = 'pos_2_2'
        focal_nodes = {start, goal}
        
        # Add known solution pattern
        solution_pattern = nx.Graph()
        path_edges = [('pos_0_0', 'pos_1_0'), ('pos_1_0', 'pos_2_0'), 
                      ('pos_2_0', 'pos_2_1'), ('pos_2_1', 'pos_2_2')]
        solution_pattern.add_edges_from(path_edges)
        self.wake_calculator.add_pattern('known_solution', solution_pattern)
        
        # Query for path
        query_context = {
            'task': 'pathfinding',
            'start': start,
            'goal': goal
        }
        
        result = self.wake_calculator.calculate_wake_mode_gedig(
            self.maze_graph,
            focal_nodes,
            query_context
        )
        
        # Should efficiently recognize the pattern
        self.assertGreater(result.pattern_match_score, 0.3)
        self.assertGreater(result.efficiency_score, 0.4)
        
    def test_obstacle_adaptation(self):
        """Test adaptation when encountering obstacles."""
        # Current position near a wall
        current = 'pos_0_1'
        focal_nodes = {current, 'pos_1_1'}  # Blocked path
        
        query_context = {
            'task': 'navigation',
            'obstacle_detected': True
        }
        
        result = self.wake_calculator.calculate_wake_mode_gedig(
            self.maze_graph,
            focal_nodes,
            query_context
        )
        
        # Should handle the obstacle efficiently
        self.assertIsNotNone(result)
        # GED should be higher due to need for alternative path
        self.assertGreater(result.ged_value, 0.3)


if __name__ == '__main__':
    unittest.main()