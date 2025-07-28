"""
Fix GraphAnalyzer to handle both NetworkX and PyTorch Geometric graphs
"""

import logging
import networkx as nx
from typing import Union, Dict, Optional

logger = logging.getLogger(__name__)


def apply_graph_analyzer_networkx_fix():
    """Fix GraphAnalyzer to handle NetworkX graphs properly."""
    from ..features.graph_reasoning.graph_analyzer import GraphAnalyzer
    
    original_calculate_metrics = GraphAnalyzer.calculate_metrics
    
    def patched_calculate_metrics(self, current_graph, previous_graph, delta_ged_func, delta_ig_func) -> Dict[str, float]:
        """Calculate metrics handling both NetworkX and PyG graphs."""
        # Handle NetworkX graphs
        if isinstance(current_graph, nx.Graph):
            if previous_graph is None:
                return {
                    "delta_ged": 0.0,
                    "delta_ig": 0.0,
                    "graph_size_current": current_graph.number_of_nodes() if current_graph else 0,
                    "graph_size_previous": 0,
                }
            
            try:
                # Both are NetworkX graphs - calculate directly
                ged = delta_ged_func(previous_graph, current_graph)
                ig = delta_ig_func(previous_graph, current_graph)
                
                return {
                    "delta_ged": ged,
                    "delta_ig": ig,
                    "graph_size_current": current_graph.number_of_nodes(),
                    "graph_size_previous": previous_graph.number_of_nodes() if previous_graph else 0,
                }
            except Exception as e:
                logger.error(f"Error calculating graph metrics: {e}")
                return {
                    "delta_ged": 0.0,
                    "delta_ig": 0.0,
                    "graph_size_current": current_graph.number_of_nodes() if current_graph else 0,
                    "graph_size_previous": previous_graph.number_of_nodes() if previous_graph else 0,
                }
        
        # Otherwise use original method (for PyG graphs)
        return original_calculate_metrics(self, current_graph, previous_graph, delta_ged_func, delta_ig_func)
    
    GraphAnalyzer.calculate_metrics = patched_calculate_metrics
    logger.info("Applied GraphAnalyzer NetworkX fix")