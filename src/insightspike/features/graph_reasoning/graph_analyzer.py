"""
Graph Analyzer
==============

Analyzes graph structures for metrics and insights.
Separated from L3GraphReasoner to follow Single Responsibility Principle.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """Analyzes graph structures and calculates metrics."""
    
    def __init__(self, config=None):
        self.config = config or {}
        
    def calculate_metrics(
        self,
        current_graph: Data,
        previous_graph: Optional[Data],
        delta_ged_func,
        delta_ig_func
    ) -> Dict[str, float]:
        """Calculate ΔGED and ΔIG metrics between graphs."""
        if previous_graph is None:
            return {
                "delta_ged": 0.0,
                "delta_ig": 0.0,
                "graph_size_current": current_graph.num_nodes if current_graph else 0,
                "graph_size_previous": 0,
            }
        
        try:
            # Calculate graph edit distance change
            # Convert PyTorch Geometric graphs to NetworkX for GED calculation
            import networkx as nx

            # Create NetworkX graphs from PyTorch Geometric data
            g_prev = nx.Graph()
            g_curr = nx.Graph()
            
            # Add nodes
            for i in range(previous_graph.num_nodes):
                g_prev.add_node(i)
            for i in range(current_graph.num_nodes):
                g_curr.add_node(i)
                
            # Add edges if available
            if hasattr(previous_graph, 'edge_index') and previous_graph.edge_index is not None:
                edges = previous_graph.edge_index.t().tolist()
                g_prev.add_edges_from(edges)
            if hasattr(current_graph, 'edge_index') and current_graph.edge_index is not None:
                edges = current_graph.edge_index.t().tolist()
                g_curr.add_edges_from(edges)
            
            ged = delta_ged_func(g_prev, g_curr)
            
            # Calculate information gain change
            # Extract feature vectors for IG calculation
            prev_vecs = previous_graph.x.numpy() if hasattr(previous_graph, 'x') and previous_graph.x is not None else None
            curr_vecs = current_graph.x.numpy() if hasattr(current_graph, 'x') and current_graph.x is not None else None
            
            if prev_vecs is not None and curr_vecs is not None:
                ig = delta_ig_func(prev_vecs, curr_vecs)
            else:
                ig = 0.0
            
            return {
                "delta_ged": float(ged),
                "delta_ig": float(ig),
                "graph_size_current": current_graph.num_nodes,
                "graph_size_previous": previous_graph.num_nodes,
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {
                "delta_ged": 0.0,
                "delta_ig": 0.0,
                "graph_size_current": current_graph.num_nodes if current_graph else 0,
                "graph_size_previous": previous_graph.num_nodes if previous_graph else 0,
            }
    
    def detect_spike(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> bool:
        """Detect if current state represents an insight spike."""
        high_ged = metrics.get("delta_ged", 0) > thresholds.get("ged", -0.5)
        high_ig = metrics.get("delta_ig", 0) > thresholds.get("ig", 0.2)
        low_conflict = conflicts.get("total", 1.0) < thresholds.get("conflict", 0.5)
        
        return high_ged and high_ig and low_conflict
    
    def assess_quality(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float]
    ) -> float:
        """Assess overall quality of reasoning process."""
        # Combine multiple factors
        metric_score = (metrics.get("delta_ged", 0) + metrics.get("delta_ig", 0)) / 2
        conflict_penalty = conflicts.get("total", 0)
        
        quality = max(0.0, min(1.0, metric_score - conflict_penalty))
        return float(quality)