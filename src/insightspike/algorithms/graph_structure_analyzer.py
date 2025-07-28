"""
Graph Structure Analyzer
=======================

Analyzes graph structural changes for insight detection.
Separates GED (distance) from structural improvement metrics.
"""

import logging
import networkx as nx
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GraphStructureAnalyzer:
    """Analyzes structural changes in graphs for insight detection."""
    
    def analyze_structural_change(
        self, 
        graph_before: Any, 
        graph_after: Any
    ) -> Dict[str, float]:
        """
        Analyze structural changes between two graphs.
        
        Returns:
            Dict containing:
            - ged: Graph Edit Distance (always non-negative)
            - structural_improvement: Positive = improvement, Negative = degradation
            - efficiency_change: Change in graph efficiency
            - hub_formation: Hub structure formation score
            - complexity_reduction: Complexity reduction score
        """
        if graph_before is None or graph_after is None:
            return self._empty_result()
            
        try:
            # 1. Calculate raw GED (always non-negative)
            ged = self._calculate_ged(graph_before, graph_after)
            
            # 2. Calculate structural improvements
            efficiency_change = self._calculate_efficiency_change(graph_before, graph_after)
            hub_score = self._calculate_hub_formation(graph_before, graph_after)
            complexity_reduction = self._calculate_complexity_reduction(graph_before, graph_after)
            
            # 3. Combined structural improvement score
            structural_improvement = (
                0.5 * efficiency_change +
                0.3 * hub_score +
                0.2 * complexity_reduction
            )
            
            return {
                "ged": ged,  # Always non-negative
                "structural_improvement": structural_improvement,  # Can be negative
                "efficiency_change": efficiency_change,
                "hub_formation": hub_score,
                "complexity_reduction": complexity_reduction,
                "nodes_before": graph_before.number_of_nodes(),
                "nodes_after": graph_after.number_of_nodes(),
                "edges_before": graph_before.number_of_edges(),
                "edges_after": graph_after.number_of_edges(),
            }
            
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            return self._empty_result()
    
    def _calculate_ged(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Calculate Graph Edit Distance (always non-negative)."""
        try:
            ged = nx.graph_edit_distance(g1, g2, timeout=1.0)
            return float(ged) if ged is not None else self._approximate_ged(g1, g2)
        except:
            return self._approximate_ged(g1, g2)
    
    def _approximate_ged(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Fast GED approximation."""
        node_diff = abs(g1.number_of_nodes() - g2.number_of_nodes())
        edge_diff = abs(g1.number_of_edges() - g2.number_of_edges())
        return float(node_diff + edge_diff * 0.5)
    
    def _calculate_efficiency_change(self, before: nx.Graph, after: nx.Graph) -> float:
        """
        Calculate change in graph efficiency.
        Positive = improvement, Negative = degradation.
        """
        eff_before = self._calculate_efficiency(before)
        eff_after = self._calculate_efficiency(after)
        return eff_after - eff_before
    
    def _calculate_efficiency(self, graph: nx.Graph) -> float:
        """Calculate graph efficiency (0 to 1)."""
        if graph.number_of_nodes() <= 1:
            return 1.0
            
        try:
            # Global efficiency
            global_eff = nx.global_efficiency(graph)
            
            # Local efficiency (clustering)
            if graph.number_of_nodes() >= 3:
                clustering = nx.average_clustering(graph)
            else:
                clustering = 0.0
            
            # Combine metrics
            return 0.7 * global_eff + 0.3 * clustering
            
        except:
            # Fallback to density
            n = graph.number_of_nodes()
            m = graph.number_of_edges()
            max_edges = n * (n - 1) / 2
            return m / max_edges if max_edges > 0 else 0.0
    
    def _calculate_hub_formation(self, before: nx.Graph, after: nx.Graph) -> float:
        """
        Calculate hub formation score.
        Positive = hub structure emerging.
        """
        # Check for nodes with significantly increased centrality
        try:
            # Degree centrality change
            cent_before = nx.degree_centrality(before)
            cent_after = nx.degree_centrality(after)
            
            # Find maximum centrality increase
            max_increase = 0.0
            for node in after.nodes():
                if node in cent_before:
                    increase = cent_after[node] - cent_before[node]
                    max_increase = max(max_increase, increase)
                else:
                    # New node - check if it's a hub
                    max_increase = max(max_increase, cent_after[node])
            
            return max_increase
            
        except:
            return 0.0
    
    def _calculate_complexity_reduction(self, before: nx.Graph, after: nx.Graph) -> float:
        """
        Calculate complexity reduction score.
        Positive = simpler structure.
        """
        # Multiple factors for complexity
        complexity_score = 0.0
        
        # 1. Diameter reduction (smaller = simpler)
        try:
            if nx.is_connected(before) and nx.is_connected(after):
                diam_before = nx.diameter(before)
                diam_after = nx.diameter(after)
                if diam_before > 0:
                    complexity_score += (diam_before - diam_after) / diam_before
        except:
            pass
        
        # 2. Average path length reduction
        try:
            apl_before = nx.average_shortest_path_length(before)
            apl_after = nx.average_shortest_path_length(after)
            if apl_before > 0:
                complexity_score += (apl_before - apl_after) / apl_before
        except:
            pass
        
        # 3. Density increase with more nodes (rare but valuable)
        n_before = before.number_of_nodes()
        n_after = after.number_of_nodes()
        if n_after > n_before:
            dens_before = nx.density(before)
            dens_after = nx.density(after)
            if dens_after > dens_before:
                complexity_score += 0.5  # Bonus for maintaining density
        
        return complexity_score
    
    def _empty_result(self) -> Dict[str, float]:
        """Return empty result structure."""
        return {
            "ged": 0.0,
            "structural_improvement": 0.0,
            "efficiency_change": 0.0,
            "hub_formation": 0.0,
            "complexity_reduction": 0.0,
            "nodes_before": 0,
            "nodes_after": 0,
            "edges_before": 0,
            "edges_after": 0,
        }


def analyze_graph_structure(graph_before: Any, graph_after: Any) -> Dict[str, float]:
    """Convenience function for structure analysis."""
    analyzer = GraphStructureAnalyzer()
    return analyzer.analyze_structural_change(graph_before, graph_after)