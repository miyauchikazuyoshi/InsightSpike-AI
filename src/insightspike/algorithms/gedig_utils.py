"""Shared utilities for GED and IG calculations.

Common functions used by both Pure and Full geDIG implementations
to avoid code duplication and ensure consistency.
"""

import math
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class GEDCalculationUtils:
    """Utilities for Graph Edit Distance calculations."""
    
    @staticmethod
    def normalize_ged_value(raw_ged: float, graph_size: int) -> float:
        """Normalize GED value by graph size.
        
        Args:
            raw_ged: Raw GED value 
            graph_size: Number of nodes in graph
            
        Returns:
            Normalized GED in [0, 1] range
        """
        if graph_size <= 1:
            return 0.0
        
        # Normalize by theoretical maximum GED
        max_possible_ged = graph_size * (graph_size - 1) / 2  # Complete graph edges
        
        if max_possible_ged == 0:
            return 0.0
        
        normalized = abs(raw_ged) / max_possible_ged
        return min(normalized, 1.0)
    
    @staticmethod
    def compute_structural_difference(
        graph_before: nx.Graph, 
        graph_after: nx.Graph
    ) -> Dict[str, float]:
        """Compute structural differences between two graphs.
        
        Args:
            graph_before: Original graph
            graph_after: Modified graph
            
        Returns:
            Dict with structural difference metrics
        """
        try:
            # Node differences
            nodes_before = set(graph_before.nodes())
            nodes_after = set(graph_after.nodes())
            
            nodes_added = len(nodes_after - nodes_before)
            nodes_removed = len(nodes_before - nodes_after)
            nodes_common = len(nodes_before & nodes_after)
            
            # Edge differences  
            edges_before = set(graph_before.edges())
            edges_after = set(graph_after.edges())
            
            edges_added = len(edges_after - edges_before)
            edges_removed = len(edges_before - edges_after)
            edges_common = len(edges_before & edges_after)
            
            # Calculate normalized differences
            total_nodes = max(len(nodes_before), len(nodes_after), 1)
            total_edges = max(len(edges_before), len(edges_after), 1)
            
            return {
                'node_change_ratio': (nodes_added + nodes_removed) / total_nodes,
                'edge_change_ratio': (edges_added + edges_removed) / total_edges,
                'structural_stability': (nodes_common + edges_common) / (total_nodes + total_edges),
                'growth_factor': len(nodes_after) / max(len(nodes_before), 1)
            }
            
        except Exception as e:
            logger.warning(f"Error computing structural difference: {e}")
            return {
                'node_change_ratio': 0.0,
                'edge_change_ratio': 0.0, 
                'structural_stability': 1.0,
                'growth_factor': 1.0
            }


class IGCalculationUtils:
    """Utilities for Information Gain calculations."""
    
    @staticmethod
    def compute_degree_entropy(graph: nx.Graph) -> float:
        """Compute entropy of degree distribution.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Entropy value (higher = more diverse degree distribution)
        """
        if len(graph.nodes()) <= 1:
            return 0.0
        
        try:
            # Get degree sequence
            degrees = [d for n, d in graph.degree()]
            
            if not degrees:
                return 0.0
            
            # Count degree frequencies
            degree_counts = {}
            for degree in degrees:
                degree_counts[degree] = degree_counts.get(degree, 0) + 1
            
            total_nodes = len(degrees)
            
            # Calculate entropy
            entropy = 0.0
            for count in degree_counts.values():
                probability = count / total_nodes
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Error computing degree entropy: {e}")
            return 0.0
    
    @staticmethod
    def compute_clustering_entropy(graph: nx.Graph) -> float:
        """Compute entropy of clustering coefficient distribution.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Clustering entropy value
        """
        if len(graph.nodes()) <= 2:
            return 0.0
        
        try:
            # Get clustering coefficients
            clustering_dict = nx.clustering(graph)
            clustering_values = list(clustering_dict.values())
            
            if not clustering_values:
                return 0.0
            
            # Bin clustering values (0.0-1.0 in 10 bins)
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            bin_counts = [0] * (len(bins) - 1)
            
            for value in clustering_values:
                for i in range(len(bins) - 1):
                    if bins[i] <= value < bins[i + 1]:
                        bin_counts[i] += 1
                        break
                else:
                    # Handle value == 1.0
                    if value == 1.0:
                        bin_counts[-1] += 1
            
            # Calculate entropy
            total_nodes = len(clustering_values)
            entropy = 0.0
            
            for count in bin_counts:
                if count > 0:
                    probability = count / total_nodes
                    entropy -= probability * math.log2(probability)
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Error computing clustering entropy: {e}")
            return 0.0
    
    @staticmethod
    def compute_information_gain(
        graph_before: nx.Graph,
        graph_after: nx.Graph,
        method: str = "degree"
    ) -> float:
        """Compute information gain between two graph states.
        
        Args:
            graph_before: Original graph state
            graph_after: New graph state  
            method: Method for IG calculation ("degree" or "clustering")
            
        Returns:
            Information gain value (positive = information increase)
        """
        try:
            if method == "degree":
                entropy_before = IGCalculationUtils.compute_degree_entropy(graph_before)
                entropy_after = IGCalculationUtils.compute_degree_entropy(graph_after)
            elif method == "clustering":
                entropy_before = IGCalculationUtils.compute_clustering_entropy(graph_before)
                entropy_after = IGCalculationUtils.compute_clustering_entropy(graph_after)
            else:
                # Combined method
                deg_before = IGCalculationUtils.compute_degree_entropy(graph_before)
                deg_after = IGCalculationUtils.compute_degree_entropy(graph_after)
                clust_before = IGCalculationUtils.compute_clustering_entropy(graph_before)
                clust_after = IGCalculationUtils.compute_clustering_entropy(graph_after)
                
                entropy_before = (deg_before + clust_before) / 2
                entropy_after = (deg_after + clust_after) / 2
            
            return entropy_after - entropy_before
            
        except Exception as e:
            logger.error(f"Error computing information gain: {e}")
            return 0.0


class GeDIGComputationUtils:
    """High-level utilities for geDIG computations."""
    
    @staticmethod
    def compute_gedig_score(
        ged_value: float,
        ig_value: float, 
        k_value: float = 2.0,
        normalization: bool = True
    ) -> Dict[str, float]:
        """Compute geDIG score with optional normalization.
        
        Args:
            ged_value: Graph Edit Distance value
            ig_value: Information Gain value
            k_value: Scaling factor for IG
            normalization: Whether to apply sigmoid normalization
            
        Returns:
            Dict with geDIG computation results
        """
        # Core geDIG formula: GED - k * IG
        raw_gedig = ged_value - k_value * ig_value
        
        # Optional normalization
        if normalization:
            normalized_gedig = GeDIGComputationUtils._sigmoid_normalize(raw_gedig)
        else:
            normalized_gedig = raw_gedig
        
        return {
            'gedig': normalized_gedig,
            'raw_gedig': raw_gedig,
            'ged': ged_value,
            'ig': ig_value,
            'k_value': k_value,
            'normalized': normalization
        }
    
    @staticmethod
    def _sigmoid_normalize(value: float, scale: float = 1.0) -> float:
        """Apply sigmoid normalization to map value to [0, 1] range."""
        try:
            return 1.0 / (1.0 + math.exp(-value / scale))
        except OverflowError:
            return 1.0 if value > 0 else 0.0
    
    @staticmethod
    def estimate_k_value(
        ged_values: List[float],
        ig_values: List[float],
        target_gedig: float = 0.0
    ) -> Optional[float]:
        """Estimate optimal k value for given GED/IG sequences.
        
        Args:
            ged_values: Sequence of GED values
            ig_values: Sequence of IG values  
            target_gedig: Target geDIG value to optimize for
            
        Returns:
            Estimated k value or None if estimation fails
        """
        if len(ged_values) != len(ig_values) or len(ged_values) < 2:
            return None
        
        try:
            # Simple least squares estimation
            # Minimize sum((GED_i - k*IG_i - target)^2)
            
            sum_ig_squared = sum(ig ** 2 for ig in ig_values if abs(ig) > 1e-12)
            sum_ged_ig = sum(ged * ig for ged, ig in zip(ged_values, ig_values) if abs(ig) > 1e-12)
            sum_target_ig = target_gedig * sum(ig for ig in ig_values if abs(ig) > 1e-12)
            
            if abs(sum_ig_squared) < 1e-12:
                return None
            
            k_estimate = (sum_ged_ig - sum_target_ig) / sum_ig_squared
            
            # Clamp to reasonable range
            return max(0.1, min(10.0, k_estimate))
            
        except Exception as e:
            logger.warning(f"K value estimation failed: {e}")
            return None


class GraphValidationUtils:
    """Utilities for graph validation and sanitization."""
    
    @staticmethod
    def validate_graph(graph: Any) -> bool:
        """Validate that object is a proper NetworkX graph.
        
        Args:
            graph: Object to validate
            
        Returns:
            True if valid NetworkX graph, False otherwise
        """
        try:
            return isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
        except Exception:
            return False
    
    @staticmethod
    def sanitize_graph_for_computation(graph: nx.Graph) -> nx.Graph:
        """Sanitize graph for safe computation.
        
        Args:
            graph: Input graph
            
        Returns:
            Sanitized copy of graph
        """
        try:
            # Create clean copy
            clean_graph = graph.copy()
            
            # Remove self loops if present
            clean_graph.remove_edges_from(nx.selfloop_edges(clean_graph))
            
            # Remove isolated nodes with no meaningful attributes
            isolated_nodes = list(nx.isolates(clean_graph))
            for node in isolated_nodes:
                if not clean_graph.nodes[node]:  # No attributes
                    clean_graph.remove_node(node)
            
            return clean_graph
            
        except Exception as e:
            logger.error(f"Graph sanitization failed: {e}")
            # Return minimal empty graph
            return nx.Graph()
    
    @staticmethod
    def compute_graph_complexity_metrics(graph: nx.Graph) -> Dict[str, float]:
        """Compute complexity metrics for a graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dict with complexity metrics
        """
        try:
            n_nodes = len(graph.nodes())
            n_edges = len(graph.edges())
            
            if n_nodes <= 1:
                return {
                    'density': 0.0,
                    'complexity_ratio': 0.0,
                    'efficiency': 0.0
                }
            
            # Graph density
            max_edges = n_nodes * (n_nodes - 1) / 2
            density = n_edges / max_edges if max_edges > 0 else 0.0
            
            # Complexity ratio (actual vs minimum spanning tree)
            min_edges = max(0, n_nodes - 1)  # MST edges
            complexity_ratio = n_edges / max(min_edges, 1)
            
            # Efficiency (inverse of path lengths)
            try:
                avg_path_length = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf')
                efficiency = 1.0 / avg_path_length if avg_path_length > 0 and avg_path_length != float('inf') else 0.0
            except:
                efficiency = 0.0
            
            return {
                'density': density,
                'complexity_ratio': complexity_ratio,
                'efficiency': efficiency
            }
            
        except Exception as e:
            logger.error(f"Error computing complexity metrics: {e}")
            return {
                'density': 0.0,
                'complexity_ratio': 0.0,
                'efficiency': 0.0
            }


__all__ = [
    'GEDCalculationUtils',
    'IGCalculationUtils', 
    'GeDIGComputationUtils',
    'GraphValidationUtils'
]