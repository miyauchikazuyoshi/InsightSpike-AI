"""
Wake Mode geDIG Implementation
==============================

Wake Mode implementation for geDIG that minimizes structural changes
and maximizes efficiency in query processing.

Based on predictive coding principles:
- Minimize prediction error (geDIG minimization)
- Use existing patterns efficiently
- Converge to known solutions
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .gedig_core import GeDIGCore, GeDIGResult

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for geDIG calculation."""
    WAKE = "wake"  # Query-driven, minimize geDIG
    SLEEP = "sleep"  # Autonomous consolidation (future)


@dataclass
class WakeModeResult(GeDIGResult):
    """Extended result for Wake Mode processing."""
    mode: str = "wake"
    pattern_match_score: float = 0.0
    convergence_score: float = 0.0
    efficiency_score: float = 0.0
    nearest_patterns: Optional[List[str]] = None


class WakeModeGeDIG(GeDIGCore):
    """
    Wake Mode geDIG calculator for efficient query processing.
    
    Key principles:
    - Minimize structural changes (low GED)
    - Maximize pattern reuse (convergent IG)
    - Efficient pathfinding to known solutions
    """
    
    def __init__(self, 
                 # Inherit base parameters
                 **base_params):
        """Initialize Wake Mode calculator."""
        super().__init__(**base_params)
        
        # Wake mode specific parameters
        self.pattern_similarity_threshold = 0.7
        self.convergence_weight = 0.6
        self.efficiency_weight = 0.4
        
        # Pattern memory for efficient matching
        self.known_patterns: Dict[str, nx.Graph] = {}
        self.pattern_embeddings: Dict[str, np.ndarray] = {}
        
    def calculate_wake_mode_gedig(self,
                                  graph: nx.Graph,
                                  focal_nodes: Set[str],
                                  query_context: Optional[Dict[str, Any]] = None) -> WakeModeResult:
        """
        Calculate geDIG in Wake Mode - minimize changes for efficient response.
        
        Args:
            graph: Current knowledge graph
            focal_nodes: Nodes relevant to current query
            query_context: Additional context about the query
            
        Returns:
            WakeModeResult with minimized geDIG value
        """
        start_time = time.time()
        
        # 1. Find nearest existing patterns
        pattern_match_score = self._find_nearest_pattern(graph, focal_nodes, query_context)
        
        # 2. Calculate minimal structural changes needed
        minimal_ged = self._calculate_minimal_ged(graph, focal_nodes)
        
        # 3. Calculate convergent information gain
        convergent_ig = self._calculate_convergent_ig(graph, focal_nodes)
        
        # 4. Wake Mode scoring: smaller is better (inverse of sleep mode)
        # This represents the "prediction error" to minimize
        efficiency_score = 1.0 / (1.0 + minimal_ged * convergent_ig)
        
        # 5. Combined Wake Mode geDIG
        wake_gedig = (
            self.convergence_weight * pattern_match_score +
            self.efficiency_weight * efficiency_score
        )
        
        # 6. Create result
        result = WakeModeResult(
            gedig_value=wake_gedig,
            ged_value=minimal_ged,
            ig_value=convergent_ig,
            structural_improvement=minimal_ged,
            information_integration=convergent_ig,
            pattern_match_score=pattern_match_score,
            convergence_score=pattern_match_score * efficiency_score,
            efficiency_score=efficiency_score,
            computation_time=time.time() - start_time,
            focal_nodes=focal_nodes,
            mode="wake"
        )
        
        logger.debug(f"Wake Mode geDIG: {wake_gedig:.4f} "
                    f"(pattern: {pattern_match_score:.4f}, "
                    f"efficiency: {efficiency_score:.4f})")
        
        return result
    
    def _find_nearest_pattern(self,
                             graph: nx.Graph,
                             focal_nodes: Set[str],
                             query_context: Optional[Dict[str, Any]]) -> float:
        """
        Find the nearest known pattern to current query.
        
        Returns similarity score [0, 1] where 1 is perfect match.
        """
        if not self.known_patterns:
            return 0.0
            
        best_similarity = 0.0
        best_pattern = None
        
        # Extract subgraph around focal nodes
        query_subgraph = self._extract_subgraph(graph, focal_nodes, radius=2)
        
        for pattern_id, pattern_graph in self.known_patterns.items():
            # Calculate graph similarity
            similarity = self._calculate_graph_similarity(query_subgraph, pattern_graph)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern = pattern_id
                
        # Bonus for query context match
        if query_context and best_pattern:
            context_bonus = self._calculate_context_similarity(
                query_context, 
                self.pattern_embeddings.get(best_pattern, {})
            )
            best_similarity = 0.8 * best_similarity + 0.2 * context_bonus
            
        return best_similarity
    
    def _calculate_minimal_ged(self, 
                              graph: nx.Graph,
                              focal_nodes: Set[str]) -> float:
        """
        Calculate minimal Graph Edit Distance needed.
        
        In Wake Mode, we want to minimize structural changes.
        Lower values indicate more efficient solutions.
        """
        # Count existing edges around focal nodes
        existing_edges = 0
        for node in focal_nodes:
            if node in graph:
                existing_edges += graph.degree(node)
                
        # Minimal changes needed (normalized)
        if not focal_nodes:
            return 1.0
            
        # Lower score for well-connected focal nodes (less change needed)
        avg_degree = existing_edges / len(focal_nodes)
        normalized_degree = avg_degree / (1.0 + np.log1p(len(graph.nodes())))
        
        # Minimal GED: inverse of connectivity
        minimal_ged = 1.0 / (1.0 + normalized_degree)
        
        return minimal_ged
    
    def _calculate_convergent_ig(self,
                                graph: nx.Graph,
                                focal_nodes: Set[str]) -> float:
        """
        Calculate convergent Information Gain.
        
        In Wake Mode, we prefer low entropy (high certainty).
        This is the opposite of exploration mode.
        """
        if not focal_nodes:
            return 0.0
            
        # Calculate entropy around focal nodes
        entropies = []
        for node in focal_nodes:
            if node in graph:
                # Node degree distribution entropy
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    degrees = [graph.degree(n) for n in neighbors]
                    # Normalize
                    total = sum(degrees)
                    if total > 0:
                        probs = [d/total for d in degrees]
                        # Shannon entropy
                        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                        entropies.append(entropy)
                        
        if not entropies:
            # If no valid nodes, return moderate value
            return 0.5
            
        # Low entropy = high certainty = good for Wake Mode
        avg_entropy = np.mean(entropies)
        max_entropy = np.log(len(graph.nodes())) if len(graph.nodes()) > 1 else 1.0
        
        # Convergent IG: inverse of normalized entropy
        convergent_ig = 1.0 - (avg_entropy / (max_entropy + 1e-10))
        
        return max(0.0, min(1.0, convergent_ig))  # Clamp to [0, 1]
    
    def _extract_subgraph(self, 
                         graph: nx.Graph,
                         focal_nodes: Set[str],
                         radius: int = 2) -> nx.Graph:
        """Extract subgraph within radius of focal nodes."""
        nodes_to_include = set()
        
        for node in focal_nodes:
            if node in graph:
                # BFS to find nodes within radius
                for n in nx.single_source_shortest_path_length(graph, node, cutoff=radius):
                    nodes_to_include.add(n)
                    
        return graph.subgraph(nodes_to_include).copy()
    
    def _calculate_graph_similarity(self,
                                   g1: nx.Graph,
                                   g2: nx.Graph) -> float:
        """
        Calculate similarity between two graphs.
        
        Simple implementation - can be enhanced with graph kernels.
        """
        if len(g1.nodes()) == 0 or len(g2.nodes()) == 0:
            return 0.0
            
        # Node overlap
        node_overlap = len(set(g1.nodes()) & set(g2.nodes()))
        node_union = len(set(g1.nodes()) | set(g2.nodes()))
        node_similarity = node_overlap / (node_union + 1e-10)
        
        # Edge overlap
        edge_overlap = len(set(g1.edges()) & set(g2.edges()))
        edge_union = len(set(g1.edges()) | set(g2.edges()))
        edge_similarity = edge_overlap / (edge_union + 1e-10)
        
        # Combined similarity
        return 0.5 * node_similarity + 0.5 * edge_similarity
    
    def _calculate_context_similarity(self,
                                    query_context: Dict[str, Any],
                                    pattern_context: Dict[str, Any]) -> float:
        """Calculate similarity between query and pattern contexts."""
        # Simple key overlap for now
        if not query_context or not pattern_context:
            return 0.0
            
        key_overlap = len(set(query_context.keys()) & set(pattern_context.keys()))
        key_union = len(set(query_context.keys()) | set(pattern_context.keys()))
        
        return key_overlap / (key_union + 1e-10)
    
    def add_pattern(self, 
                   pattern_id: str,
                   pattern_graph: nx.Graph,
                   pattern_embedding: Optional[np.ndarray] = None):
        """Add a known pattern for future matching."""
        self.known_patterns[pattern_id] = pattern_graph.copy()
        if pattern_embedding is not None:
            self.pattern_embeddings[pattern_id] = pattern_embedding
            
    def clear_patterns(self):
        """Clear all stored patterns."""
        self.known_patterns.clear()
        self.pattern_embeddings.clear()


# Convenience function
def calculate_wake_mode_gedig(graph: nx.Graph,
                             focal_nodes: Set[str],
                             query_context: Optional[Dict[str, Any]] = None,
                             **kwargs) -> WakeModeResult:
    """
    Calculate geDIG in Wake Mode.
    
    This is a convenience function that creates a calculator
    and performs the calculation.
    """
    calculator = WakeModeGeDIG(**kwargs)
    return calculator.calculate_wake_mode_gedig(graph, focal_nodes, query_context)