"""Optimized GraphManager for geDIG wiring."""

from typing import List, Optional, Dict, Any
import networkx as nx
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


class OptimizedGraphManager:
    """Optimized version of GraphManager with caching and smarter geDIG wiring."""
    
    def __init__(self, gedig_evaluator: Optional[GeDIGEvaluator] = None):
        self.graph = nx.Graph()
        self.gedig_evaluator = gedig_evaluator or GeDIGEvaluator()
        self._gedig_cache: Dict[tuple, float] = {}
        self.edge_logs: List[Dict[str, Any]] = []
        self.graph_history: List[nx.Graph] = []  # Compatibility with MazeNavigator
        self.edge_creation_log = []  # Compatibility with original GraphManager
    
    def add_episode_node(self, episode: Episode) -> None:
        """Add episode node to graph."""
        self.graph.add_node(
            episode.episode_id,
            position=episode.position,
            timestamp=episode.timestamp
        )
    
    def wire_edges(self, episodes: List[Episode], strategy: str = 'simple') -> None:
        """Wire episodes with specified strategy."""
        if strategy == 'simple':
            self._wire_simple(episodes)
        elif strategy == 'gedig_optimized':
            self._wire_with_gedig_optimized(episodes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _wire_simple(self, episodes: List[Episode]) -> None:
        """Simple sequential wiring."""
        for i in range(1, len(episodes)):
            self.graph.add_edge(
                episodes[i].episode_id,
                episodes[i-1].episode_id
            )
    
    def _wire_with_gedig_optimized(
        self,
        episodes: List[Episode],
        threshold: float = -0.1,  # More permissive default
        adaptive: bool = True
    ) -> None:
        """
        Optimized geDIG-based wiring with caching and adaptive threshold.
        
        Key optimizations:
        1. No graph copying - calculate incrementally
        2. Cache geDIG values
        3. Adaptive threshold based on observed values
        4. Early stopping when good connection found
        """
        if len(episodes) < 2:
            return
        
        # Sort by timestamp
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)
        
        # Track geDIG values for adaptive threshold
        observed_values = []
        
        for i in range(1, len(sorted_episodes)):
            current = sorted_episodes[i]
            
            # Find best connection with caching
            best_connection = None
            best_gedig = None
            
            # Limit search to recent episodes (exponential backoff)
            search_limit = min(i, 5 + i // 10)  # Max 5-15 episodes back
            
            for j in range(max(0, i - search_limit), i):
                other = sorted_episodes[j]
                
                # Check cache first
                cache_key = (current.episode_id, other.episode_id)
                if cache_key in self._gedig_cache:
                    gedig_value = self._gedig_cache[cache_key]
                else:
                    # Calculate geDIG value WITHOUT copying the graph
                    gedig_value = self._calculate_gedig_incremental(
                        current.episode_id,
                        other.episode_id
                    )
                    self._gedig_cache[cache_key] = gedig_value
                
                observed_values.append(gedig_value)
                
                # Track best connection
                if best_gedig is None or gedig_value < best_gedig:
                    best_gedig = gedig_value
                    best_connection = other.episode_id
                
                # Early stopping if very good connection found
                if adaptive and gedig_value < threshold * 2:
                    break
            
            # Adaptive threshold adjustment
            if adaptive and len(observed_values) >= 10:
                # Use percentile of observed values
                import numpy as np
                adaptive_threshold = np.percentile(observed_values, 20)
                threshold = min(threshold, adaptive_threshold)
            
            # Connect if threshold met
            if best_connection is not None and best_gedig is not None:
                # More permissive: connect if below threshold OR if it's the best we've seen
                if best_gedig <= threshold or (i < 5 and best_gedig <= 0):
                    self.graph.add_edge(current.episode_id, best_connection)
                    self.edge_logs.append({
                        'from': current.episode_id,
                        'to': best_connection,
                        'gedig': best_gedig,
                        'threshold': threshold
                    })
    
    def _calculate_gedig_incremental(
        self,
        node1: int,
        node2: int
    ) -> float:
        """
        Calculate geDIG value for adding an edge WITHOUT copying the graph.
        
        This is an approximation but much faster:
        - Count current edges around both nodes
        - Estimate structural change from adding the edge
        - Return approximate geDIG value
        """
        # Get current node degrees
        deg1 = self.graph.degree(node1) if self.graph.has_node(node1) else 0
        deg2 = self.graph.degree(node2) if self.graph.has_node(node2) else 0
        
        # Check if edge already exists
        if self.graph.has_edge(node1, node2):
            return 0.0  # No change
        
        # Estimate structural improvement
        # Adding edge between low-degree nodes is good (negative value)
        # Adding edge between high-degree nodes is less good
        n = self.graph.number_of_nodes()
        if n == 0:
            return -1.0
        
        # Simple heuristic: prefer connecting isolated or low-degree nodes
        avg_degree = (deg1 + deg2) / 2.0
        max_degree = max(self.graph.degree(n) for n in self.graph.nodes()) if self.graph.nodes() else 1
        
        # More negative (better) for low-degree nodes
        structural_score = -(1.0 - avg_degree / (max_degree + 1))
        
        # Penalty for creating very dense regions
        if avg_degree > n * 0.3:  # If nodes are already well-connected
            structural_score *= 0.5
        
        return structural_score * 0.3  # Scale to typical geDIG range
    
    def get_graph_snapshot(self) -> nx.Graph:
        """Get current graph snapshot (compatibility method)."""
        return self.graph.copy()
    
    def save_snapshot(self) -> None:
        """Save current graph snapshot (compatibility method)."""
        self.graph_history.append(self.get_graph_snapshot())
    
    def get_connected_episodes(self, episode_id: int) -> List[int]:
        """Get connected episode IDs."""
        if episode_id not in self.graph:
            return []
        return list(self.graph.neighbors(episode_id))
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'num_components': nx.number_connected_components(self.graph),
            'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        
        if stats['is_connected'] and self.graph.number_of_nodes() > 0:
            stats['diameter'] = nx.diameter(self.graph)
            stats['radius'] = nx.radius(self.graph)
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph)
        
        if self.graph.number_of_nodes() > 0:
            degrees = [d for n, d in self.graph.degree()]
            stats['average_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        return stats