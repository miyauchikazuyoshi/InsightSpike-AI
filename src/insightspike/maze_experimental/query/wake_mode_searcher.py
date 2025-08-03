"""
Wake mode specific searcher that integrates sphere search with existing system.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .sphere_search import SphereSearch, SimpleSphereSearch, NeighborNode


class WakeModeSearcher:
    """
    Searcher specifically for wake mode processing.
    Bridges sphere search with existing episode search interface.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize with reference to memory manager.
        
        Args:
            memory_manager: Layer2 memory manager instance
        """
        self.memory_manager = memory_manager
        self.sphere_searcher = None
        self._init_searcher()
    
    def _init_searcher(self):
        """Initialize sphere searcher with current episodes."""
        if not self.memory_manager.episodes:
            return
        
        # Build node vectors from episodes
        node_vectors = {
            str(i): episode.vec
            for i, episode in enumerate(self.memory_manager.episodes)
            if episode.vec is not None
        }
        
        # Use simple searcher for small datasets, FAISS for large
        if len(node_vectors) < 1000:
            self.sphere_searcher = SimpleSphereSearch(node_vectors)
        else:
            try:
                self.sphere_searcher = SphereSearch(node_vectors)
            except:
                # Fallback if FAISS not available
                self.sphere_searcher = SimpleSphereSearch(node_vectors)
    
    def search_wake_mode(
        self,
        query_embedding: np.ndarray,
        radius: float = 0.8,
        max_neighbors: int = 20,
        use_donut: bool = False,
        inner_radius: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Search episodes using wake mode sphere search.
        
        Args:
            query_embedding: Query vector
            radius: Search radius (outer radius if using donut)
            max_neighbors: Maximum neighbors to return
            use_donut: Whether to use donut search
            inner_radius: Inner radius for donut search
            
        Returns:
            List of episode dictionaries with distance info
        """
        if self.sphere_searcher is None:
            self._init_searcher()
            if self.sphere_searcher is None:
                return []
        
        # Perform search
        if use_donut:
            neighbors = self.sphere_searcher.search_donut(
                query_embedding, inner_radius, radius, max_neighbors
            )
        else:
            neighbors = self.sphere_searcher.search_sphere(
                query_embedding, radius, max_neighbors
            )
        
        # Convert to episode format
        results = []
        for neighbor in neighbors:
            idx = int(neighbor.node_id)
            if 0 <= idx < len(self.memory_manager.episodes):
                episode = self.memory_manager.episodes[idx]
                
                results.append({
                    'episode': episode,
                    'index': idx,
                    'distance': neighbor.distance,
                    'relative_position': neighbor.relative_position,
                    'score': 1.0 - (neighbor.distance / radius),  # Simple scoring
                    'text': episode.text,
                    'c_value': getattr(episode, 'c', 0.5)
                })
        
        return results
    
    def update_index(self):
        """Update searcher when episodes change."""
        self._init_searcher()


def integrate_wake_search(memory_manager, config=None):
    """
    Factory function to add wake mode search to existing memory manager.
    
    Args:
        memory_manager: Existing Layer2 memory manager
        config: Optional configuration
        
    Returns:
        Enhanced memory manager with wake mode search
    """
    # Add wake searcher as attribute
    memory_manager._wake_searcher = WakeModeSearcher(memory_manager)
    
    # Add wake search method
    def search_episodes_wake_mode(
        self,
        query: str,
        k: int = 5,
        radius: float = 0.8,
        use_donut: bool = False,
        inner_radius: float = 0.2
    ):
        """Wake mode search using sphere/donut search."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []
        
        # Use wake searcher
        return self._wake_searcher.search_wake_mode(
            query_embedding,
            radius=radius,
            max_neighbors=k,
            use_donut=use_donut,
            inner_radius=inner_radius
        )
    
    # Attach method
    memory_manager.search_episodes_wake_mode = search_episodes_wake_mode.__get__(
        memory_manager, memory_manager.__class__
    )
    
    return memory_manager