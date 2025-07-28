"""
Memory Interfaces
================

Protocols for memory management components.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
import numpy as np

from ..core.episode import Episode
from torch_geometric.data import Data


@runtime_checkable
class IMemoryManager(Protocol):
    """
    Interface for memory management.
    """
    
    def add_episode(self, episode: Episode) -> None:
        """
        Add an episode to memory.
        
        Args:
            episode: Episode to add
        """
        ...
    
    def get_recent_episodes(self, k: int = 10) -> List[Episode]:
        """
        Get recent episodes.
        
        Args:
            k: Number of episodes
            
        Returns:
            List of recent episodes
        """
        ...
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[tuple[Episode, float]]:
        """
        Search for similar episodes.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            List of (episode, similarity) tuples
        """
        ...
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        ...


@runtime_checkable
class IMemorySearch(Protocol):
    """
    Interface for advanced memory search with graph support.
    """
    
    def search_with_graph(
        self,
        query_embedding: np.ndarray,
        episodes: List[Episode],
        graph_data: Optional[Data] = None,
        k: int = 10,
        enable_multi_hop: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search memory using graph-based traversal.
        
        Args:
            query_embedding: Query vector
            episodes: Available episodes
            graph_data: PyG graph structure
            k: Number of results
            enable_multi_hop: Whether to use multi-hop search
            
        Returns:
            List of search results with metadata
        """
        ...
    
    def extract_subgraph(
        self,
        center_nodes: List[int],
        graph_data: Data,
        radius: int = 1
    ) -> Dict[str, Any]:
        """
        Extract a subgraph around specified nodes.
        
        Args:
            center_nodes: Starting nodes
            graph_data: Full graph
            radius: Hop radius
            
        Returns:
            Subgraph information
        """
        ...