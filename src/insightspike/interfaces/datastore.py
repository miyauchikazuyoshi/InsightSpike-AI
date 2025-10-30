"""
DataStore Interfaces
===================

Protocols for data storage implementations.
"""

from typing import Protocol, List, Optional, Dict, Any, runtime_checkable
import numpy as np

from ..core.episode import Episode


@runtime_checkable
class IEpisodeStore(Protocol):
    """
    Interface for episode storage.
    """
    
    def add_episode(self, episode: Episode) -> str:
        """
        Add an episode to the store.
        
        Args:
            episode: Episode to add
            
        Returns:
            ID of the added episode
        """
        ...
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Get an episode by ID.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            Episode if found, None otherwise
        """
        ...
    
    def list_episodes(self, limit: Optional[int] = None) -> List[Episode]:
        """
        List all episodes.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of episodes
        """
        ...
    
    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            True if deleted
        """
        ...
    
    def clear_all(self) -> None:
        """Clear all episodes."""
        ...


@runtime_checkable
class IDataStore(IEpisodeStore, Protocol):
    """
    Full datastore interface with search capabilities.
    """
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[tuple[Episode, float]]:
        """
        Search for similar episodes.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            threshold: Similarity threshold
            
        Returns:
            List of (episode, similarity) tuples
        """
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get store statistics.
        
        Returns:
            Statistics dictionary
        """
        ...
    
    def save_graph(self, graph_data: Any, graph_id: str) -> None:
        """
        Save a graph structure.
        
        Args:
            graph_data: Graph to save
            graph_id: Graph identifier
        """
        ...
    
    def load_graph(self, graph_id: str) -> Optional[Any]:
        """
        Load a graph structure.
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            Graph data if found
        """
        ...