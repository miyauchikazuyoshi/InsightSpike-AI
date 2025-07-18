"""
DataStore Adapters
=================

Adapters to make existing code work with DataStore interface.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.base.datastore import DataStore
from ...core.episode import Episode

logger = logging.getLogger(__name__)


class L2MemoryAdapter:
    """Adapter to make L2MemoryManager use DataStore"""
    
    def __init__(self, datastore: DataStore):
        """Initialize adapter
        
        Args:
            datastore: DataStore instance to use
        """
        self.datastore = datastore
        self.namespace = "l2_memory"
    
    def save_episodes(self, episodes: List[Episode]) -> bool:
        """Save episodes using DataStore
        
        Args:
            episodes: List of Episode objects
            
        Returns:
            Success status
        """
        # Convert Episodes to dicts
        episode_dicts = []
        for ep in episodes:
            episode_dicts.append({
                "text": ep.text,
                "vec": ep.vec,
                "c": ep.c,
                "timestamp": ep.timestamp,
                "metadata": ep.metadata
            })
        
        # Save episodes
        success = self.datastore.save_episodes(episode_dicts, namespace=self.namespace)
        
        if success:
            # Also save vectors for search
            vectors = np.array([ep.vec for ep in episodes])
            metadata = [{"idx": i, "text": ep.text} for i, ep in enumerate(episodes)]
            self.datastore.save_vectors(vectors, metadata, namespace=self.namespace)
        
        return success
    
    def load_episodes(self) -> List[Episode]:
        """Load episodes from DataStore
        
        Returns:
            List of Episode objects
        """
        episode_dicts = self.datastore.load_episodes(namespace=self.namespace)
        
        episodes = []
        for ep_dict in episode_dicts:
            episode = Episode(
                text=ep_dict["text"],
                vec=np.array(ep_dict["vec"], dtype=np.float32),
                c=ep_dict.get("c", 0.5),
                timestamp=ep_dict.get("timestamp"),
                metadata=ep_dict.get("metadata", {})
            )
            episodes.append(episode)
        
        return episodes
    
    def search_similar(self, query_vec: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        """Search for similar episodes
        
        Args:
            query_vec: Query vector
            k: Number of results
            
        Returns:
            Tuple of (indices, distances)
        """
        return self.datastore.search_vectors(query_vec, k=k, namespace=self.namespace)
    
    def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Save L2 metadata
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Success status
        """
        return self.datastore.save_metadata(metadata, key="l2_metadata", namespace=self.namespace)
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load L2 metadata
        
        Returns:
            Metadata dict or None
        """
        return self.datastore.load_metadata(key="l2_metadata", namespace=self.namespace)
    
    def clear(self) -> bool:
        """Clear all L2 data
        
        Returns:
            Success status
        """
        return self.datastore.clear_namespace(self.namespace)


class L3GraphAdapter:
    """Adapter to make L3GraphReasoner use DataStore"""
    
    def __init__(self, datastore: DataStore):
        """Initialize adapter
        
        Args:
            datastore: DataStore instance to use
        """
        self.datastore = datastore
        self.namespace = "l3_graphs"
    
    def save_graph(self, graph: Any, graph_id: str) -> bool:
        """Save graph using DataStore
        
        Args:
            graph: Graph object (PyTorch Geometric Data or NetworkX)
            graph_id: Unique identifier for the graph
            
        Returns:
            Success status
        """
        return self.datastore.save_graph(graph, graph_id=graph_id, namespace=self.namespace)
    
    def load_graph(self, graph_id: str) -> Optional[Any]:
        """Load graph from DataStore
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            Graph object or None
        """
        return self.datastore.load_graph(graph_id=graph_id, namespace=self.namespace)
    
    def save_analysis_history(self, history: List[Dict[str, Any]]) -> bool:
        """Save analysis history
        
        Args:
            history: List of analysis results
            
        Returns:
            Success status
        """
        return self.datastore.save_metadata(
            {"history": history}, 
            key="analysis_history", 
            namespace=self.namespace
        )
    
    def load_analysis_history(self) -> List[Dict[str, Any]]:
        """Load analysis history
        
        Returns:
            List of analysis results
        """
        metadata = self.datastore.load_metadata(key="analysis_history", namespace=self.namespace)
        return metadata.get("history", []) if metadata else []
    
    def list_graphs(self) -> List[str]:
        """List available graph IDs
        
        Returns:
            List of graph IDs
        """
        return self.datastore.list_keys(self.namespace)
    
    def delete_graph(self, graph_id: str) -> bool:
        """Delete a graph
        
        Args:
            graph_id: Graph to delete
            
        Returns:
            Success status
        """
        return self.datastore.delete(graph_id, namespace=self.namespace)