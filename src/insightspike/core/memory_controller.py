"""
Memory Controller
================

Handles all memory-related operations.
Separated from MainAgent to follow Single Responsibility Principle.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from ..interfaces import IMemoryManager, IDataStore, IEmbedder
from ..core.episode import Episode
from ..implementations.graph.pyg_graph_builder import PyGGraphBuilder
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class MemoryController:
    """
    Manages memory operations and episode lifecycle.
    
    Responsibilities:
    - Add and retrieve episodes
    - Manage episode lifecycle
    - Build and maintain knowledge graphs
    - Search memory efficiently
    """
    
    def __init__(
        self,
        memory_manager: IMemoryManager,
        datastore: IDataStore,
        embedder: IEmbedder,
        graph_builder: PyGGraphBuilder
    ):
        """
        Initialize memory controller with dependencies.
        
        Args:
            memory_manager: Memory management component
            datastore: Persistent storage
            embedder: Text embedding component
            graph_builder: Graph construction component
        """
        self.memory_manager = memory_manager
        self.datastore = datastore
        self.embedder = embedder
        self.graph_builder = graph_builder
        
        self._episode_cache: List[Episode] = []
        self._current_graph: Optional[Data] = None
    
    def add_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Episode:
        """
        Add new knowledge to memory.
        
        Args:
            text: Knowledge text
            metadata: Optional metadata
            
        Returns:
            Created episode
        """
        try:
            # Create embedding
            embedding = self.embedder.encode(text)
            
            # Create episode
            episode = Episode(
                text=text,
                embedding=embedding,
                metadata=metadata or {},
                c=0.5  # Initial confidence
            )
            
            # Add to memory manager
            self.memory_manager.add_episode(episode)
            
            # Add to datastore
            self.datastore.add_episode(episode.to_dict())
            
            # Update cache
            self._episode_cache.append(episode)
            
            # Rebuild graph if needed
            if len(self._episode_cache) % 10 == 0:
                self._rebuild_graph()
            
            logger.debug(f"Added knowledge: {text[:50]}...")
            return episode
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            raise
    
    def retrieve_relevant(
        self,
        query: str,
        k: int = 10,
        use_graph: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant episodes for a query.
        
        Args:
            query: Query text
            k: Number of results
            use_graph: Whether to use graph-based search
            
        Returns:
            List of relevant episodes with metadata
        """
        try:
            # Encode query
            query_embedding = self.embedder.encode(query)
            
            if use_graph and self._current_graph is not None:
                # Graph-based search
                return self._graph_search(query_embedding, k)
            else:
                # Vector similarity search
                return self._vector_search(query_embedding, k)
                
        except Exception as e:
            logger.error(f"Failed to retrieve relevant episodes: {e}")
            return []
    
    def update_episode_reward(self, episode_id: str, reward: float) -> None:
        """
        Update episode's c-value based on reasoning result.
        
        Args:
            episode_id: Episode identifier
            reward: Reward value to apply
        """
        try:
            # Find episode in cache
            episode = next((e for e in self._episode_cache if e.id == episode_id), None)
            
            if episode:
                # Update c-value
                old_c = episode.c
                episode.c = min(1.0, max(0.0, old_c + reward))
                
                # Update in datastore
                self.datastore.update_episode(episode_id, episode.to_dict())
                
                logger.debug(f"Updated episode {episode_id}: c={old_c:.3f} -> {episode.c:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to update episode reward: {e}")
    
    def get_memory_state(self) -> Dict[str, Any]:
        """
        Get current memory state and statistics.
        
        Returns:
            Memory state information
        """
        stats = self.memory_manager.get_statistics()
        
        return {
            "episode_count": len(self._episode_cache),
            "graph_nodes": self._current_graph.num_nodes if self._current_graph else 0,
            "graph_edges": self._current_graph.num_edges if self._current_graph else 0,
            "memory_stats": stats,
            "average_c_value": np.mean([e.c for e in self._episode_cache]) if self._episode_cache else 0.0
        }
    
    def clear_memory(self, preserve_high_value: bool = False) -> None:
        """
        Clear memory with optional preservation of high-value episodes.
        
        Args:
            preserve_high_value: Whether to keep high c-value episodes
        """
        if preserve_high_value:
            # Keep episodes with c > 0.8
            high_value = [e for e in self._episode_cache if e.c > 0.8]
            self._episode_cache = high_value
            
            # Clear and re-add to memory manager
            self.memory_manager.clear_memory()
            for episode in high_value:
                self.memory_manager.add_episode(episode)
        else:
            # Clear everything
            self._episode_cache.clear()
            self.memory_manager.clear_memory()
            self._current_graph = None
        
        logger.info(f"Memory cleared. Remaining episodes: {len(self._episode_cache)}")
    
    def _rebuild_graph(self) -> None:
        """Rebuild the knowledge graph from current episodes."""
        if not self._episode_cache:
            self._current_graph = None
            return
        
        try:
            self._current_graph = self.graph_builder.build_graph(self._episode_cache)
            logger.debug(f"Rebuilt graph: {self._current_graph.num_nodes} nodes, {self._current_graph.num_edges} edges")
        except Exception as e:
            logger.error(f"Failed to rebuild graph: {e}")
    
    def _vector_search(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Simple vector similarity search."""
        results = self.memory_manager.search_similar(query_embedding, k)
        
        return [
            {
                "episode": episode,
                "similarity": sim,
                "text": episode.text,
                "embedding": episode.embedding,
                "metadata": episode.metadata
            }
            for episode, sim in results
        ]
    
    def _graph_search(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Graph-based multi-hop search."""
        # First find initial nodes via vector search
        initial_results = self._vector_search(query_embedding, k // 2)
        
        if not self._current_graph:
            return initial_results
        
        # TODO: Implement multi-hop graph traversal
        # For now, return initial results
        return initial_results
    
    def load_from_datastore(self) -> None:
        """Load episodes from persistent storage."""
        try:
            # Get all episodes from datastore
            episode_dicts = self.datastore.list_episodes()
            
            # Convert to Episode objects
            self._episode_cache = []
            for episode_dict in episode_dicts:
                episode = Episode.from_dict(episode_dict)
                self._episode_cache.append(episode)
                self.memory_manager.add_episode(episode)
            
            # Rebuild graph
            if self._episode_cache:
                self._rebuild_graph()
            
            logger.info(f"Loaded {len(self._episode_cache)} episodes from datastore")
            
        except Exception as e:
            logger.error(f"Failed to load from datastore: {e}")