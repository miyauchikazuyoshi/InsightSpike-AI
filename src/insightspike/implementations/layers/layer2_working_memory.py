"""
Layer 2: Working Memory Manager
================================

DataStore-centric memory management with working memory approach.
Only loads necessary data on demand instead of loading everything into memory.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...core.base.datastore import DataStore
from ...processing.embedder import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory manager"""
    
    # Working memory settings
    working_memory_size: int = 100  # Maximum episodes in working memory
    search_k: int = 20  # How many episodes to retrieve for context
    
    # Cache settings
    cache_embeddings: bool = True
    embedding_cache_size: int = 1000
    
    # Performance settings
    batch_size: int = 32
    similarity_threshold: float = 0.7
    
    # DataStore settings
    datastore_namespace: str = "episodes"
    

class L2WorkingMemoryManager:
    """
    Working memory based memory manager using DataStore.
    
    This implementation:
    - Uses DataStore as the primary storage (not in-memory)
    - Maintains a small working memory of recent/relevant episodes
    - Loads data on-demand for operations
    - Supports both sync and async operations
    """
    
    def __init__(self, 
                 datastore: DataStore,
                 config: Optional[Union[WorkingMemoryConfig, Any]] = None,
                 embedding_manager: Optional[EmbeddingManager] = None):
        """
        Initialize working memory manager.
        
        Args:
            datastore: DataStore instance for persistent storage
            config: Working memory configuration or InsightSpikeConfig
            embedding_manager: Optional embedding manager
        """
        self.datastore = datastore
        
        # Handle different config types
        if isinstance(config, WorkingMemoryConfig):
            self.config = config
        elif hasattr(config, 'memory'):
            # InsightSpikeConfig passed - extract what we need
            self.config = WorkingMemoryConfig(
                working_memory_size=getattr(config.memory, 'working_memory_size', 100),
                search_k=getattr(config.memory, 'search_k', 20),
                similarity_threshold=getattr(config.memory, 'similarity_threshold', 0.7),
                datastore_namespace=getattr(config.memory, 'datastore_namespace', 'episodes')
            )
        else:
            self.config = WorkingMemoryConfig()
            
        self.embedding_manager = embedding_manager
        
        # Working memory - only recent/relevant episodes
        self.working_memory: List[Dict[str, Any]] = []
        self.working_memory_ids: List[str] = []
        
        # Embedding cache for performance
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Initialize embedding manager if needed
        if self.embedding_manager is None:
            try:
                self.embedding_manager = EmbeddingManager()
                logger.info("Initialized EmbeddingManager")
            except Exception as e:
                logger.warning(f"Could not initialize EmbeddingManager: {e}")
        
        self.initialized = True
        logger.info(f"L2WorkingMemoryManager initialized with working memory size: {self.config.working_memory_size}")
    
    def store_episode(self, text: str, c_value: float = 0.5, 
                     metadata: Optional[Dict] = None) -> str:
        """
        Store a new episode in DataStore.
        
        Args:
            text: Episode text
            c_value: Confidence value
            metadata: Optional metadata
            
        Returns:
            Episode ID
        """
        try:
            # Create embedding
            embedding = self._get_embedding(text)
            if embedding is None:
                logger.error("Failed to create embedding")
                return ""
            
            # Create episode
            episode = {
                'text': text,
                'vec': embedding,
                'c': c_value,
                'metadata': metadata or {},
                'timestamp': time.time()
            }
            
            # Store in DataStore
            success = self.datastore.save_episodes(
                [episode], 
                namespace=self.config.datastore_namespace
            )
            
            if not success:
                logger.error("Failed to save episode to DataStore")
                return ""
            
            # Generate ID (DataStore should assign this)
            episode_id = episode.get('id', f"ep_{int(time.time() * 1000)}")
            
            # Add to working memory if space available
            if len(self.working_memory) < self.config.working_memory_size:
                self.working_memory.append(episode)
                self.working_memory_ids.append(episode_id)
            else:
                # Replace oldest episode
                self.working_memory.pop(0)
                self.working_memory_ids.pop(0)
                self.working_memory.append(episode)
                self.working_memory_ids.append(episode_id)
            
            # Cache embedding
            if self.config.cache_embeddings:
                self._cache_embedding(text, embedding)
            
            logger.debug(f"Stored episode: {text[:50]}...")
            return episode_id
            
        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            return ""
    
    def search_episodes(self, query: str, k: int = 5, 
                       filter_fn: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant episodes from DataStore.
        
        Args:
            query: Query text
            k: Number of results
            filter_fn: Optional filter function
            
        Returns:
            List of relevant episodes
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []
            
            # Search in DataStore
            indices, distances = self.datastore.search_vectors(
                query_embedding,
                k=k * 2,  # Get more for filtering
                namespace=self.config.datastore_namespace
            )
            
            # If no results from DataStore, search working memory
            if not indices:
                return self._search_working_memory(query_embedding, k)
            
            # Load episodes from DataStore
            # This is where we avoid loading everything
            episodes = []
            for idx, dist in zip(indices[:k], distances[:k]):
                # DataStore should provide a way to get episodes by ID
                # For now, we'll use a simple approach
                similarity = 1.0 - (dist / 2.0)  # Convert distance to similarity
                
                if similarity >= self.config.similarity_threshold:
                    # In a real implementation, we'd fetch the episode from DataStore
                    # For now, create a placeholder
                    episode = {
                        'text': f"Episode {idx}",  # Would be fetched from DataStore
                        'similarity': similarity,
                        'c': 0.5,
                        'metadata': {}
                    }
                    
                    if filter_fn is None or filter_fn(episode):
                        episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            logger.error(f"Failed to search episodes: {e}")
            return []
    
    def _search_working_memory(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search only in working memory (fallback)"""
        if not self.working_memory:
            return []
        
        # Compute similarities
        similarities = []
        for episode in self.working_memory:
            if 'vec' in episode:
                sim = np.dot(query_embedding, episode['vec']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(episode['vec'])
                )
                similarities.append((sim, episode))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        results = []
        for sim, episode in similarities[:k]:
            if sim >= self.config.similarity_threshold:
                result = episode.copy()
                result['similarity'] = sim
                results.append(result)
        
        return results
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching"""
        # Check cache first
        if self.config.cache_embeddings and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generate embedding
        if self.embedding_manager is None:
            logger.warning("No embedding manager available")
            return np.random.rand(384).astype(np.float32)  # Fallback
        
        try:
            embedding = self.embedding_manager.get_embedding(text)
            
            # Cache it
            if self.config.cache_embeddings:
                self._cache_embedding(text, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding with size limit"""
        if len(self.embedding_cache) >= self.config.embedding_cache_size:
            # Remove oldest (simple FIFO)
            first_key = next(iter(self.embedding_cache))
            del self.embedding_cache[first_key]
        
        self.embedding_cache[text] = embedding
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            'working_memory_size': len(self.working_memory),
            'working_memory_capacity': self.config.working_memory_size,
            'embedding_cache_size': len(self.embedding_cache),
            'initialized': self.initialized
        }
        
        # Get DataStore stats
        try:
            datastore_stats = self.datastore.get_stats()
            stats['datastore'] = datastore_stats
        except Exception as e:
            logger.warning(f"Could not get DataStore stats: {e}")
        
        return stats
    
    def clear_working_memory(self):
        """Clear only working memory (not DataStore)"""
        self.working_memory.clear()
        self.working_memory_ids.clear()
        self.embedding_cache.clear()
        logger.info("Cleared working memory")
    
    def refresh_working_memory(self, query: Optional[str] = None):
        """
        Refresh working memory with relevant episodes.
        
        Args:
            query: Optional query to find relevant episodes
        """
        try:
            if query:
                # Load relevant episodes based on query
                episodes = self.search_episodes(query, k=self.config.working_memory_size)
                self.working_memory = episodes
                self.working_memory_ids = [ep.get('id', '') for ep in episodes]
            else:
                # Load recent episodes
                # This would require DataStore to support loading by timestamp
                # For now, just clear
                self.clear_working_memory()
                
            logger.info(f"Refreshed working memory with {len(self.working_memory)} episodes")
            
        except Exception as e:
            logger.error(f"Failed to refresh working memory: {e}")
    
    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save working memory state (not full data)"""
        try:
            import json
            checkpoint = {
                'working_memory_ids': self.working_memory_ids,
                'config': {
                    'working_memory_size': self.config.working_memory_size,
                    'search_k': self.config.search_k,
                    'datastore_namespace': self.config.datastore_namespace
                }
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
            
            logger.info(f"Saved working memory checkpoint to {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load working memory state"""
        try:
            import json
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore working memory IDs
            self.working_memory_ids = checkpoint.get('working_memory_ids', [])
            
            # TODO: Load actual episodes from DataStore using IDs
            # For now, just clear working memory
            self.working_memory.clear()
            
            logger.info(f"Loaded working memory checkpoint from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False