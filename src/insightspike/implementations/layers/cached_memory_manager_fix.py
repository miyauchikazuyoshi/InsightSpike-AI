"""
Fixed CachedMemoryManager with Search Methods

Adds missing search_episodes and _encode_text methods to CachedMemoryManager.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .cached_memory_manager import CachedMemoryManager
from ...core.episode import Episode


class CachedMemoryManagerFixed(CachedMemoryManager):
    """Fixed version with search functionality."""
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        return self.embedder.get_embedding(text)
    
    def search_episodes(
        self, 
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Episode, float]]:
        """
        Search for similar episodes using embeddings.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (episode, similarity) tuples
        """
        # Get all episodes from cache and datastore
        all_episodes = []
        
        # Add cached episodes
        for ep_id, episode in self.cache.items():
            all_episodes.append((ep_id, episode))
        
        # Load episodes from datastore
        try:
            stored_episodes = self.datastore.load_episodes(namespace="episodes")
            for ep_dict in stored_episodes:
                ep_id = ep_dict.get("id")
                if ep_id and ep_id not in self.cache:
                    # Create Episode object from dict
                    episode = Episode(
                        text=ep_dict.get("text", ""),
                        vec=np.array(ep_dict.get("vec", [])),
                        confidence=ep_dict.get("confidence", 0.5),
                        metadata=ep_dict.get("metadata", {})
                    )
                    all_episodes.append((ep_id, episode))
        except Exception as e:
            logger.warning(f"Failed to load episodes from datastore: {e}")
        
        if not all_episodes:
            return []
        
        # Compute similarities
        episode_embeddings = []
        episode_objects = []
        
        for ep_id, episode in all_episodes:
            if hasattr(episode, 'vec') and episode.vec is not None:
                # Ensure proper shape
                vec = episode.vec
                if vec.ndim == 1:
                    vec = vec.reshape(1, -1)
                elif vec.shape[0] != 1:
                    vec = vec[0:1]  # Take first row if multiple
                    
                episode_embeddings.append(vec.flatten())
                episode_objects.append(episode)
        
        if not episode_embeddings:
            return []
        
        # Stack embeddings
        embeddings_matrix = np.vstack(episode_embeddings)
        
        # Ensure query embedding is properly shaped
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.shape[0] != 1:
            query_embedding = query_embedding[0:1]
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(episode_objects):
                episode = episode_objects[idx]
                similarity = float(similarities[idx])
                results.append((episode, similarity))
        
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        cache_episodes = len(self.cache)
        try:
            stored_episodes = len(self.datastore.load_episodes(namespace="episodes"))
        except:
            stored_episodes = 0
            
        total_episodes = cache_episodes + stored_episodes
        
        return {
            "total_episodes": total_episodes,
            "cached_episodes": cache_episodes,
            "stored_episodes": stored_episodes,
            "cache_stats": self.cache_stats,
            "capacity": self.cache_size,
            "utilization": cache_episodes / self.cache_size if self.cache_size > 0 else 0
        }
    
    @property
    def episodes(self):
        """Compatibility property for accessing episodes."""
        # Return a list-like object that provides episode access
        class EpisodeAccessor:
            def __init__(self, manager):
                self.manager = manager
                self._episodes_cache = None
            
            def __len__(self):
                return self.manager.get_memory_stats()["total_episodes"]
            
            def __getitem__(self, index):
                # Load all episodes if not cached
                if self._episodes_cache is None:
                    self._episodes_cache = []
                    
                    # Add from cache
                    for ep in self.manager.cache.values():
                        self._episodes_cache.append(ep)
                    
                    # Add from datastore
                    try:
                        stored = self.manager.datastore.load_episodes(namespace="episodes")
                        for ep_dict in stored:
                            if ep_dict.get("id") not in self.manager.cache:
                                episode = Episode(
                                    text=ep_dict.get("text", ""),
                                    vec=np.array(ep_dict.get("vec", [])),
                                    confidence=ep_dict.get("confidence", 0.5),
                                    metadata=ep_dict.get("metadata", {})
                                )
                                self._episodes_cache.append(episode)
                    except:
                        pass
                
                if 0 <= index < len(self._episodes_cache):
                    return self._episodes_cache[index]
                raise IndexError(f"Episode index {index} out of range")
        
        return EpisodeAccessor(self)


# Monkey patch the original class
import logging
logger = logging.getLogger(__name__)

# Import and replace
import sys
sys.modules['insightspike.implementations.layers.cached_memory_manager'].CachedMemoryManager = CachedMemoryManagerFixed