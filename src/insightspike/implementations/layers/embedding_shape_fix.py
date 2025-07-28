"""
Embedding Shape Fix - Ensure embeddings are 1D arrays (384,) not 2D (1, 384)
"""

import numpy as np
from .cached_memory_manager import CachedMemoryManager
from ...core.episode import Episode


# Patch CachedMemoryManager to fix embedding shape
original_add_episode = CachedMemoryManager.add_episode

def fixed_add_episode(self, text: str, c_value: float = 0.5, metadata=None):
    """Fixed add_episode that ensures embeddings are 1D."""
    # Create embedding
    embedding = self.embedder.get_embedding(text)
    
    # Fix shape if needed
    if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
        embedding = embedding.flatten()
    
    # Create episode with fixed embedding
    episode = Episode(
        text=text,
        vec=embedding,  # Use 1D array
        confidence=c_value,
        metadata=metadata or {}
    )
    
    # Generate episode ID
    import uuid
    episode_id = str(uuid.uuid4())
    
    # Store in cache
    self._add_to_cache(episode_id, episode)
    
    # Save to DataStore
    try:
        episodes_to_save = [{
            "id": episode_id,
            "text": text,
            "vec": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "confidence": c_value,
            "c_value": c_value,
            "timestamp": episode.timestamp,
            "metadata": metadata or {}
        }]
        
        # Append to existing episodes
        existing = self.datastore.load_episodes(namespace="episodes")
        all_episodes = existing + episodes_to_save
        self.datastore.save_episodes(all_episodes, namespace="episodes")
        
        self.cache_stats['misses'] += 1
        self._check_memory()
        
        # Return index for compatibility
        return len(existing)
        
    except Exception as e:
        logger.error(f"Failed to save episode: {e}")
        return -1


# Apply the patch
CachedMemoryManager.add_episode = fixed_add_episode


# Also patch Episode creation in search_episodes
original_search = CachedMemoryManager.search_episodes

def fixed_search_episodes(self, query_embedding, top_k=10):
    """Fixed search that handles embedding shapes correctly."""
    # Ensure query embedding is 1D
    if isinstance(query_embedding, np.ndarray) and query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()
    
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
                # Fix vec shape when loading
                vec = np.array(ep_dict.get("vec", []))
                if vec.ndim > 1:
                    vec = vec.flatten()
                
                # Create Episode object from dict
                episode = Episode(
                    text=ep_dict.get("text", ""),
                    vec=vec,
                    confidence=ep_dict.get("confidence", 0.5),
                    metadata=ep_dict.get("metadata", {})
                )
                all_episodes.append((ep_id, episode))
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load episodes from datastore: {e}")
    
    if not all_episodes:
        return []
    
    # Compute similarities
    episode_embeddings = []
    episode_objects = []
    
    for ep_id, episode in all_episodes:
        if hasattr(episode, 'vec') and episode.vec is not None:
            vec = episode.vec
            # Ensure 1D
            if vec.ndim > 1:
                vec = vec.flatten()
            episode_embeddings.append(vec)
            episode_objects.append(episode)
    
    if not episode_embeddings:
        return []
    
    # Stack embeddings
    embeddings_matrix = np.vstack(episode_embeddings)
    
    # Compute cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if idx < len(episode_objects) and similarities[idx] > 0:
            results.append((episode_objects[idx], float(similarities[idx])))
    
    return results


CachedMemoryManager.search_episodes = fixed_search_episodes


def apply_embedding_shape_fix():
    """Apply the embedding shape fix."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Applied embedding shape fix to CachedMemoryManager")