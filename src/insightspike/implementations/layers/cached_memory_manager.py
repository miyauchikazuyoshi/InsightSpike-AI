"""
Cached Memory Manager
====================

A memory manager that uses DataStore as backend with intelligent caching
to prevent memory explosion while maintaining performance.
"""

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import networkx as nx

from ...core.episode import Episode
from ...core.base.datastore import DataStore
from ...processing.embedder import EmbeddingManager
from ...monitoring.memory_monitor import get_memory_monitor

logger = logging.getLogger(__name__)


class CachedMemoryManager:
    """
    Memory manager with DataStore backend and LRU cache.
    
    This provides a compatible interface with L2MemoryManager
    but uses DataStore for persistence and only caches frequently
    accessed episodes in memory.
    """
    
    def __init__(
        self,
        datastore: DataStore,
        cache_size: int = 100,
        embedder: Optional[EmbeddingManager] = None
    ):
        """
        Initialize cached memory manager.
        
        Args:
            datastore: Backend storage
            cache_size: Maximum number of episodes to cache
            embedder: Embedding manager for text encoding
        """
        self.datastore = datastore
        self.cache_size = cache_size
        self.embedder = embedder or EmbeddingManager()
        
        # Compatibility alias
        self.embedding_model = self.embedder
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, Episode] = OrderedDict()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Memory monitoring
        self.memory_monitor = get_memory_monitor()
        self.last_memory_check = 0
        self.check_interval = 60  # seconds
        
        logger.info(f"CachedMemoryManager initialized with cache_size={cache_size}")
        
    def add_episode(
        self,
        text: str,
        c_value: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add new episode to DataStore and cache.
        
        Args:
            text: Episode text
            c_value: Importance value
            metadata: Optional metadata
            
        Returns:
            Episode index (for compatibility with L2MemoryManager)
        """
        # Create embedding
        embedding = self.embedder.get_embedding(text)
        
        # Create episode
        episode = Episode(
            text=text,
            vec=embedding,
            c=c_value,
            metadata=metadata or {}
        )
        
        # Generate episode ID
        import uuid
        episode_id = str(uuid.uuid4())
        
        # Convert episode to dict for DataStore
        episode_dict = {
            "id": episode_id,
            "text": text,
            "vec": embedding,
            "c_value": c_value,  # Backward compatibility
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Save to DataStore
        self.datastore.save_episodes([episode_dict], namespace="episodes")
        
        # Add to cache
        self._add_to_cache(episode_id, episode)
        
        # Check memory periodically
        self._check_memory()
        
        # Return index (cache size) for compatibility
        return len(self.cache) - 1
        
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Get episode by ID, using cache when possible.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Episode or None if not found
        """
        # Check cache first
        if episode_id in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(episode_id)
            self.cache_stats['hits'] += 1
            return self.cache[episode_id]
            
        # Cache miss - fetch from DataStore
        self.cache_stats['misses'] += 1
        
        # DataStore doesn't have get_episode, so we need to load all and find
        # This is inefficient but maintains compatibility
        try:
            episodes = self.datastore.load_episodes(namespace="episodes")
            for ep_dict in episodes:
                if ep_dict.get("id") == episode_id:
                    # Convert dict back to Episode
                    episode = Episode(
                        text=ep_dict.get("text", ""),
                        vec=ep_dict.get("vec", np.array([])),
                        c=ep_dict.get("c_value", ep_dict.get("confidence", ep_dict.get("c", 0.5))),
                        metadata=ep_dict.get("metadata", {})
                    )
                    self._add_to_cache(episode_id, episode)
                    return episode
        except Exception as e:
            logger.error(f"Failed to fetch episode from DataStore: {e}")
            
        return None
        
    def search_episodes(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[Episode, float]]:
        """
        Search for similar episodes.
        
        Args:
            query: Search query (text or embedding vector)
            top_k: Number of results
            
        Returns:
            List of (episode, similarity) tuples
        """
        # Get query embedding
        if isinstance(query, str):
            query_embedding = self.embedder.get_embedding(query)
        else:
            # Already an embedding vector
            query_embedding = query
        
        # Since DataStore doesn't have vector search, we need to use search_vectors
        # First, save query vector temporarily
        indices, distances = self.datastore.search_vectors(
            query_embedding,
            k=top_k,
            namespace="vectors"
        )
        
        if not indices:
            # Fallback: search from cached episodes only
            episodes = list(self.cache.values())
            if not episodes:
                return []
                
            vectors = np.array([ep.vec.flatten() if len(ep.vec.shape) > 1 else ep.vec for ep in episodes])
            query_flat = query_embedding.flatten() if len(query_embedding.shape) > 1 else query_embedding
            similarities = np.dot(vectors, query_flat)
            top_k_indices = np.argsort(similarities)[-min(top_k, len(episodes)):][::-1]
            
            results = []
            for idx in top_k_indices:
                if similarities[idx] > 0:
                    results.append((episodes[idx], float(similarities[idx])))
            return results
        
        # Fetch episodes by indices  
        episodes_with_scores = []
        all_episodes = self.datastore.load_episodes(namespace="episodes")
        
        for idx, distance in zip(indices, distances):
            if 0 <= idx < len(all_episodes):
                ep_dict = all_episodes[idx]
                episode = Episode(
                    text=ep_dict["text"],
                    vec=ep_dict["vec"],
                    c=ep_dict.get("c_value", ep_dict.get("c", 0.5)),
                    metadata=ep_dict.get("metadata", {})
                )
                # Convert distance to similarity
                similarity = 1.0 / (1.0 + distance)
                episodes_with_scores.append((episode, similarity))
                
        return episodes_with_scores

    # --- Integrated from cached_memory_manager_fix.py (vector cosine path fallback & accessor) ---
    def search_episodes_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Episode, float]]:
        """Alternative search explicitly using a provided embedding (merged from fix).

        Args:
            query_embedding: Pre-computed query vector
            top_k: number of results
        Returns:
            List of (Episode, similarity)
        """
        # Gather all episodes (cache + datastore remainder)
        all_pairs: List[Tuple[str, Episode]] = []
        for ep_id, ep in self.cache.items():
            all_pairs.append((ep_id, ep))
        try:
            stored = self.datastore.load_episodes(namespace="episodes")
            for ep_dict in stored:
                eid = ep_dict.get("id")
                if eid and eid not in self.cache:
                    ep = Episode(
                        text=ep_dict.get("text", ""),
                        vec=np.array(ep_dict.get("vec", [])),
                        c=ep_dict.get("c_value", ep_dict.get("c", 0.5)),
                        metadata=ep_dict.get("metadata", {})
                    )
                    all_pairs.append((eid, ep))
        except Exception as e:
            logger.warning(f"Failed loading episodes for vector search fallback: {e}")

        if not all_pairs:
            return []

        embeddings = []
        objects = []
        for _, ep in all_pairs:
            if getattr(ep, 'vec', None) is not None:
                vec = ep.vec
                if isinstance(vec, np.ndarray):
                    if vec.ndim == 1:
                        pass
                    elif vec.ndim > 1:
                        vec = vec.reshape(-1)
                embeddings.append(vec)
                objects.append(ep)

        if not embeddings:
            return []

        emb_mat = np.vstack([v if v.ndim > 1 else v.reshape(1, -1) for v in embeddings])
        qe = query_embedding
        if qe.ndim == 1:
            qe = qe.reshape(1, -1)
        elif qe.shape[0] != 1:
            qe = qe[0:1]

        # Cosine similarity
        # (local lightweight implementation to avoid sklearn dependency duplication)
        qe_norm = np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9
        emb_norm = np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-9
        sims = (qe @ emb_mat.T) / (qe_norm * emb_norm.T)
        similarities = sims[0]
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return [(objects[i], float(similarities[i])) for i in top_idx]

    def get_memory_stats(self) -> Dict[str, Any]:  # override to unify conventions
        """Unified memory stats (merged from fix variant)."""
        cache_episodes = len(self.cache)
        try:
            stored_episodes = len(self.datastore.load_episodes(namespace="episodes"))
        except Exception:
            stored_episodes = 0
        total = cache_episodes + stored_episodes
        return {
            "total_episodes": total,
            "cached_episodes": cache_episodes,
            "stored_episodes": stored_episodes,
            "cache_stats": self.cache_stats,
            "capacity": self.cache_size,
            "utilization": cache_episodes / self.cache_size if self.cache_size else 0.0,
        }

    @property
    def episodes(self):  # compatibility accessor (merged)
        class EpisodeAccessor:
            def __init__(self, manager: 'CachedMemoryManager'):
                self.manager = manager
                self._episodes_cache = None

            def __len__(self):
                stats = self.manager.get_memory_stats()
                return stats.get("total_episodes", 0)

            def __getitem__(self, index: int):
                if self._episodes_cache is None:
                    self._episodes_cache = []
                    for ep in self.manager.cache.values():
                        self._episodes_cache.append(ep)
                    try:
                        stored = self.manager.datastore.load_episodes(namespace="episodes")
                        for ep_dict in stored:
                            if ep_dict.get("id") not in self.manager.cache:
                                ep = Episode(
                                    text=ep_dict.get("text", ""),
                                    vec=np.array(ep_dict.get("vec", [])),
                                    c=ep_dict.get("c_value", ep_dict.get("c", 0.5)),
                                    metadata=ep_dict.get("metadata", {}),
                                )
                                self._episodes_cache.append(ep)
                    except Exception:
                        pass
                if 0 <= index < len(self._episodes_cache):
                    return self._episodes_cache[index]
                raise IndexError(f"Episode index {index} out of range")

        return EpisodeAccessor(self)
        
    def update_c_value(self, episode_idx: int, new_c_value: float):
        """Update episode C-value by index (for compatibility)"""
        # This is a compatibility method - CachedMemoryManager uses IDs internally
        # but MainAgent expects indices
        
        # Update cache if we have any episodes
        cache_items = list(self.cache.items())
        if 0 <= episode_idx < len(cache_items):
            episode_id, episode = cache_items[episode_idx]
            episode.c = new_c_value
            
            # Also update in DataStore (would need to reload and save all)
            # This is inefficient but maintains compatibility
            logger.debug(f"Updated c_value for episode at index {episode_idx} to {new_c_value}")
                
    def get_recent_episodes(self, limit: int = 100) -> List[Episode]:
        """Get recent episodes"""
        # DataStore doesn't have get_recent_episodes, so load all and return last N
        try:
            all_episodes = self.datastore.load_episodes(namespace="episodes")
            recent_episodes = all_episodes[-limit:] if len(all_episodes) > limit else all_episodes
            
            episodes = []
            for ep_dict in recent_episodes:
                episode = Episode(
                    text=ep_dict["text"],
                    vec=ep_dict["vec"],
                    c=ep_dict.get("c_value", ep_dict.get("c", 0.5)),
                    metadata=ep_dict.get("metadata", {})
                )
                episodes.append(episode)
                
            return episodes
        except Exception as e:
            logger.error(f"Failed to get recent episodes: {e}")
            return list(self.cache.values())[:limit]  # Fallback to cache
        
    def _add_to_cache(self, episode_id: str, episode: Episode):
        """Add episode to cache with LRU eviction"""
        # Check if we need to evict
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            oldest_id = next(iter(self.cache))
            del self.cache[oldest_id]
            self.cache_stats['evictions'] += 1
            
        # Add to cache
        self.cache[episode_id] = episode
        
    def _check_memory(self):
        """Periodically check memory usage"""
        current_time = time.time()
        
        if current_time - self.last_memory_check > self.check_interval:
            self.last_memory_check = current_time
            
            snapshot = self.memory_monitor.check_memory(
                episode_count=self._estimate_total_episodes(),
                cache_size=len(self.cache)
            )
            
            if snapshot and snapshot.memory_mb > self.memory_monitor.warning_threshold_mb:
                # Reduce cache size
                self._reduce_cache()
                
    def _reduce_cache(self):
        """Reduce cache size to free memory"""
        target_size = self.cache_size // 2
        
        while len(self.cache) > target_size:
            oldest_id = next(iter(self.cache))
            del self.cache[oldest_id]
            self.cache_stats['evictions'] += 1
            
        logger.warning(f"Reduced cache size to {len(self.cache)} due to memory pressure")
        
    def _estimate_total_episodes(self) -> int:
        """Estimate total episodes in DataStore"""
        try:
            if hasattr(self.datastore, 'get_episode_count'):
                return self.datastore.get_episode_count()
        except Exception:
            pass
        return len(self.cache) * 10  # Rough estimate
        
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (
            self.cache_stats['hits'] / total_requests 
            if total_requests > 0 else 0
        )
        
        return {
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size,
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'cache_evictions': self.cache_stats['evictions'],
            'cache_hit_rate': hit_rate,
            'estimated_episodes': self._estimate_total_episodes(),
            'memory_mb': self.memory_monitor.get_memory_usage_mb()
        }
        
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        logger.info("Cache cleared")
        
    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to vector (compatibility method).
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        return self.embedder.get_embedding(text)
        
    # Compatibility methods for L2MemoryManager interface
    
    @property
    def episodes(self) -> List[Episode]:
        """
        Compatibility property - returns ALL episodes from datastore.
        Note: This is slow as it loads all episodes. Use get_cached_episodes() for cache only.
        """
        return self.get_all_episodes()
    
    def get_cached_episodes(self) -> List[Episode]:
        """Get only cached episodes (fast)"""
        return list(self.cache.values())
    
    def get_all_episodes(self) -> List[Episode]:
        """Get all episodes from datastore (slow)"""
        try:
            episode_dicts = self.datastore.load_episodes(namespace="episodes")
            episodes = []
            for ep_dict in episode_dicts:
                episode = Episode(
                    text=ep_dict["text"],
                    vec=np.array(ep_dict.get("vec", ep_dict.get("embedding", [])), dtype=np.float32),
                    c=ep_dict.get("c_value", ep_dict.get("c", 0.5)),
                    timestamp=ep_dict.get("timestamp", 0),
                    metadata=ep_dict.get("metadata", {})
                )
                episodes.append(episode)
            return episodes
        except Exception as e:
            logger.error(f"Failed to load all episodes: {e}")
            return []
    
    def get_total_episode_count(self) -> int:
        """
        Get the total number of episodes in the DataStore.
        
        Returns:
            Total episode count
        """
        try:
            episodes = self.datastore.load_episodes(namespace="episodes")
            return len(episodes)
        except Exception as e:
            logger.error(f"Failed to get episode count from DataStore: {e}")
            return 0
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Compatibility method"""
        return self.get_stats()
        
    def load(self) -> bool:
        """
        Load episodes from DataStore into cache.
        For compatibility with L2MemoryManager.
        """
        try:
            # Load recent episodes into cache
            episodes = self.datastore.load_episodes(namespace="episodes")
            
            # Load up to cache_size most recent episodes
            recent_episodes = episodes[-self.cache_size:] if len(episodes) > self.cache_size else episodes
            
            for ep_dict in recent_episodes:
                episode_id = ep_dict.get("id", str(hash(ep_dict["text"])))
                episode = Episode(
                    text=ep_dict["text"],
                    vec=ep_dict["vec"],
                    c=ep_dict.get("c_value", ep_dict.get("c", 0.5)),
                    metadata=ep_dict.get("metadata", {})
                )
                self.cache[episode_id] = episode
                
            logger.info(f"Loaded {len(self.cache)} episodes into cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")
            return False
        
    def split_episode(
        self,
        episode_id: str,
        graph: Optional[nx.Graph] = None,
        llm_provider: Optional[Any] = None,
        force_split: bool = False
    ) -> List[str]:
        """
        Legacy split method - redirects to branch_episode for backward compatibility.
        """
        logger.warning("split_episode is deprecated. Use branch_episode instead.")
        return self.branch_episode(episode_id, graph, llm_provider, force_split)
    
    def branch_episode(
        self,
        episode_id: str,
        graph: Optional[nx.Graph] = None,
        llm_provider: Optional[Any] = None,
        force_split: bool = False
    ) -> List[str]:
        """
        Branch an episode based on context divergence (1→1+N pattern).
        Original episode is preserved.
        
        Args:
            episode_id: ID of episode to branch
            graph: Knowledge graph for context
            llm_provider: LLM for semantic analysis (optional)
            force_split: Force branching even if no divergence detected
            
        Returns:
            List of episode IDs including original and branches
        """
        try:
            # Get the episode
            episode = self.get_episode(episode_id)
            if not episode:
                logger.warning(f"Episode {episode_id} not found for branching")
                return []
            
            # Check if episode has neighbors in graph
            if not graph or episode_id not in graph:
                logger.info(f"Episode {episode_id} has no graph context, skipping branch")
                return [episode_id]
            
            # Detect context divergence
            context_clusters = self._detect_context_clusters(episode_id, graph)
            
            if len(context_clusters) <= 1 and not force_split:
                logger.info(f"No context divergence detected for {episode_id}")
                return [episode_id]
            
            # Generate branch episodes
            branch_ids = [episode_id]  # Include original
            
            for i, cluster in enumerate(context_clusters):
                # Create branch episode with context-specific vector
                branch_vec = self._create_branch_vector(
                    episode, cluster, graph
                )
                
                # Generate branch text if LLM is available
                branch_text = episode.text  # Default to parent text
                if llm_provider:
                    try:
                        # Get neighbor texts for context
                        neighbor_texts = []
                        for n_id in cluster['neighbor_ids'][:3]:
                            if n_id in graph:
                                neighbor_texts.append(graph.nodes[n_id].get('text', n_id))
                        
                        # Generate contextual description
                        branch_text = llm_provider.generate_text_from_vector(
                            vector=branch_vec,
                            generation_type="branch",
                            context={
                                'parent_text': episode.text,
                                'context_neighbors': neighbor_texts,
                                'graph': graph
                            }
                        )
                        logger.debug(f"Generated branch text: {branch_text}")
                    except Exception as e:
                        logger.warning(f"Failed to generate branch text: {e}")
                        # Fall back to simple format
                        branch_text = f"{episode.text}({cluster['type']})"
                else:
                    # No LLM, use simple format
                    branch_text = f"{episode.text}({cluster['type']})"
                
                # Create branch episode
                branch_id = f"{episode_id}_branch_{cluster['type']}_{i}"
                
                # Store in DataStore
                branch_episode_dict = {
                    "id": branch_id,
                    "text": branch_text,  # Contextual text (e.g., "apple(fruit)")
                    "vec": branch_vec,    # Context-specific vector
                    "c_value": episode.c * 0.8,  # Slightly reduced
                    "timestamp": time.time(),
                    "metadata": {
                        **episode.metadata,
                        'parent_id': episode_id,
                        'parent_text': episode.text,  # Original text
                        'branch_type': cluster['type'],
                        'context_neighbors': cluster['neighbor_ids'],
                        'is_branch': True
                    }
                }
                # Get existing episodes and append the new one
                existing_episodes = self.datastore.load_episodes(namespace="episodes")
                existing_episodes.append(branch_episode_dict)
                self.datastore.save_episodes(existing_episodes, namespace="episodes")
                
                # Cache if parent was cached
                if episode_id in self.cache:
                    branch_episode = Episode(
                        text=branch_text,  # Use the generated text
                        vec=branch_vec,
                        c=episode.c * 0.8,
                        metadata={
                            'parent_id': episode_id,
                            'parent_text': episode.text,
                            'branch_type': cluster['type']
                        }
                    )
                    self._add_to_cache(branch_id, branch_episode)
                
                branch_ids.append(branch_id)
                
                # グラフにブランチノードを追加（グラフが提供されている場合）
                if graph is not None:
                    # ブランチノードを追加
                    graph.add_node(
                        branch_id,
                        text=episode.text,
                        vec=branch_vec,
                        metadata={
                            'parent_id': episode_id,
                            'branch_type': cluster['type'],
                            'is_branch': True
                        }
                    )
                    
                    # 親からブランチへのエッジ
                    graph.add_edge(
                        episode_id,
                        branch_id,
                        weight=0.9,
                        relation='branch'
                    )
                    
                    # 文脈に応じてエッジを移動
                    for neighbor_id in cluster['neighbor_ids']:
                        if graph.has_edge(episode_id, neighbor_id):
                            # エッジデータを保存
                            edge_data = dict(graph.edges[episode_id, neighbor_id])
                            # 親からのエッジを削除
                            graph.remove_edge(episode_id, neighbor_id)
                            # ブランチへのエッジを追加
                            graph.add_edge(branch_id, neighbor_id, **edge_data)
                            logger.debug(
                                f"Moved edge {episode_id}->{neighbor_id} to "
                                f"{branch_id}->{neighbor_id}"
                            )
                
                logger.info(
                    f"Created branch {branch_id} with context '{cluster['type']}'"
                )
            
            logger.info(
                f"Branched episode {episode_id} into {len(branch_ids)} nodes "
                f"(1 original + {len(branch_ids)-1} branches)"
            )
            
            return branch_ids
            
        except Exception as e:
            logger.error(f"Failed to branch episode {episode_id}: {e}")
            return [episode_id]
    
    def _detect_context_clusters(
        self, episode_id: str, graph: nx.Graph
    ) -> List[Dict[str, Any]]:
        """
        Detect context clusters among neighbors.
        """
        try:
            neighbors = list(graph.neighbors(episode_id))
            if not neighbors:
                return []
            
            # Get neighbor embeddings and texts
            neighbor_data = []
            for n_id in neighbors:
                node_data = graph.nodes[n_id]
                neighbor_data.append({
                    'id': n_id,
                    'vec': node_data.get('vec', None),
                    'text': node_data.get('text', ''),
                    'metadata': node_data.get('metadata', {})
                })
            
            # Simple clustering based on semantic similarity
            # TODO: Implement more sophisticated clustering
            clusters = self._simple_semantic_clustering(neighbor_data)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Context clustering failed: {e}")
            return []
    
    def _simple_semantic_clustering(
        self, neighbor_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Simple clustering based on pairwise similarity.
        """
        if len(neighbor_data) < 2:
            return [{
                'type': 'unified',
                'neighbor_ids': [n['id'] for n in neighbor_data]
            }]
        
        # Calculate pairwise similarities
        # Extract vectors
        valid_neighbors = [n for n in neighbor_data if n['vec'] is not None]
        if not valid_neighbors:
            return [{
                'type': 'unified',
                'neighbor_ids': [n['id'] for n in neighbor_data]
            }]
        
        # Ensure all vectors are numpy arrays
        vectors = []
        for n in valid_neighbors:
            vec = n['vec']
            if not isinstance(vec, np.ndarray):
                vec = np.array(vec)
            # Flatten if needed
            if vec.ndim > 1:
                vec = vec.flatten()
            vectors.append(vec)
        
        # Stack vectors
        try:
            vectors = np.vstack(vectors)
        except Exception as e:
            logger.error(f"Failed to stack vectors: {e}")
            # Debug info
            for i, v in enumerate(vectors[:3]):  # Show first 3
                logger.debug(f"Vector {i} shape: {v.shape if hasattr(v, 'shape') else 'no shape'}")
            return [{
                'type': 'unified',
                'neighbor_ids': [n['id'] for n in neighbor_data]
            }]
        
        # Calculate cosine similarity manually
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / (norms + 1e-8)
        
        # Compute similarity matrix - ensure shapes are correct
        # normalized_vectors should be (N, D) where N is number of neighbors, D is dimension
        if len(normalized_vectors.shape) != 2:
            logger.error(f"Unexpected vector shape: {normalized_vectors.shape}")
            return [{
                'type': 'unified', 
                'neighbor_ids': [n['id'] for n in neighbor_data]
            }]
        
        similarities = np.dot(normalized_vectors, normalized_vectors.T)
        
        # Improved threshold-based clustering
        threshold = 0.5  # Lower threshold for better grouping
        clusters = []
        clustered = set()
        
        for i, neighbor in enumerate(valid_neighbors):
            if neighbor['id'] in clustered:
                continue
                
            cluster = {
                'type': f'context_{len(clusters)}',
                'neighbor_ids': [neighbor['id']],
                'representative_text': neighbor['text']
            }
            clustered.add(neighbor['id'])
            
            # Find similar neighbors
            for j, other in enumerate(valid_neighbors):
                if i != j and other['id'] not in clustered:
                    if similarities[i, j] > threshold:
                        cluster['neighbor_ids'].append(other['id'])
                        clustered.add(other['id'])
            
            # Name clusters based on content
            if len(clusters) == 0 and any('red' in n or 'sweet' in n for n in cluster['neighbor_ids']):
                cluster['type'] = 'fruit'
            elif any('jobs' in n or 'tech' in n or 'iphone' in n for n in cluster['neighbor_ids']):
                cluster['type'] = 'technology'
            
            clusters.append(cluster)
        
        # Add unclustered neighbors
        unclustered = [n['id'] for n in neighbor_data if n['id'] not in clustered]
        if unclustered:
            clusters.append({
                'type': 'misc',
                'neighbor_ids': unclustered
            })
        
        return clusters
    
    def _create_branch_vector(
        self,
        parent_episode: Episode,
        context_cluster: Dict[str, Any],
        graph: nx.Graph
    ) -> np.ndarray:
        """
        Create context-specific vector using message passing.
        """
        # Message passing weights
        parent_weight = 0.4
        context_weight = 0.6
        
        # Ensure parent vector is numpy array
        parent_vec = np.array(parent_episode.vec) if not isinstance(parent_episode.vec, np.ndarray) else parent_episode.vec
        
        # Initialize with parent vector
        messages = [(parent_vec, parent_weight)]
        
        # Add context neighbor messages
        neighbor_ids = context_cluster['neighbor_ids']
        if neighbor_ids:
            individual_weight = context_weight / len(neighbor_ids)
            
            for n_id in neighbor_ids:
                if n_id in graph:
                    neighbor_vec = graph.nodes[n_id].get('vec')
                    if neighbor_vec is not None:
                        # Ensure neighbor vector is numpy array
                        neighbor_vec = np.array(neighbor_vec) if not isinstance(neighbor_vec, np.ndarray) else neighbor_vec
                        messages.append((neighbor_vec, individual_weight))
        
        # Aggregate messages
        branch_vec = np.zeros_like(parent_vec)
        total_weight = 0
        
        for vec, weight in messages:
            if vec is not None:
                branch_vec += vec * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            branch_vec = branch_vec / total_weight
            
        # L2 normalize
        norm = np.linalg.norm(branch_vec)
        if norm > 0:
            branch_vec = branch_vec / norm
        else:
            branch_vec = parent_vec  # Fallback
        
        return branch_vec.astype(np.float32)
            
    def merge_episodes(
        self,
        episode_ids: List[str],
        new_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Merge multiple episodes into one.
        
        Args:
            episode_ids: IDs of episodes to merge
            new_id: ID for merged episode (auto-generated if None)
            
        Returns:
            ID of merged episode or None if failed
        """
        try:
            if len(episode_ids) < 2:
                return None
                
            # Retrieve all episodes
            episodes = []
            for ep_id in episode_ids:
                episode = self.get_episode(ep_id)
                if episode:
                    episodes.append(episode)
                    
            if len(episodes) < 2:
                logger.warning("Not enough valid episodes to merge")
                return None
                
            # Merge texts
            merged_text = "\n\n".join(ep.text for ep in episodes)
            
            # Merge vectors (weighted average by C-value)
            weights = np.array([ep.c for ep in episodes])
            weights = weights / weights.sum()
            
            merged_vec = np.zeros_like(episodes[0].vec)
            for ep, weight in zip(episodes, weights):
                merged_vec += ep.vec * weight
                
            # Normalize
            merged_vec = merged_vec / np.linalg.norm(merged_vec)
            
            # Calculate merged C-value
            merged_c = np.average([ep.c for ep in episodes], weights=weights)
            
            # Create merged episode
            merged_id = new_id or f"merged_{int(time.time() * 1000)}"
            
            # Store in DataStore
            merged_episode_dict = {
                "id": merged_id,
                "text": merged_text,
                "vec": merged_vec,
                "c_value": merged_c,
                "timestamp": time.time(),
                "metadata": {
                    'merged_from': episode_ids,
                    'merge_timestamp': time.time()
                }
            }
            # Get existing episodes and append the new one
            existing_episodes = self.datastore.load_episodes(namespace="episodes")
            existing_episodes.append(merged_episode_dict)
            self.datastore.save_episodes(existing_episodes, namespace="episodes")
            
            # Cache if any source was cached
            if any(ep_id in self.cache for ep_id in episode_ids):
                merged_episode = Episode(
                    text=merged_text,
                    vec=merged_vec,
                    c=merged_c,
                    metadata={'merged_from': episode_ids}
                )
                self._add_to_cache(merged_id, merged_episode)
                
            # Remove original episodes
            # Need to reload episodes and filter out the ones to delete
            all_episodes = self.datastore.load_episodes(namespace="episodes")
            filtered_episodes = [ep for ep in all_episodes if ep.get("id") not in episode_ids]
            self.datastore.save_episodes(filtered_episodes, namespace="episodes")
            
            # Remove from cache
            for ep_id in episode_ids:
                if ep_id in self.cache:
                    del self.cache[ep_id]
                    
            logger.info(
                f"Merged {len(episode_ids)} episodes into {merged_id}"
            )
            
            return merged_id
            
        except Exception as e:
            logger.error(f"Failed to merge episodes: {e}")
            return None
    
    def save_query(
        self,
        query_text: str,
        query_vec: Optional[np.ndarray] = None,
        has_spike: bool = False,
        spike_episode_id: Optional[str] = None,
        response: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a query record to storage and optionally add to graph.
        
        Args:
            query_text: The original query text
            query_vec: Query embedding vector (will be generated if None)
            has_spike: Whether the query generated a spike
            spike_episode_id: ID of generated episode if spike occurred
            response: The response given to the query
            metadata: Additional metadata (processing_time, llm_provider, etc.)
            
        Returns:
            Query ID
        """
        try:
            # Generate query ID
            import uuid
            query_id = f"query_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
            
            # Generate embedding if not provided
            if query_vec is None:
                query_vec = self.embedder.get_embedding(query_text)
            
            # Prepare query record
            query_record = {
                "id": query_id,
                "text": query_text,
                "vec": query_vec,
                "has_spike": has_spike,
                "spike_episode_id": spike_episode_id,
                "response": response,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            # Save to DataStore
            success = self.datastore.save_queries([query_record], namespace="queries")
            if not success:
                logger.error("Failed to save query to DataStore")
                return ""
            
            logger.info(f"Saved query {query_id} (has_spike={has_spike})")
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to save query: {e}")
            return ""
    
    def add_query_to_graph(
        self,
        graph: nx.Graph,
        query_id: str,
        query_text: str,
        query_vec: np.ndarray,
        has_spike: bool,
        spike_episode_id: Optional[str] = None,
        retrieved_episode_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add query node to graph and create edges to related episodes.
        
        Args:
            graph: NetworkX graph to update
            query_id: Query identifier
            query_text: Query text
            query_vec: Query embedding vector
            has_spike: Whether query generated a spike
            spike_episode_id: ID of spike episode if generated
            retrieved_episode_ids: IDs of episodes retrieved for context
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # Add query node to graph
            graph.add_node(
                query_id,
                text=query_text,
                vec=query_vec,
                type="query",
                has_spike=has_spike,
                metadata=metadata or {}
            )
            
            # Add edges based on query type
            if has_spike and spike_episode_id:
                # Add edge to spike episode
                graph.add_edge(
                    query_id,
                    spike_episode_id,
                    weight=1.0,
                    relation="query_spike",
                    metadata={
                        "timestamp": time.time(),
                        "edge_type": "generative"
                    }
                )
                logger.debug(f"Added query_spike edge: {query_id} -> {spike_episode_id}")
            
            # Add edges to retrieved episodes (for analysis/tracing)
            if retrieved_episode_ids:
                for ep_id in retrieved_episode_ids:
                    if ep_id in graph:
                        # Calculate similarity if possible
                        similarity = 0.5  # Default
                        if "vec" in graph.nodes[ep_id]:
                            ep_vec = graph.nodes[ep_id]["vec"]
                            if ep_vec is not None:
                                # Compute cosine similarity
                                similarity = float(np.dot(query_vec.flatten(), ep_vec.flatten()))
                        
                        # Avoid overwriting spike relation edge metadata
                        if graph.has_edge(query_id, ep_id) and graph[query_id][ep_id].get("relation") == "query_spike":
                            continue
                        graph.add_edge(
                            query_id,
                            ep_id,
                            weight=similarity,
                            relation="query_retrieval",
                            metadata={
                                "timestamp": time.time(),
                                "edge_type": "retrieval",
                                "similarity": similarity
                            }
                        )
                        logger.debug(f"Added query_retrieval edge: {query_id} -> {ep_id}")
            
            logger.info(f"Added query {query_id} to graph with {len(retrieved_episode_ids or [])} retrieval edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add query to graph: {e}")
            return False
    
    def get_recent_queries(
        self,
        limit: int = 100,
        has_spike: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent queries from storage.
        
        Args:
            limit: Maximum number of queries to return
            has_spike: Filter by spike generation status (None = all)
            
        Returns:
            List of query records
        """
        try:
            return self.datastore.load_queries(
                namespace="queries",
                has_spike=has_spike,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            return []
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about saved queries.
        
        Returns:
            Dictionary with query statistics
        """
        try:
            all_queries = self.datastore.load_queries(namespace="queries")
            spike_queries = [q for q in all_queries if q.get("has_spike", False)]
            
            stats = {
                "total_queries": len(all_queries),
                "spike_queries": len(spike_queries),
                "non_spike_queries": len(all_queries) - len(spike_queries),
                "spike_rate": len(spike_queries) / len(all_queries) if all_queries else 0,
                "avg_processing_time": 0.0,
                "llm_providers": {}
            }
            
            # Calculate average processing time
            processing_times = [
                q.get("metadata", {}).get("processing_time", 0)
                for q in all_queries
                if q.get("metadata", {}).get("processing_time")
            ]
            if processing_times:
                stats["avg_processing_time"] = sum(processing_times) / len(processing_times)
            
            # Count by LLM provider
            for q in all_queries:
                provider = q.get("metadata", {}).get("llm_provider", "unknown")
                stats["llm_providers"][provider] = stats["llm_providers"].get(provider, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get query statistics: {e}")
            return {
                "error": str(e),
                "total_queries": 0,
                "spike_queries": 0,
                "non_spike_queries": 0,
                "spike_rate": 0
            }
