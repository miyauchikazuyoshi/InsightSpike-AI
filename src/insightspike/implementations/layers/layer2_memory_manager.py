"""
Layer 2: Memory Manager
======================

Graph-centric episodic memory management system (Hippocampus + Locus Coeruleus analog).
Consolidates all memory management variants into a single configurable class.
"""

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

# Optional import with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from ...config import get_config
from ...core.episode import Episode
from ...processing.embedder import EmbeddingManager

# Optional components - not yet implemented
# from ..memory.importance_scorer import ImportanceScorer
# from ..memory.memory_index import MemoryIndex
from .scalable_graph_builder import ScalableGraphBuilder

logger = logging.getLogger(__name__)


class MemoryMode(Enum):
    """Different operation modes for memory management"""
    BASIC = "basic"                    # Standard C-value based
    ENHANCED = "enhanced"              # Graph-aware with conflicts
    SCALABLE = "scalable"             # Optimized for large datasets
    GRAPH_CENTRIC = "graph_centric"    # Pure graph-based (no C-values)


@dataclass
class MemoryConfig:
    """Unified configuration for memory manager"""
    
    # Mode selection
    mode: MemoryMode = MemoryMode.SCALABLE
    
    # Core settings
    embedding_dim: int = 384
    max_episodes: int = 10000
    
    # Feature toggles
    use_c_values: bool = True
    use_graph_integration: bool = False
    use_conflict_detection: bool = False
    use_importance_scoring: bool = False
    use_scalable_indexing: bool = True
    use_hierarchical_graph: bool = False
    
    # Index settings
    faiss_index_type: str = "Flat"  # Flat, IVF, IVFPQ
    ivf_nlist: int = 100
    ivfpq_m: int = 8
    
    # Graph settings
    similarity_threshold: float = 0.7
    max_graph_edges: int = 10000
    
    # Performance settings
    batch_size: int = 32
    cache_embeddings: bool = True
    
    @classmethod
    def from_mode(cls, mode: MemoryMode, **kwargs) -> "MemoryConfig":
        """Create config with presets for a specific mode"""
        config = cls(mode=mode, **kwargs)
        
        if mode == MemoryMode.BASIC:
            # Simple C-value based memory
            config.use_graph_integration = False
            config.use_conflict_detection = False
            config.use_scalable_indexing = False
            
        elif mode == MemoryMode.ENHANCED:
            # Add graph awareness and conflict detection
            config.use_graph_integration = True
            config.use_conflict_detection = True
            config.use_importance_scoring = True
            
        elif mode == MemoryMode.SCALABLE:
            # Optimized for performance
            config.use_scalable_indexing = True
            config.faiss_index_type = "IVF"
            config.cache_embeddings = True
            
        elif mode == MemoryMode.GRAPH_CENTRIC:
            # No C-values, pure graph
            config.use_c_values = False
            config.use_graph_integration = True
            config.use_importance_scoring = True
            
        return config


class L2MemoryManager:
    """
    Layer 2 Memory Manager - Graph-centric episodic memory.
    
    Brain analog: Hippocampus (memory formation) + Locus Coeruleus (attention/importance)
    
    Features:
    - FAISS-indexed vector search for efficient retrieval
    - Configurable modes: Basic, Enhanced, Scalable, Graph-Centric
    - Episode management: merge, split, prune operations
    - Transitioning from C-values to graph-based importance
    """
    
    def __init__(self, config: Optional[Union[MemoryConfig, Dict[str, Any], Any]] = None):
        """Initialize with unified configuration"""
        # Handle different config types
        if isinstance(config, MemoryConfig):
            self.config = config
        elif isinstance(config, dict):
            # Create MemoryConfig from dict
            memory_config = config.get('memory', {})
            self.config = MemoryConfig(
                mode=MemoryMode(memory_config.get('mode', 'scalable')),
                embedding_dim=memory_config.get('embedding_dim', 384),
                max_episodes=memory_config.get('max_episodes', 10000),
                use_c_values=memory_config.get('use_c_values', True),
                use_graph_integration=memory_config.get('use_graph_integration', False),
                use_scalable_indexing=memory_config.get('use_scalable_indexing', True),
                batch_size=memory_config.get('batch_size', 32),
                cache_embeddings=memory_config.get('cache_embeddings', True)
            )
        elif hasattr(config, 'memory'):
            # Handle InsightSpikeConfig or similar
            self.config = MemoryConfig(
                max_episodes=getattr(config.memory, 'max_episodes', 10000),
                batch_size=getattr(config.memory, 'batch_size', 32)
            )
        else:
            self.config = MemoryConfig()
            
        # Core components
        self.episodes: List[Episode] = []
        self.embedding_model = None
        self.faiss_index = None
        self.graph_builder = None
        self.importance_scorer = None
        
        # State tracking
        self.embedding_cache = {} if self.config.cache_embeddings else None
        self.last_graph_update = 0
        self.initialized = False
        
        self._setup_components()
        
    def _setup_components(self):
        """Setup components based on configuration"""
        # Embedding model
        try:
            self.embedding_model = EmbeddingManager()
            # EmbeddingManager doesn't have get_embedding_dim, use default
            if hasattr(self.embedding_model, 'model') and hasattr(self.embedding_model.model, 'get_sentence_embedding_dimension'):
                self.config.embedding_dim = self.embedding_model.model.get_sentence_embedding_dimension()
            logger.info(f"Using embedding dimension: {self.config.embedding_dim}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            
        # FAISS index
        self._setup_faiss_index()
        
        # Graph builder (if needed)
        if self.config.use_graph_integration:
            # ScalableGraphBuilder uses config object, not individual params
            graph_config = type('Config', (), {
                'graph': type('Graph', (), {
                    'similarity_threshold': self.config.similarity_threshold
                })(),
                'scalable_graph': type('ScalableGraph', (), {
                    'top_k_neighbors': 10,
                    'batch_size': 1000
                })()
            })()
            self.graph_builder = ScalableGraphBuilder(config=graph_config)
            logger.info("Graph integration enabled")
            
        # Importance scorer (if needed)
        if self.config.use_importance_scoring:
            # self.importance_scorer = ImportanceScorer()  # Not yet implemented
            self.importance_scorer = None
            logger.info("Importance scoring requested but not yet implemented")
            
        self.initialized = True
        
    def _setup_faiss_index(self):
        """Setup FAISS index based on configuration"""
        dim = self.config.embedding_dim
        
        # Always use Flat index initially, will be upgraded if needed
        self.faiss_index = faiss.IndexFlatL2(dim)
        logger.info("Using Flat FAISS index")
        self._index_type = "Flat"
        
        # We'll upgrade to IVF when we have enough data
        if False and self.config.faiss_index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, self.config.ivf_nlist)
            logger.info(f"Using IVF FAISS index with {self.config.ivf_nlist} cells")
            
        elif self.config.faiss_index_type == "IVFPQ":
            quantizer = faiss.IndexFlatL2(dim)
            self.faiss_index = faiss.IndexIVFPQ(
                quantizer, dim, self.config.ivf_nlist, self.config.ivfpq_m, 8
            )
            logger.info(f"Using IVFPQ FAISS index")
            
                
    def add_episode(self, text: str, metadata: Optional[Dict] = None) -> int:
        """Add episode (graph-centric interface)"""
        c_value = metadata.get('c_value', 0.5) if metadata else 0.5
        return self.store_episode(text, c_value, metadata)
        
    def store_episode(self, text: str, c_value: float = 0.5, 
                     metadata: Optional[Dict] = None) -> int:
        """
        Store a new episode in memory.
        
        Unified interface that handles all modes appropriately.
        """
        if not self.initialized:
            logger.error("Memory manager not initialized")
            return -1
            
        try:
            # Create embedding
            embedding = self._get_embedding(text)
            if embedding is None:
                return -1
                
            # Create episode
            episode = Episode(
                text=text,
                vec=embedding,
                c=c_value if self.config.use_c_values else 1.0,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # Mode-specific processing
            if self.config.mode == MemoryMode.GRAPH_CENTRIC:
                # Calculate graph-based importance
                if self.importance_scorer and len(self.episodes) > 0:
                    importance = self._calculate_graph_importance(episode)
                    episode.metadata['importance'] = importance
                    
            elif self.config.use_conflict_detection:
                # Check for conflicts
                conflicts = self._detect_conflicts(episode)
                if conflicts:
                    self._handle_conflicts(episode, conflicts)
                    
            # Add to episodes
            episode_idx = len(self.episodes)
            self.episodes.append(episode)
            
            # Update index
            self._update_index(episode, episode_idx)
            
            # Update graph if needed
            if self.config.use_graph_integration:
                self._update_graph(episode, episode_idx)
                
            logger.debug(f"Stored episode {episode_idx}: {text[:50]}...")
            return episode_idx
            
        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            return -1
            
    def search_episodes(self, query: str, k: int = 5, 
                       filter_fn: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant episodes.
        
        Unified search that works across all modes.
        """
        if not self.initialized or not self.episodes:
            return []
            
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []
                
            # Search using FAISS
            distances, indices = self._search_index(query_embedding, k * 2)
            
            # Build results
            results = []
            for dist, idx in zip(distances, indices):
                if idx < 0 or idx >= len(self.episodes):
                    continue
                    
                episode = self.episodes[idx]
                
                # Apply filter if provided
                if filter_fn and not filter_fn(episode):
                    continue
                    
                # Calculate relevance score
                similarity = 1.0 / (1.0 + dist)
                
                # Mode-specific scoring
                if self.config.mode == MemoryMode.GRAPH_CENTRIC:
                    # Use graph-based importance
                    importance = episode.metadata.get('importance', 0.5)
                    relevance = similarity * importance
                else:
                    # Use C-value
                    relevance = similarity * episode.c
                    
                result = {
                    'text': episode.text,
                    'similarity': similarity,
                    'relevance': relevance,
                    'c_value': episode.c,
                    'index': idx,
                    'timestamp': episode.timestamp,
                    'metadata': episode.metadata
                }
                
                results.append(result)
                
            # Sort by relevance and return top k
            results.sort(key=lambda x: x['relevance'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
            
    def update_c_value(self, episode_idx: int, new_c_value: float):
        """Update C-value for an episode"""
        if not self.config.use_c_values:
            logger.warning("C-values disabled in current mode")
            return
            
        if 0 <= episode_idx < len(self.episodes):
            self.episodes[episode_idx].c = max(0.0, min(1.0, new_c_value))
            logger.debug(f"Updated episode {episode_idx} C-value to {new_c_value}")
            
    def merge_episodes(self, indices: List[int]) -> int:
        """Merge multiple episodes into one"""
        if len(indices) < 2:
            return -1
            
        # Gather episodes
        episodes_to_merge = [self.episodes[i] for i in indices if 0 <= i < len(self.episodes)]
        if len(episodes_to_merge) < 2:
            return -1
            
        # Combine texts
        combined_text = " ".join([ep.text for ep in episodes_to_merge])
        
        # Average C-values (if used)
        avg_c_value = np.mean([ep.c for ep in episodes_to_merge])
        
        # Merge metadata
        merged_metadata = {}
        for ep in episodes_to_merge:
            merged_metadata.update(ep.metadata)
        merged_metadata['merged_from'] = indices
        
        # Remove old episodes (in reverse order to maintain indices)
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.episodes):
                del self.episodes[idx]
                
        # Add merged episode
        return self.store_episode(combined_text, avg_c_value, merged_metadata)
        
    def prune_low_value_episodes(self, threshold: float = 0.1) -> int:
        """Remove episodes below threshold"""
        if not self.config.use_c_values:
            logger.warning("Pruning not available without C-values")
            return 0
            
        initial_count = len(self.episodes)
        
        # Filter episodes
        self.episodes = [ep for ep in self.episodes if ep.c >= threshold]
        
        # Rebuild index
        self._rebuild_index()
        
        pruned_count = initial_count - len(self.episodes)
        logger.info(f"Pruned {pruned_count} low-value episodes")
        return pruned_count
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self.episodes:
            return {
                'total_episodes': 0,
                'mode': self.config.mode.value,
                'features_enabled': self._get_enabled_features()
            }
            
        c_values = [ep.c for ep in self.episodes] if self.config.use_c_values else []
        
        stats = {
            'total_episodes': len(self.episodes),
            'mode': self.config.mode.value,
            'features_enabled': self._get_enabled_features(),
            'index_type': self.config.faiss_index_type,
            'embedding_dim': self.config.embedding_dim
        }
        
        if c_values:
            stats.update({
                'c_value_mean': np.mean(c_values),
                'c_value_std': np.std(c_values),
                'c_value_min': np.min(c_values),
                'c_value_max': np.max(c_values)
            })
            
        if self.config.use_graph_integration and self.graph_builder:
            # ScalableGraphBuilder doesn't have get_current_graph, so we skip graph stats
            # TODO: Implement graph stats retrieval for ScalableGraphBuilder
            stats['graph_nodes'] = len(self.episodes)  # Approximate with episode count
            stats['graph_edges'] = 0  # Unknown without building graph
                
        return stats
        
    def save(self, path: Optional[str] = None) -> bool:
        """Save memory state to disk.
        
        .. deprecated:: 2.0
           This method is deprecated. Memory persistence should be handled
           by MainAgent using DataStore abstraction.
        """
        warnings.warn(
            "save is deprecated. Use MainAgent with DataStore for persistence.",
            DeprecationWarning,
            stacklevel=2
        )
        if not path:
            config = get_config()
            path = os.path.join(config.paths.data_dir, "layer2_memory.json")
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare data
            data = {
                'mode': self.config.mode.value,
                'episodes': [],
                'config': {
                    'embedding_dim': self.config.embedding_dim,
                    'use_c_values': self.config.use_c_values,
                    'use_graph_integration': self.config.use_graph_integration
                }
            }
            
            # Save episodes
            for ep in self.episodes:
                ep_data = {
                    'text': ep.text,
                    'c_value': ep.c,
                    'timestamp': ep.timestamp,
                    'metadata': ep.metadata,
                    'embedding': ep.vec.tolist() if isinstance(ep.vec, np.ndarray) else ep.vec
                }
                data['episodes'].append(ep_data)
                
            # Write to file
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.episodes)} episodes to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return False
            
    def load(self, path: Optional[str] = None) -> bool:
        """Load memory state from disk.
        
        .. deprecated:: 2.0
           This method is deprecated. Memory persistence should be handled
           by MainAgent using DataStore abstraction.
        """
        warnings.warn(
            "load is deprecated. Use MainAgent with DataStore for persistence.",
            DeprecationWarning,
            stacklevel=2
        )
        if not path:
            config = get_config()
            path = os.path.join(config.paths.data_dir, "layer2_memory.json")
            
        if not os.path.exists(path):
            logger.info(f"No saved memory found at {path}")
            return False
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Clear current state
            self.episodes.clear()
            self._rebuild_index()
            
            # Load episodes
            for ep_data in data.get('episodes', []):
                episode = Episode(
                    text=ep_data['text'],
                    vec=np.array(ep_data['embedding']),
                    c=ep_data.get('c_value', 0.5),
                    timestamp=ep_data.get('timestamp', time.time()),
                    metadata=ep_data.get('metadata', {})
                )
                self.episodes.append(episode)
                
            # Rebuild index with loaded episodes
            self._rebuild_index()
            
            logger.info(f"Loaded {len(self.episodes)} episodes from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return False
            
    # Private helper methods
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""
        if not self.embedding_model:
            return np.random.randn(self.config.embedding_dim).astype(np.float32)
            
        try:
            # Check cache
            if self.embedding_cache is not None and text in self.embedding_cache:
                return self.embedding_cache[text]
                
            # Generate embedding
            embedding = self.embedding_model.encode(text)[0]
            
            # Ensure it's a numpy array with float32 dtype
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            elif embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
            
            # Cache if enabled
            if self.embedding_cache is not None:
                self.embedding_cache[text] = embedding
                
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
            
    def _update_index(self, episode: Episode, idx: int):
        """Update FAISS index with new episode"""
        if self.faiss_index is None:
            return
            
        # For IVF indices, train if needed
        if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
            if len(self.episodes) >= self.config.ivf_nlist:
                vectors = np.array([ep.vec for ep in self.episodes])
                self.faiss_index.train(vectors)
                logger.info("FAISS index trained")
                
        # Add vector
        if hasattr(self.faiss_index, 'is_trained') and self.faiss_index.is_trained:
            self.faiss_index.add(np.array([episode.vec]))
        elif not hasattr(self.faiss_index, 'is_trained'):
            # Flat index, always ready
            self.faiss_index.add(np.array([episode.vec]))
            
    def _search_index(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return np.array([]), np.array([])
            
        # Ensure k doesn't exceed index size
        k = min(k, self.faiss_index.ntotal)
        
        # Search
        distances, indices = self.faiss_index.search(np.array([query_vec]), k)
        return distances[0], indices[0]
        
    def _rebuild_index(self):
        """Rebuild FAISS index from scratch"""
        self._setup_faiss_index()
        
        if not self.episodes:
            return
            
        # Add all vectors
        vectors = np.array([ep.vec for ep in self.episodes])
        
        # Train if needed
        if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
            self.faiss_index.train(vectors)
            
        # Add vectors
        self.faiss_index.add(vectors)
        logger.info(f"Rebuilt index with {len(self.episodes)} episodes")
        
    def _update_graph(self, episode: Episode, idx: int):
        """Update graph with new episode"""
        if not self.graph_builder:
            return
            
        # Only update periodically for performance
        if len(self.episodes) % 10 == 0:
            # Build full document list
            documents = []
            for i, ep in enumerate(self.episodes):
                doc = {
                    'text': ep.text,
                    'embedding': ep.vec,
                    'episode_idx': i,
                    'metadata': ep.metadata
                }
                documents.append(doc)
                
            # Update graph
            self.graph_builder.build_graph(documents)
            self.last_graph_update = time.time()
            
    def _calculate_graph_importance(self, episode: Episode) -> float:
        """Calculate importance based on graph position"""
        if not self.importance_scorer or not self.episodes:
            return 0.5
            
        # Simple similarity-based importance
        similarities = []
        for other in self.episodes[-10:]:  # Last 10 episodes
            sim = np.dot(episode.vec, other.vec) / (np.linalg.norm(episode.vec) * np.linalg.norm(other.vec))
            similarities.append(sim)
            
        return float(np.mean(similarities)) if similarities else 0.5
        
    def _detect_conflicts(self, episode: Episode) -> List[int]:
        """Detect conflicting episodes"""
        if not self.config.use_conflict_detection:
            return []
            
        conflicts = []
        
        # Search for similar episodes
        if len(self.episodes) > 0:
            distances, indices = self._search_index(episode.vec, k=5)
            
            for dist, idx in zip(distances, indices):
                if idx < 0 or idx >= len(self.episodes):
                    continue
                    
                similarity = 1.0 / (1.0 + dist)
                if similarity > 0.9:  # Very similar
                    # Check if texts are contradictory (simple heuristic)
                    other = self.episodes[idx]
                    if self._texts_conflict(episode.text, other.text):
                        conflicts.append(idx)
                        
        return conflicts
        
    def _texts_conflict(self, text1: str, text2: str) -> bool:
        """Simple heuristic to detect conflicting texts"""
        # Check for negation patterns
        negations = ['not', 'no', 'never', 'neither', 'nor', 'nothing']
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # If one has negation and high overlap, might be conflict
        has_negation1 = any(neg in words1 for neg in negations)
        has_negation2 = any(neg in words2 for neg in negations)
        
        if has_negation1 != has_negation2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap > 0.5
            
        return False
        
    def _handle_conflicts(self, episode: Episode, conflicts: List[int]):
        """Handle detected conflicts"""
        logger.warning(f"Detected {len(conflicts)} conflicts for new episode")
        
        # For now, just log conflicts
        # Could implement more sophisticated resolution
        for idx in conflicts:
            logger.debug(f"Conflict with episode {idx}: {self.episodes[idx].text[:50]}...")
            
    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled features"""
        features = []
        
        if self.config.use_c_values:
            features.append("c_values")
        if self.config.use_graph_integration:
            features.append("graph_integration")
        if self.config.use_conflict_detection:
            features.append("conflict_detection")
        if self.config.use_importance_scoring:
            features.append("importance_scoring")
        if self.config.use_scalable_indexing:
            features.append("scalable_indexing")
            
        return features


# Convenience functions for backward compatibility
def create_memory_manager(mode: str = "scalable", **kwargs) -> L2MemoryManager:
    """Create memory manager with specified mode"""
    mode_map = {
        "basic": MemoryMode.BASIC,
        "enhanced": MemoryMode.ENHANCED,
        "scalable": MemoryMode.SCALABLE,
        "graph_centric": MemoryMode.GRAPH_CENTRIC
    }
    
    memory_mode = mode_map.get(mode.lower(), MemoryMode.SCALABLE)
    config = MemoryConfig.from_mode(memory_mode, **kwargs)
    return L2MemoryManager(config)


# Aliases for backward compatibility
EnhancedL2MemoryManager = L2MemoryManager
L2EnhancedScalableMemory = L2MemoryManager
GraphCentricMemoryManager = L2MemoryManager
Memory = L2MemoryManager  # Common alias