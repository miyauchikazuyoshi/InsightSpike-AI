"""
Layer 2: Memory Manager
======================

Graph-centric episodic memory management system (Hippocampus + Locus Coeruleus analog).
Consolidates all memory management variants into a single configurable class.

DIAG: When INSIGHTSPIKE_DIAG_IMPORT=1, lightweight import progress markers are printed
to isolate import-time stalls. Optionally skip heavy deps with INSIGHTSPIKE_SKIP_ST=1.
"""

import os as _os
_L2_DIAG = _os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1'
if _L2_DIAG:
    print('[layer2_memory_manager] module import start', flush=True)

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import vector index factory for backend-agnostic implementation
if _L2_DIAG:
    print('[layer2_memory_manager] before VectorIndexFactory import', flush=True)
from ...vector_index import VectorIndexFactory
if _L2_DIAG:
    print('[layer2_memory_manager] after VectorIndexFactory import', flush=True)

# Optional import with fallback
if _L2_DIAG and _os.getenv('INSIGHTSPIKE_SKIP_ST', '1') == '1':
    # Skip heavy sentence_transformers import during diagnostics unless explicitly allowed
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    if _L2_DIAG:
        print('[layer2_memory_manager] sentence_transformers skipped (diagnostic fast-path)', flush=True)
else:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        if _L2_DIAG:
            print('[layer2_memory_manager] sentence_transformers imported', flush=True)
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        SentenceTransformer = None  # type: ignore
        if _L2_DIAG:
            print('[layer2_memory_manager] sentence_transformers not available', flush=True)

if _L2_DIAG:
    print('[layer2_memory_manager] before config import', flush=True)
from ...config import get_config
if _L2_DIAG:
    print('[layer2_memory_manager] after config import', flush=True)
from ...core.episode import Episode
if _L2_DIAG:
    print('[layer2_memory_manager] after episode import', flush=True)
_LITE_OR_MIN = _os.getenv('INSIGHTSPIKE_LITE_MODE') == '1' or _os.getenv('INSIGHTSPIKE_MIN_IMPORT') == '1'
if _LITE_OR_MIN and _L2_DIAG:
    print('[layer2_memory_manager] skipping embedder import (lite/min mode)', flush=True)

if not _LITE_OR_MIN or _os.getenv('INSIGHTSPIKE_ENABLE_FULL_EMBEDDING') == '1':
    if _L2_DIAG:
        print('[layer2_memory_manager] importing embedder (full mode)', flush=True)
    from ...processing.embedder import EmbeddingManager  # type: ignore
    if _L2_DIAG:
        print('[layer2_memory_manager] after embedder import', flush=True)
else:
    # Provide minimal stub EmbeddingManager to satisfy downstream references
    if _L2_DIAG:
        print('[layer2_memory_manager] defining EmbeddingManager stub', flush=True)
    class EmbeddingManager:  # type: ignore
        def __init__(self, *a, **kw):
            self.model_name = 'stub'
            self.dimension = 384
        def get_model(self):
            return self
        def encode(self, texts, **kw):
            if isinstance(texts, str):
                texts = [texts]
            import numpy as _np
            return _np.zeros((len(texts), 384), dtype=_np.float32)
        def get_embedding(self, text: str):
            import numpy as _np
            return _np.zeros(384, dtype=_np.float32)

# Optional components - not yet implemented
# from ..memory.importance_scorer import ImportanceScorer
# from ..memory.memory_index import MemoryIndex
from .scalable_graph_builder import ScalableGraphBuilder
if _L2_DIAG:
    print('[layer2_memory_manager] after scalable_graph_builder import', flush=True)

logger = logging.getLogger(__name__)


class MemoryMode(Enum):
    """Different operation modes for memory management"""

    BASIC = "basic"  # Standard C-value based
    ENHANCED = "enhanced"  # Graph-aware with conflicts
    SCALABLE = "scalable"  # Optimized for large datasets
    GRAPH_CENTRIC = "graph_centric"  # Pure graph-based (no C-values)


@dataclass
class MemoryConfig:
    """Unified configuration for memory manager"""

    # Mode selection
    mode: MemoryMode = MemoryMode.SCALABLE

    # Core settings
    embedding_dim: int = 384
    max_episodes: int = 10000

    # Memory management settings
    enable_aging: bool = True
    aging_factor: float = 0.95  # Decay factor per day
    min_age_days: int = 7  # Don't age episodes younger than this
    max_age_days: int = 90  # Maximum age before automatic pruning
    prune_on_overflow: bool = True  # Auto-prune when reaching max_episodes

    # Feature toggles
    use_c_values: bool = True
    use_graph_integration: bool = False
    use_conflict_detection: bool = False
    use_importance_scoring: bool = False
    use_scalable_indexing: bool = True
    use_hierarchical_graph: bool = False

    # Index settings
    vector_search_backend: str = "auto"  # auto, numpy, faiss

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
            config.vector_search_backend = "auto"
            config.cache_embeddings = True
            # Provide an IVF style FAISS default hint for large scale (tests expect)
            if not hasattr(config, 'faiss_index_type'):
                setattr(config, 'faiss_index_type', 'IVF')

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
    - Backend-agnostic vector search for efficient retrieval
    - Configurable modes: Basic, Enhanced, Scalable, Graph-Centric
    - Episode management: merge, split, prune operations
    - Transitioning from C-values to graph-based importance
    """

    def __init__(
        self, config: Optional[Union[MemoryConfig, Dict[str, Any], Any]] = None
    ):
        """Initialize with unified configuration"""
        if _L2_DIAG:
            print('[layer2_memory_manager] L2MemoryManager.__init__ start', flush=True)
        # Handle different config types
        if isinstance(config, MemoryConfig):
            self.config = config
        elif isinstance(config, dict):
            # Create MemoryConfig from dict
            memory_config = config.get("memory", {})
            self.config = MemoryConfig(
                mode=MemoryMode(memory_config.get("mode", "scalable")),
                embedding_dim=memory_config.get("embedding_dim", 384),
                max_episodes=memory_config.get("max_episodes", 10000),
                use_c_values=memory_config.get("use_c_values", True),
                use_graph_integration=memory_config.get("use_graph_integration", False),
                use_scalable_indexing=memory_config.get("use_scalable_indexing", True),
                batch_size=memory_config.get("batch_size", 32),
                cache_embeddings=memory_config.get("cache_embeddings", True),
            )
            # Inject default vector index parameters if absent (for tests / normalization parity)
            if not hasattr(self.config, 'faiss_index_type'):
                setattr(self.config, 'faiss_index_type', memory_config.get('faiss_index_type', 'FlatL2'))
            if not hasattr(self.config, 'metric'):
                setattr(self.config, 'metric', memory_config.get('metric', 'l2'))
            # Track which were defaults
            applied = []
            if 'faiss_index_type' not in memory_config:
                applied.append('faiss_index_type')
            if 'metric' not in memory_config:
                applied.append('metric')
            if not hasattr(self.config, 'defaults_applied'):
                setattr(self.config, 'defaults_applied', applied)
            else:
                try:
                    # Extend existing list if present
                    current = getattr(self.config, 'defaults_applied')
                    for item in applied:
                        if item not in current:
                            current.append(item)
                except Exception:
                    pass
        elif hasattr(config, "memory"):
            # Handle Pydantic InsightSpikeConfig or similar
            m = config.memory
            self.config = MemoryConfig(
                # capacities and limits
                max_episodes=getattr(m, "max_episodes", 10000),
                batch_size=getattr(m, "batch_size", 32),
            )
            # Map commonly used toggles and thresholds if present
            for attr in (
                "use_c_values",
                "use_graph_integration",
                "use_scalable_indexing",
                "cache_embeddings",
                "vector_search_backend",
                "similarity_threshold",
                "embedding_dim",
            ):
                if hasattr(m, attr):
                    setattr(self.config, attr, getattr(m, attr))
            # Vector index params
            if hasattr(m, "faiss_index_type"):
                setattr(self.config, "faiss_index_type", getattr(m, "faiss_index_type"))
            if hasattr(m, "metric"):
                setattr(self.config, "metric", getattr(m, "metric"))
        else:
            self.config = MemoryConfig()

        # Core components
        self.episodes: List[Episode] = []
        self.embedding_model = None
        self.vector_index = None
        self.graph_builder = None
        self.importance_scorer = None

        # State tracking
        self.embedding_cache = {} if self.config.cache_embeddings else None
        self.last_graph_update = 0
        self.initialized = False

        self._setup_components()
        if _L2_DIAG:
            print('[layer2_memory_manager] L2MemoryManager.__init__ end (components set up)', flush=True)

    def _setup_components(self):
        """Setup components based on configuration"""
        if _L2_DIAG:
            print('[layer2_memory_manager] _setup_components start', flush=True)
        # Embedding model
        try:
            if _L2_DIAG:
                print('[layer2_memory_manager] loading EmbeddingManager...', flush=True)
            self.embedding_model = EmbeddingManager()
            # EmbeddingManager doesn't have get_embedding_dim, use default
            if hasattr(self.embedding_model, "model") and hasattr(
                self.embedding_model.model, "get_sentence_embedding_dimension"
            ):
                self.config.embedding_dim = (
                    self.embedding_model.model.get_sentence_embedding_dimension()
                )
            logger.info(f"Using embedding dimension: {self.config.embedding_dim}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            if _L2_DIAG:
                print(f'[layer2_memory_manager] EmbeddingManager load failed: {e}', flush=True)

        # Vector index
        self._setup_vector_index()
        if _L2_DIAG:
            print('[layer2_memory_manager] vector index ready', flush=True)

        # Graph builder (if needed)
        if self.config.use_graph_integration:
            # ScalableGraphBuilder uses config object, not individual params
            graph_config = type(
                "Config",
                (),
                {
                    "graph": type(
                        "Graph",
                        (),
                        {"similarity_threshold": self.config.similarity_threshold},
                    )(),
                    "scalable_graph": type(
                        "ScalableGraph", (), {"top_k_neighbors": 10, "batch_size": 1000}
                    )(),
                },
            )()
            self.graph_builder = ScalableGraphBuilder(config=graph_config)
            logger.info("Graph integration enabled")
            if _L2_DIAG:
                print('[layer2_memory_manager] graph integration enabled', flush=True)

        # Importance scorer (if needed)
        if self.config.use_importance_scoring:
            # self.importance_scorer = ImportanceScorer()  # Not yet implemented
            self.importance_scorer = None
            logger.info("Importance scoring requested but not yet implemented")
            if _L2_DIAG:
                print('[layer2_memory_manager] importance scoring placeholder', flush=True)

        self.initialized = True
        if _L2_DIAG:
            print('[layer2_memory_manager] _setup_components end', flush=True)

    def _setup_vector_index(self):
        """Setup vector index based on configuration"""
        dim = self.config.embedding_dim

        # Get vector search backend from config
        backend = "auto"
        if hasattr(self.config, "vector_search_backend"):
            backend = self.config.vector_search_backend
        elif hasattr(self.config, "use_faiss") and not self.config.use_faiss:
            backend = "numpy"

        # Initialize vector index using factory
        self.vector_index = VectorIndexFactory.create_index(
            dimension=dim,
            index_type=backend
        )
        logger.info(f"Using {type(self.vector_index).__name__} for vector search")

    def add_episode(self, text: str, metadata: Optional[Dict] = None) -> int:
        """Add episode (graph-centric interface)"""
        c_value = metadata.get("c_value", 0.5) if metadata else 0.5
        return self.store_episode(text, c_value, metadata)

    def store_episode(
        self, text: str, c_value: float = 0.5, metadata: Optional[Dict] = None
    ) -> int:
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
                metadata=metadata or {},
            )

            # Mode-specific processing
            if self.config.mode == MemoryMode.GRAPH_CENTRIC:
                # Calculate graph-based importance
                if self.importance_scorer and len(self.episodes) > 0:
                    importance = self._calculate_graph_importance(episode)
                    episode.metadata["importance"] = importance

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

            # Apply aging periodically (every 100 episodes)
            if len(self.episodes) % 100 == 0:
                self.age_episodes()

            # Enforce size limit if needed
            if len(self.episodes) > self.config.max_episodes:
                self.enforce_size_limit()

            # Update graph if needed
            if self.config.use_graph_integration:
                self._update_graph(episode, episode_idx)

            logger.debug(f"Stored episode {episode_idx}: {text[:50]}...")
            return episode_idx

        except Exception as e:
            import traceback
            logger.error(f"Failed to store episode: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return -1

    def search_episodes(
        self, query: str, k: int = 5, filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant episodes using the configured vector index.

        Returns a list of dicts with keys: text, similarity, relevance, c_value, index, timestamp, metadata.
        Always returns a list (never None).
        """
        if not self.initialized or not self.episodes:
            return []

        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []

            # Search using vector index
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
                    importance = episode.metadata.get("importance", 0.5)
                    relevance = similarity * importance
                else:
                    # Use C-value
                    relevance = similarity * episode.c

                result = {
                    "text": episode.text,
                    "similarity": similarity,
                    "relevance": relevance,
                    "c_value": episode.c,
                    "index": idx,
                    "timestamp": episode.timestamp,
                    "metadata": episode.metadata,
                }

                results.append(result)

            # Sort by relevance and return top k
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:k]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    # Backward-compat convenience alias used by integration tests
    def retrieve(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Backward-compat convenience alias used by integration tests.
        Delegates to search_episodes(query_text, k=top_k) and always returns a list.
        """
        try:
            res = self.search_episodes(query_text, k=top_k)
            return res or []
        except Exception:
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
        """Merge multiple episodes into one, selecting the most similar pair if indices not specified"""
        if len(indices) < 2:
            # If indices not properly specified, find most similar pair automatically
            most_similar_indices = self._find_most_similar_episodes()
            if most_similar_indices is None:
                return -1
            indices = most_similar_indices

        # Gather episodes
        episodes_to_merge = [
            self.episodes[i] for i in indices if 0 <= i < len(self.episodes)
        ]
        if len(episodes_to_merge) < 2:
            return -1

        # Combine texts with separator for clarity
        combined_text = " [MERGED] ".join([ep.text for ep in episodes_to_merge])

        # Weighted average C-values based on text length
        text_lengths = [len(ep.text) for ep in episodes_to_merge]
        total_length = sum(text_lengths)
        if total_length > 0:
            weighted_c_value = sum(
                ep.c * length / total_length
                for ep, length in zip(episodes_to_merge, text_lengths)
            )
        else:
            weighted_c_value = np.mean([ep.c for ep in episodes_to_merge])

        # Merge metadata
        merged_metadata = {}
        for ep in episodes_to_merge:
            merged_metadata.update(ep.metadata)
        merged_metadata["merged_from"] = indices
        merged_metadata["merge_timestamp"] = time.time()

        # Remove old episodes (in reverse order to maintain indices)
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.episodes):
                del self.episodes[idx]

        # Add merged episode
        return self.store_episode(combined_text, weighted_c_value, merged_metadata)

    def split_episode(self, episode_idx: int, split_points: Optional[List[int]] = None) -> List[int]:
        """Split an episode into multiple smaller episodes
        
        Args:
            episode_idx: Index of episode to split
            split_points: Optional list of character positions to split at.
                         If None, splits on sentence boundaries.
                         
        Returns:
            List of indices of the new episodes created from the split
        """
        if not 0 <= episode_idx < len(self.episodes):
            logger.warning(f"Invalid episode index {episode_idx}")
            return []
            
        episode = self.episodes[episode_idx]
        text = episode.text
        
        # Determine split points
        if split_points is None:
            # Split on sentence boundaries
            import re
            # Simple sentence splitting - looks for period, exclamation, or question mark followed by space
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) <= 1:
                logger.warning(f"Episode {episode_idx} cannot be split (only one sentence)")
                return []
        else:
            # Split at specified character positions
            parts = []
            last_pos = 0
            for pos in sorted(split_points):
                if 0 < pos < len(text):
                    parts.append(text[last_pos:pos].strip())
                    last_pos = pos
            parts.append(text[last_pos:].strip())
            sentences = [p for p in parts if len(p) > 10]
        
        if len(sentences) <= 1:
            logger.warning(f"Episode {episode_idx} cannot be split into meaningful parts")
            return []
            
        # Remove the original episode
        original_c = episode.c
        original_metadata = episode.metadata.copy()
        del self.episodes[episode_idx]
        
        # Create new episodes from sentences
        new_indices = []
        for i, sentence in enumerate(sentences):
            # Slightly reduce C-value for split episodes to indicate uncertainty
            split_c = original_c * 0.9
            
            # Update metadata
            split_metadata = original_metadata.copy()
            split_metadata["split_from_idx"] = episode_idx
            split_metadata["split_part"] = i + 1
            split_metadata["split_total"] = len(sentences)
            split_metadata["split_timestamp"] = time.time()
            
            # Store the new episode
            new_idx = self.store_episode(sentence, split_c, split_metadata)
            if new_idx >= 0:
                new_indices.append(new_idx)
                
        logger.info(f"Split episode {episode_idx} into {len(new_indices)} parts")
        return new_indices

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

    def _find_most_similar_episodes(
        self, num_candidates: int = 10
    ) -> Optional[List[int]]:
        """Find the most similar pair of episodes based on cosine similarity"""
        if len(self.episodes) < 2:
            return None

        # For efficiency, only consider recent episodes
        candidate_indices = list(range(min(num_candidates, len(self.episodes))))

        max_similarity = -1.0
        best_pair = None

        # Compute pairwise similarities
        for i in range(len(candidate_indices)):
            for j in range(i + 1, len(candidate_indices)):
                idx_i = candidate_indices[i]
                idx_j = candidate_indices[j]

                # Compute cosine similarity
                vec_i = self.episodes[idx_i].vec
                vec_j = self.episodes[idx_j].vec

                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)

                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_pair = [idx_i, idx_j]

        if best_pair and max_similarity > 0.8:  # Only merge if highly similar
            logger.info(
                f"Found similar episodes to merge: {best_pair} (similarity: {max_similarity:.3f})"
            )
            return best_pair

        return None

    def age_episodes(self) -> int:
        """Apply time-based aging to episodes and prune old ones"""
        if not self.config.enable_aging:
            return 0

        current_time = time.time()
        aged_count = 0
        to_prune = []

        for i, episode in enumerate(self.episodes):
            age_days = (current_time - episode.timestamp) / (24 * 3600)

            # Skip young episodes
            if age_days < self.config.min_age_days:
                continue

            # Mark very old episodes for pruning
            if age_days > self.config.max_age_days:
                to_prune.append(i)
                continue

            # Apply aging decay to C-value
            decay = self.config.aging_factor ** (age_days - self.config.min_age_days)
            old_c = episode.c
            episode.c = max(0.01, episode.c * decay)  # Keep minimum C-value

            if old_c != episode.c:
                aged_count += 1

        # Prune very old episodes
        for idx in sorted(to_prune, reverse=True):
            del self.episodes[idx]

        if to_prune:
            self._rebuild_index()
            logger.info(f"Pruned {len(to_prune)} episodes due to age")

        logger.info(f"Aged {aged_count} episodes")
        return aged_count + len(to_prune)

    def enforce_size_limit(self) -> int:
        """Enforce maximum episode limit by pruning lowest value episodes"""
        if (
            not self.config.prune_on_overflow
            or len(self.episodes) <= self.config.max_episodes
        ):
            return 0

        # Sort by C-value and age
        episode_scores = []
        current_time = time.time()

        for i, episode in enumerate(self.episodes):
            age_factor = min(
                1.0, (current_time - episode.timestamp) / (30 * 24 * 3600)
            )  # 30 days = 1.0
            score = episode.c * (1 - 0.3 * age_factor)  # 30% weight on age
            episode_scores.append((i, score))

        # Sort by score (lowest first)
        episode_scores.sort(key=lambda x: x[1])

        # Prune 10% of lowest scoring episodes
        prune_count = max(1, int(0.1 * len(self.episodes)))
        prune_count = min(
            prune_count, len(self.episodes) - self.config.max_episodes + 100
        )  # Leave some buffer

        to_prune = [idx for idx, _ in episode_scores[:prune_count]]

        # Remove in reverse order
        for idx in sorted(to_prune, reverse=True):
            del self.episodes[idx]

        self._rebuild_index()
        logger.info(f"Pruned {len(to_prune)} episodes to enforce size limit")
        return len(to_prune)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "mode": self.config.mode.value,
                "features_enabled": self._get_enabled_features(),
            }

        c_values = [ep.c for ep in self.episodes] if self.config.use_c_values else []

        stats = {
            "total_episodes": len(self.episodes),
            "mode": self.config.mode.value,
            "features_enabled": self._get_enabled_features(),
            "index_type": getattr(self.vector_index, '__class__', type(self.vector_index)).__name__ if self.vector_index else "None",
            "embedding_dim": self.config.embedding_dim,
        }

        if c_values:
            stats.update(
                {
                    "c_value_mean": np.mean(c_values),
                    "c_value_std": np.std(c_values),
                    "c_value_min": np.min(c_values),
                    "c_value_max": np.max(c_values),
                }
            )

        if self.config.use_graph_integration and self.graph_builder:
            # ScalableGraphBuilder doesn't have get_current_graph, so we skip graph stats
            # TODO: Implement graph stats retrieval for ScalableGraphBuilder
            stats["graph_nodes"] = len(self.episodes)  # Approximate with episode count
            stats["graph_edges"] = 0  # Unknown without building graph

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
            stacklevel=2,
        )
        if not path:
            config = get_config()
            path = os.path.join(config.paths.data_dir, "layer2_memory.json")

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Prepare data
            data = {
                "mode": self.config.mode.value,
                "episodes": [],
                "config": {
                    "embedding_dim": self.config.embedding_dim,
                    "use_c_values": self.config.use_c_values,
                    "use_graph_integration": self.config.use_graph_integration,
                },
            }

            # Save episodes
            for ep in self.episodes:
                ep_data = {
                    "text": ep.text,
                    "c_value": ep.c,
                    "timestamp": ep.timestamp,
                    "metadata": ep.metadata,
                    "embedding": ep.vec.tolist()
                    if isinstance(ep.vec, np.ndarray)
                    else ep.vec,
                }
                data["episodes"].append(ep_data)

            # Write to file
            with open(path, "w") as f:
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
            stacklevel=2,
        )
        if not path:
            config = get_config()
            path = os.path.join(config.paths.data_dir, "layer2_memory.json")

        if not os.path.exists(path):
            logger.info(f"No saved memory found at {path}")
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Clear current state
            self.episodes.clear()
            self._rebuild_index()

            # Load episodes
            for ep_data in data.get("episodes", []):
                episode = Episode(
                    text=ep_data["text"],
                    vec=np.array(ep_data["embedding"]),
                    c=ep_data.get("c_value", 0.5),
                    timestamp=ep_data.get("timestamp", time.time()),
                    metadata=ep_data.get("metadata", {}),
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
            # EmbeddingManager returns a 2D array from encode, we need 1D
            embeddings = self.embedding_model.encode(text)
            if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2:
                embedding = embeddings[0]  # Get first (and only) embedding
            else:
                embedding = embeddings

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
        """Update vector index with new episode"""
        if self.vector_index is None:
            return

        # Add vector to index
        try:
            self.vector_index.add(np.array([episode.vec]))
        except Exception as e:
            logger.error(f"Failed to add vector to index: {e}")

    def _search_index(
        self, query_vec: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search vector index"""
        if self.vector_index is None:
            return np.array([]), np.array([])

        try:
            # Search
            distances, indices = self.vector_index.search(np.array([query_vec]), k)
            return distances[0], indices[0]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return np.array([]), np.array([])

    def _rebuild_index(self):
        """Rebuild vector index from scratch"""
        self._setup_vector_index()

        if not self.episodes:
            return

        # Add all vectors
        vectors = np.array([ep.vec for ep in self.episodes])

        # Add vectors to new index
        try:
            self.vector_index.add(vectors)
            logger.info(f"Rebuilt index with {len(self.episodes)} episodes")
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")

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
                    "text": ep.text,
                    "embedding": ep.vec,
                    "episode_idx": i,
                    "metadata": ep.metadata,
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
            sim = np.dot(episode.vec, other.vec) / (
                np.linalg.norm(episode.vec) * np.linalg.norm(other.vec)
            )
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
        negations = ["not", "no", "never", "neither", "nor", "nothing"]

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
            logger.debug(
                f"Conflict with episode {idx}: {self.episodes[idx].text[:50]}..."
            )

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
        "graph_centric": MemoryMode.GRAPH_CENTRIC,
    }

    memory_mode = mode_map.get(mode.lower(), MemoryMode.SCALABLE)
    config = MemoryConfig.from_mode(memory_mode, **kwargs)
    return L2MemoryManager(config)


# Aliases for backward compatibility
EnhancedL2MemoryManager = L2MemoryManager
L2EnhancedScalableMemory = L2MemoryManager
GraphCentricMemoryManager = L2MemoryManager
Memory = L2MemoryManager  # Common alias
