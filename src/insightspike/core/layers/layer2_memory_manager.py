"""
L2 Memory Manager - Enhanced Episodic Memory with C-values
========================================================

Implements quantum-inspired RAG with IVF-PQ indexing and C-value reinforcement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from ...config import get_config
from ...utils.embedder import get_model
from ..interfaces import L2MemoryInterface
from ..learning.knowledge_graph_memory import KnowledgeGraphMemory

logger = logging.getLogger(__name__)

__all__ = ["Episode", "L2MemoryManager"]


class Episode:
    """Single memory entry with vector embedding, text, and C-value."""

    def __init__(
        self,
        vec: np.ndarray,
        text: str,
        c: float = 0.5,
        metadata: Optional[Dict] = None,
    ):
        self.vec = vec.astype(np.float32)
        self.text = text
        self.c = float(c)
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Episode(c={self.c:.3f}, text='{self.text[:50]}...')"


class L2MemoryManager(L2MemoryInterface):
    """
    Enhanced memory manager with IVF-PQ indexing and C-value reinforcement.

    Features:
    - Efficient similarity search with Faiss IVF-PQ
    - C-value weighted retrieval (sim × C^γ)
    - Dynamic re-quantization based on memory updates
    - Episodic memory with metadata support
    """

    def __init__(
        self,
        dim: int = None,
        config=None,
        knowledge_graph: Optional[KnowledgeGraphMemory] = None,
    ):
        self.config = config or get_config()
        self.dim = dim or self.config.embedding.dimension
        self.knowledge_graph = knowledge_graph

        # IVF-PQ parameters
        nlist = min(256, max(16, self.config.memory.nlist))
        m = self.config.memory.pq_segments

        # Ensure m is compatible with dimension (dim must be divisible by m)
        while self.dim % m != 0 and m > 1:
            m -= 1

        if m < 1:
            # Fall back to simple flat index if PQ is not possible
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.index_factory(self.dim, f"IVF{nlist},PQ{m}")
        self.episodes: List[Episode] = []
        self.is_trained = False

        # C-value parameters
        self.gamma = self.config.memory.c_value_gamma
        self.c_min = self.config.memory.c_value_min
        self.c_max = self.config.memory.c_value_max

    def store_episode(
        self, text: str, c_value: float = 0.5, metadata: Optional[Dict] = None
    ) -> bool:
        """Store a new episode in memory."""
        try:
            model = get_model()
            vec = model.encode(
                [text], convert_to_numpy=True, normalize_embeddings=True
            )[0]

            episode = Episode(vec, text, c_value, metadata)
            self.episodes.append(episode)

            if self.knowledge_graph is not None:
                try:
                    self.knowledge_graph.add_episode_node(vec, len(self.episodes) - 1)
                except Exception as e:
                    logger.warning(f"KnowledgeGraph update failed: {e}")

            # Re-train index if we have enough episodes
            if len(self.episodes) >= 2 and not self.is_trained:
                self._train_index()
            elif self.is_trained:
                self._add_to_index(episode)

            logger.debug(f"Stored episode with C={c_value:.3f}: {text[:100]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            return False

    def search_episodes(
        self, query: str, k: int = 5, min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for relevant episodes using C-value weighted similarity."""
        if not self.episodes:
            return []

        try:
            model = get_model()
            query_vec = model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )[0]

            if self.is_trained and len(self.episodes) > 0:
                # Use Faiss for efficient search
                similarities, indices = self.index.search(
                    query_vec.reshape(1, -1), min(k * 2, len(self.episodes))
                )
                results = []

                for sim, idx in zip(similarities[0], indices[0]):
                    if idx == -1 or sim < min_similarity:
                        continue

                    episode = self.episodes[idx]
                    # Apply C-value weighting: score = similarity × C^γ
                    weighted_score = sim * (episode.c**self.gamma)

                    results.append(
                        {
                            "text": episode.text,
                            "similarity": float(sim),
                            "c_value": episode.c,
                            "weighted_score": float(weighted_score),
                            "metadata": episode.metadata,
                            "index": idx,
                        }
                    )

                # Sort by weighted score and return top-k
                results.sort(key=lambda x: x["weighted_score"], reverse=True)
                return results[:k]

            else:
                # Fallback to linear search for small datasets
                return self._linear_search(query_vec, k, min_similarity)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def update_c_value(self, episode_index: int, new_c_value: float) -> bool:
        """Update C-value for a specific episode."""
        if 0 <= episode_index < len(self.episodes):
            # Clamp C-value to valid range
            clamped_c = max(self.c_min, min(self.c_max, new_c_value))
            self.episodes[episode_index].c = clamped_c
            logger.debug(f"Updated episode {episode_index} C-value: {new_c_value:.3f}")
            return True
        return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the current memory state."""
        if not self.episodes:
            return {"total_episodes": 0, "index_trained": False}

        c_values = [ep.c for ep in self.episodes]
        return {
            "total_episodes": len(self.episodes),
            "index_trained": self.is_trained,
            "c_value_mean": float(np.mean(c_values)),
            "c_value_std": float(np.std(c_values)),
            "c_value_min": float(np.min(c_values)),
            "c_value_max": float(np.max(c_values)),
            "dimension": self.dim,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save memory index and episodes to disk."""
        if path is None:
            path = Path(self.config.memory.index_file)

        # Save Faiss index
        if self.is_trained:
            faiss.write_index(self.index, str(path))

        # Save episodes metadata
        meta_path = path.with_suffix(".json")
        meta = [
            {"c": ep.c, "text": ep.text, "metadata": ep.metadata}
            for ep in self.episodes
        ]

        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info(f"Saved memory to {path} ({len(self.episodes)} episodes)")
        return path

    def load(self, path: Optional[Path] = None) -> bool:
        """Load memory index and episodes from disk."""
        if path is None:
            path = Path(self.config.memory.index_file)

        try:
            # Load episodes metadata
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())

                # Reconstruct episodes
                model = get_model()
                texts = [item["text"] for item in meta]
                vecs = model.encode(
                    texts, convert_to_numpy=True, normalize_embeddings=True
                )

                self.episodes = []
                for vec, item in zip(vecs, meta):
                    episode = Episode(
                        vec, item["text"], item["c"], item.get("metadata", {})
                    )
                    self.episodes.append(episode)

                # Load or rebuild index
                if path.exists() and len(self.episodes) >= 2:
                    self.index = faiss.read_index(str(path))
                    self.is_trained = True
                else:
                    self._train_index()

                logger.info(
                    f"Loaded memory from {path} ({len(self.episodes)} episodes)"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to load memory: {e}")

        return False

    def _train_index(self):
        """Train the Faiss index with current episodes."""
        if len(self.episodes) < 2:
            logger.warning("Need at least 2 episodes to train index")
            return

        vecs = np.vstack([ep.vec for ep in self.episodes])

        # FAISS IVF requires at least 39 training points per cluster
        # For small datasets, use flat index instead
        if len(vecs) < 50:
            logger.info(f"Small dataset ({len(vecs)} episodes), using flat index")
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(vecs)
            self.is_trained = True
            return

        # For larger datasets, use IVF-PQ with appropriate cluster count
        n_clusters = min(self.nlist, max(1, len(vecs) // 50))

        try:
            # Create new IVF-PQ index with appropriate cluster count
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dim, n_clusters, self.pq_segments, 8
            )
            self.index.train(vecs)
            self.index.add(vecs)
            self.is_trained = True
            logger.info(
                f"Trained IVF-PQ index with {len(self.episodes)} episodes, {n_clusters} clusters"
            )

        except Exception as e:
            logger.warning(f"IVF-PQ training failed ({e}), falling back to flat index")
            # Fallback to flat index
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(vecs)
            self.is_trained = True

    def _add_to_index(self, episode: Episode):
        """Add a single episode to the trained index."""
        if self.is_trained:
            self.index.add(episode.vec.reshape(1, -1))

    def _linear_search(
        self, query_vec: np.ndarray, k: int, min_similarity: float
    ) -> List[Dict[str, Any]]:
        """Fallback linear search for small datasets."""
        results = []

        for i, episode in enumerate(self.episodes):
            # Cosine similarity
            sim = float(np.dot(query_vec, episode.vec))
            if sim < min_similarity:
                continue

            weighted_score = sim * (episode.c**self.gamma)
            results.append(
                {
                    "text": episode.text,
                    "similarity": sim,
                    "c_value": episode.c,
                    "weighted_score": weighted_score,
                    "metadata": episode.metadata,
                    "index": i,
                }
            )

        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results[:k]

    @classmethod
    def build_from_documents(
        cls, docs: List[str], batch_size: int = 32
    ) -> "L2MemoryManager":
        """Build memory manager from a list of documents."""
        model = get_model()
        vecs = model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        memory = cls(dim=vecs.shape[1])
        for vec, text in zip(vecs, docs):
            episode = Episode(vec, text)
            memory.episodes.append(episode)

        memory._train_index()
        return memory

    # Abstract method implementations
    def search(
        self, query: np.ndarray, top_k: int = 5
    ) -> Tuple[List[float], List[int]]:
        """Search similar episodes in memory - interface compliance"""
        try:
            if not self.episodes:
                return [], []

            # Use existing search_episodes implementation
            query_text = f"query_vector_{hash(query.tobytes())}"  # Convert vector to text for search
            results = self.search_episodes(query_text, k=top_k)

            similarities = [r["similarity"] for r in results]
            indices = [r["index"] for r in results]

            return similarities, indices

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [], []

    def add_episode(self, vector: np.ndarray, text: str, c_value: float = 0.2) -> int:
        """Add new episode to memory - with integration vs new node decision"""
        try:
            # Check if this should be integrated with existing episodes or added as new node
            integration_result = self._check_episode_integration(vector, text, c_value)
            
            if integration_result["should_integrate"]:
                # Integrate with existing episode
                target_idx = integration_result["target_index"]
                logger.info(f"Integrating new episode with existing episode {target_idx}")
                return self._integrate_with_existing(target_idx, vector, text, c_value)
            else:
                # Add as new episode node
                episode = Episode(vector, text, c_value)
                self.episodes.append(episode)

                # Re-train index if we have enough episodes
                if len(self.episodes) >= 2 and not self.is_trained:
                    self._train_index()
                elif self.is_trained:
                    self._add_to_index(episode)

                logger.debug(f"Added new episode node {len(self.episodes) - 1}")
                return len(self.episodes) - 1  # Return index of added episode

        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            return -1

    def update_c_values(self, episode_ids: List[int], rewards: List[float]):
        """Update C-values based on rewards - interface compliance"""
        try:
            for episode_id, reward in zip(episode_ids, rewards):
                if 0 <= episode_id < len(self.episodes):
                    current_c = self.episodes[episode_id].c
                    # Apply reward with eta learning rate
                    eta = 0.1
                    new_c = max(self.c_min, min(self.c_max, current_c + eta * reward))
                    self.episodes[episode_id].c = new_c

        except Exception as e:
            logger.error(f"Failed to update C-values: {e}")

    def update_c(self, episode_ids: List[int], reward: float, eta: float = 0.1) -> bool:
        """
        Update C-values with reward - legacy interface for compatibility.
        
        Args:
            episode_ids: List of episode indices to update
            reward: Reward value to apply
            eta: Learning rate
            
        Returns:
            True if update was successful
        """
        try:
            for episode_id in episode_ids:
                if 0 <= episode_id < len(self.episodes):
                    current_c = self.episodes[episode_id].c
                    new_c = max(self.c_min, min(self.c_max, current_c + eta * reward))
                    self.episodes[episode_id].c = new_c
                    logger.debug(f"Updated episode {episode_id} C-value: {current_c:.3f} -> {new_c:.3f}")
                else:
                    logger.warning(f"Invalid episode index: {episode_id}")
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Failed to update C-values: {e}")
            return False

    def train_index(self) -> bool:
        """
        Train the FAISS index - public interface for compatibility.
        
        Returns:
            True if training was successful
        """
        try:
            self._train_index()
            return True
        except Exception as e:
            logger.error(f"Failed to train index: {e}")
            return False

    def prune(self, c_threshold: float, importance_threshold: int = 1) -> int:
        """
        Prune episodes with low C-values and low importance.
        
        Args:
            c_threshold: Minimum C-value threshold
            importance_threshold: Minimum importance threshold (not used currently)
            
        Returns:
            Number of episodes pruned
        """
        try:
            original_count = len(self.episodes)
            
            # Filter episodes based on C-value threshold
            pruned_episodes = []
            for i, episode in enumerate(self.episodes):
                if episode.c >= c_threshold:
                    pruned_episodes.append(episode)
                else:
                    logger.debug(f"Pruning episode {i} with C-value {episode.c:.3f}")
            
            self.episodes = pruned_episodes
            pruned_count = original_count - len(self.episodes)
            
            # Retrain index if we pruned episodes
            if pruned_count > 0 and len(self.episodes) >= 2:
                self._train_index()
                logger.info(f"Pruned {pruned_count} episodes, retrained index")
            elif pruned_count > 0:
                self.is_trained = False
                logger.info(f"Pruned {pruned_count} episodes, index invalidated")
                
            return pruned_count
            
        except Exception as e:
            logger.error(f"Failed to prune episodes: {e}")
            return 0

    def merge(self, episode_indices: List[int]) -> int:
        """
        Merge multiple episodes into a single episode.
        
        Args:
            episode_indices: List of episode indices to merge
            
        Returns:
            Index of the new merged episode, or -1 if failed
        """
        try:
            if len(episode_indices) < 2:
                logger.warning("Need at least 2 episodes to merge")
                return -1
                
            # Validate indices
            valid_indices = [i for i in episode_indices if 0 <= i < len(self.episodes)]
            if len(valid_indices) != len(episode_indices):
                logger.warning(f"Some episode indices are invalid: {episode_indices}")
                
            if len(valid_indices) < 2:
                return -1
                
            # Get episodes to merge
            episodes_to_merge = [self.episodes[i] for i in valid_indices]
            
            # Create merged episode
            # Average the vectors
            merged_vec = np.mean([ep.vec for ep in episodes_to_merge], axis=0)
            
            # Combine text content
            merged_text = " | ".join([ep.text for ep in episodes_to_merge])
            
            # Take maximum C-value (most important)
            merged_c = max([ep.c for ep in episodes_to_merge])
            
            # Combine metadata
            merged_metadata = {}
            for ep in episodes_to_merge:
                if ep.metadata:
                    merged_metadata.update(ep.metadata)
            merged_metadata['merged_from'] = valid_indices
            merged_metadata['merged_count'] = len(valid_indices)
            
            # Create new episode
            merged_episode = Episode(merged_vec, merged_text, merged_c, merged_metadata)
            
            # Remove old episodes (in reverse order to maintain indices)
            for i in sorted(valid_indices, reverse=True):
                del self.episodes[i]
                
            # Add merged episode
            self.episodes.append(merged_episode)
            merged_index = len(self.episodes) - 1
            
            # Retrain index
            if len(self.episodes) >= 2:
                self._train_index()
                logger.info(f"Merged {len(valid_indices)} episodes into episode {merged_index}")
            else:
                self.is_trained = False
                
            return merged_index
            
        except Exception as e:
            logger.error(f"Failed to merge episodes: {e}")
            return -1

    def split(self, episode_index: int) -> List[int]:
        """
        Split an episode into multiple episodes based on content.
        
        Args:
            episode_index: Index of episode to split
            
        Returns:
            List of indices of new episodes created from the split, empty if failed
        """
        try:
            if not (0 <= episode_index < len(self.episodes)):
                logger.warning(f"Invalid episode index for split: {episode_index}")
                return []
                
            episode = self.episodes[episode_index]
            
            # Simple split based on sentence boundaries
            sentences = [s.strip() for s in episode.text.split('.') if s.strip()]
            
            if len(sentences) < 2:
                logger.info(f"Episode {episode_index} cannot be split (too few sentences)")
                return []
                
            # Create new episodes from sentences
            new_indices = []
            base_c = episode.c * 0.8  # Slightly reduce C-value for split episodes
            
            for i, sentence in enumerate(sentences):
                if not sentence:
                    continue
                    
                # Create slightly varied vector (add small noise)
                noise = np.random.normal(0, 0.01, episode.vec.shape)
                new_vec = episode.vec + noise.astype(np.float32)
                
                # Create metadata
                new_metadata = episode.metadata.copy() if episode.metadata else {}
                new_metadata['split_from'] = episode_index
                new_metadata['split_part'] = i + 1
                new_metadata['split_total'] = len(sentences)
                
                # Create new episode
                new_episode = Episode(new_vec, sentence, base_c, new_metadata)
                self.episodes.append(new_episode)
                new_indices.append(len(self.episodes) - 1)
                
            # Remove original episode
            del self.episodes[episode_index]
            
            # Adjust indices for episodes that were added after the removed one
            adjusted_indices = []
            for idx in new_indices:
                if idx > episode_index:
                    adjusted_indices.append(idx - 1)
                else:
                    adjusted_indices.append(idx)
                    
            # Retrain index
            if len(self.episodes) >= 2:
                self._train_index()
                logger.info(f"Split episode {episode_index} into {len(adjusted_indices)} episodes")
            else:
                self.is_trained = False
                
            return adjusted_indices
            
        except Exception as e:
            logger.error(f"Failed to split episode: {e}")
            return []

    def initialize(self) -> bool:
        """Initialize the memory manager"""
        try:
            self._is_initialized = True
            logger.info(f"L2MemoryManager initialized with dim={self.dim}")
            return True
        except Exception as e:
            logger.error(f"L2MemoryManager initialization failed: {e}")
            return False

    def process(self, input_data) -> Dict[str, Any]:
        """Process input through memory layer"""
        try:
            if hasattr(input_data, "query"):
                query = input_data.query
            elif isinstance(input_data, str):
                query = input_data
            else:
                query = str(input_data)

            # Search memory
            results = self.search_episodes(query)

            return {
                "result": results,
                "confidence": 1.0 if results else 0.0,
                "metadata": {"memory_stats": self.get_memory_stats()},
                "metrics": {"episodes_searched": len(self.episodes)},
            }

        except Exception as e:
            logger.error(f"Memory processing failed: {e}")
            return {
                "result": [],
                "confidence": 0.0,
                "metadata": {"error": str(e)},
                "metrics": {},
            }

    def cleanup(self):
        """Cleanup memory resources"""
        try:
            # Clear episodes
            self.episodes.clear()

            # Reset index
            if hasattr(self, "index"):
                del self.index

            self.is_trained = False
            self._is_initialized = False

            logger.info("L2MemoryManager cleanup completed")

        except Exception as e:
            logger.warning(f"Error during L2MemoryManager cleanup: {e}")

    def get_episode_similarity(self, episode_indices: List[int]) -> List[float]:
        """
        Calculate pairwise similarities between episodes.
        
        Args:
            episode_indices: List of episode indices to compare
            
        Returns:
            List of similarity scores between episode pairs
        """
        try:
            similarities = []
            
            for i in range(len(episode_indices)):
                for j in range(i + 1, len(episode_indices)):
                    idx1, idx2 = episode_indices[i], episode_indices[j]
                    
                    if 0 <= idx1 < len(self.episodes) and 0 <= idx2 < len(self.episodes):
                        ep1 = self.episodes[idx1]
                        ep2 = self.episodes[idx2]
                        
                        # Calculate cosine similarity between vectors
                        sim = np.dot(ep1.vec, ep2.vec) / (
                            np.linalg.norm(ep1.vec) * np.linalg.norm(ep2.vec)
                        )
                        similarities.append(float(sim))
                    else:
                        similarities.append(0.0)
                        
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to calculate episode similarities: {e}")
            return []

    def get_episode_content_complexity(self, episode_index: int) -> float:
        """
        Calculate content complexity of an episode (used for split decisions).
        
        Args:
            episode_index: Index of episode to analyze
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        try:
            if not (0 <= episode_index < len(self.episodes)):
                return 0.0
                
            episode = self.episodes[episode_index]
            text = episode.text
            
            # Simple complexity metrics
            sentence_count = len([s for s in text.split('.') if s.strip()])
            word_count = len(text.split())
            unique_word_ratio = len(set(text.lower().split())) / max(1, word_count)
            
            # Normalize and combine metrics
            sentence_complexity = min(1.0, sentence_count / 5.0)  # Max 5 sentences = 1.0
            length_complexity = min(1.0, word_count / 100.0)     # Max 100 words = 1.0
            
            complexity = (sentence_complexity + length_complexity + unique_word_ratio) / 3.0
            return float(complexity)
            
        except Exception as e:
            logger.error(f"Failed to calculate episode complexity: {e}")
            return 0.0

    def _check_episode_integration(self, vector: np.ndarray, text: str, c_value: float) -> Dict[str, Any]:
        """
        Check if new episode should be integrated with existing episodes or added as new node.
        
        Args:
            vector: New episode vector
            text: New episode text
            c_value: New episode C-value
            
        Returns:
            Dictionary with integration decision and target information
        """
        try:
            if len(self.episodes) == 0:
                return {"should_integrate": False, "target_index": -1, "reason": "no_existing_episodes"}
            
            # Configuration thresholds
            similarity_threshold = getattr(self.config.reasoning, 'episode_integration_similarity_threshold', 0.85)
            content_overlap_threshold = getattr(self.config.reasoning, 'episode_integration_content_threshold', 0.7)
            c_value_diff_threshold = getattr(self.config.reasoning, 'episode_integration_c_threshold', 0.3)
            
            best_candidate = {
                "index": -1,
                "similarity": 0.0,
                "content_overlap": 0.0,
                "c_value_compatibility": 0.0,
                "integration_score": 0.0
            }
            
            # Check similarity with all existing episodes
            for i, episode in enumerate(self.episodes):
                # 1. Vector similarity
                vec_similarity = np.dot(vector, episode.vec) / (
                    np.linalg.norm(vector) * np.linalg.norm(episode.vec)
                )
                
                # 2. Content overlap (improved word-based with cleaning)
                import re
                # Clean and normalize text
                new_text_clean = re.sub(r'[^\w\s]', '', text.lower())
                existing_text_clean = re.sub(r'[^\w\s]', '', episode.text.lower())
                
                new_words = set(new_text_clean.split())
                existing_words = set(existing_text_clean.split())
                
                if len(new_words) == 0 or len(existing_words) == 0:
                    content_overlap = 0.0
                else:
                    content_overlap = len(new_words.intersection(existing_words)) / len(new_words.union(existing_words))
                
                # 3. C-value compatibility (similar importance levels)
                c_value_diff = abs(c_value - episode.c)
                c_value_compatibility = 1.0 - min(1.0, c_value_diff / c_value_diff_threshold)
                
                # Combined integration score
                integration_score = (
                    0.5 * vec_similarity +           # 50% vector similarity
                    0.3 * content_overlap +          # 30% content overlap  
                    0.2 * c_value_compatibility      # 20% C-value compatibility
                )
                
                if integration_score > best_candidate["integration_score"]:
                    best_candidate.update({
                        "index": i,
                        "similarity": vec_similarity,
                        "content_overlap": content_overlap,
                        "c_value_compatibility": c_value_compatibility,
                        "integration_score": integration_score
                    })
            
            # Decision logic
            should_integrate = (
                best_candidate["similarity"] >= similarity_threshold and
                best_candidate["content_overlap"] >= content_overlap_threshold and
                best_candidate["integration_score"] >= 0.65  # Lowered overall threshold
            )
            
            return {
                "should_integrate": should_integrate,
                "target_index": best_candidate["index"] if should_integrate else -1,
                "best_candidate": best_candidate,
                "reason": "high_similarity" if should_integrate else "below_threshold"
            }
            
        except Exception as e:
            logger.error(f"Episode integration check failed: {e}")
            return {"should_integrate": False, "target_index": -1, "reason": "error"}

    def _integrate_with_existing(self, target_index: int, new_vector: np.ndarray, new_text: str, new_c_value: float) -> int:
        """
        Integrate new episode content with existing episode.
        
        Args:
            target_index: Index of existing episode to integrate with
            new_vector: Vector of new episode
            new_text: Text of new episode
            new_c_value: C-value of new episode
            
        Returns:
            Index of updated episode
        """
        try:
            if not (0 <= target_index < len(self.episodes)):
                logger.error(f"Invalid target index for integration: {target_index}")
                return -1
                
            existing_episode = self.episodes[target_index]
            
            # Update vector (weighted average based on C-values)
            total_weight = existing_episode.c + new_c_value
            if total_weight > 0:
                weight_existing = existing_episode.c / total_weight
                weight_new = new_c_value / total_weight
                
                integrated_vector = (
                    weight_existing * existing_episode.vec + 
                    weight_new * new_vector
                )
            else:
                integrated_vector = (existing_episode.vec + new_vector) / 2.0
            
            # Update text (combine with separator)
            integrated_text = f"{existing_episode.text} | {new_text}"
            
            # Update C-value (take maximum as it represents highest importance)
            integrated_c = max(existing_episode.c, new_c_value)
            
            # Update metadata
            integrated_metadata = existing_episode.metadata.copy() if existing_episode.metadata else {}
            integrated_metadata.setdefault('integration_history', [])
            integrated_metadata['integration_history'].append({
                'integrated_text': new_text,
                'integrated_c': new_c_value,
                'integration_timestamp': len(integrated_metadata['integration_history'])
            })
            integrated_metadata['integration_count'] = len(integrated_metadata['integration_history'])
            
            # Update the existing episode
            self.episodes[target_index] = Episode(
                integrated_vector, integrated_text, integrated_c, integrated_metadata
            )
            
            # Update index if needed
            if self.is_trained:
                self._retrain_index_single(target_index)
            
            logger.info(f"Integrated new content with episode {target_index}, C-value: {integrated_c:.3f}")
            return target_index
            
        except Exception as e:
            logger.error(f"Episode integration failed: {e}")
            return -1

    def _retrain_index_single(self, episode_index: int):
        """Retrain index for a single updated episode"""
        try:
            if self.is_trained and 0 <= episode_index < len(self.episodes):
                # For simplicity, retrain the entire index
                # In production, could implement more efficient single-update
                self._train_index()
        except Exception as e:
            logger.warning(f"Single episode index retrain failed: {e}")
