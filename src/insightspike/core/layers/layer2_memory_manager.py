"""
L2 Memory Manager - Enhanced Episodic Memory with C-values
========================================================

Implements quantum-inspired RAG with IVF-PQ indexing and C-value reinforcement.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

import faiss
import numpy as np

from ..interfaces import L2MemoryInterface
from ...config import get_config
from ...utils.embedder import get_model

logger = logging.getLogger(__name__)

__all__ = ["Episode", "L2MemoryManager"]


class Episode:
    """Single memory entry with vector embedding, text, and C-value."""

    def __init__(self, vec: np.ndarray, text: str, c: float = 0.5, metadata: Optional[Dict] = None):
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

    def __init__(self, dim: int = None, config=None):
        self.config = config or get_config()
        self.dim = dim or self.config.embedding.dimension
        
        # IVF-PQ parameters
        nlist = min(256, max(16, self.config.memory.nlist))
        m = self.config.memory.pq_segments
        
        self.index = faiss.index_factory(self.dim, f"IVF{nlist},PQ{m}")
        self.episodes: List[Episode] = []
        self.is_trained = False
        
        # C-value parameters
        self.gamma = self.config.memory.c_value_gamma
        self.c_min = self.config.memory.c_value_min
        self.c_max = self.config.memory.c_value_max

    def store_episode(self, text: str, c_value: float = 0.5, metadata: Optional[Dict] = None) -> bool:
        """Store a new episode in memory."""
        try:
            model = get_model()
            vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            episode = Episode(vec, text, c_value, metadata)
            self.episodes.append(episode)
            
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

    def search_episodes(self, query: str, k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """Search for relevant episodes using C-value weighted similarity."""
        if not self.episodes:
            return []
            
        try:
            model = get_model()
            query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            if self.is_trained and len(self.episodes) > 0:
                # Use Faiss for efficient search
                similarities, indices = self.index.search(query_vec.reshape(1, -1), min(k * 2, len(self.episodes)))
                results = []
                
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx == -1 or sim < min_similarity:
                        continue
                        
                    episode = self.episodes[idx]
                    # Apply C-value weighting: score = similarity × C^γ
                    weighted_score = sim * (episode.c ** self.gamma)
                    
                    results.append({
                        'text': episode.text,
                        'similarity': float(sim),
                        'c_value': episode.c,
                        'weighted_score': float(weighted_score),
                        'metadata': episode.metadata,
                        'index': idx
                    })
                
                # Sort by weighted score and return top-k
                results.sort(key=lambda x: x['weighted_score'], reverse=True)
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
            return {'total_episodes': 0, 'index_trained': False}
            
        c_values = [ep.c for ep in self.episodes]
        return {
            'total_episodes': len(self.episodes),
            'index_trained': self.is_trained,
            'c_value_mean': float(np.mean(c_values)),
            'c_value_std': float(np.std(c_values)),
            'c_value_min': float(np.min(c_values)),
            'c_value_max': float(np.max(c_values)),
            'dimension': self.dim
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
        meta = [{
            'c': ep.c,
            'text': ep.text,
            'metadata': ep.metadata
        } for ep in self.episodes]
        
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
                texts = [item['text'] for item in meta]
                vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                
                self.episodes = []
                for vec, item in zip(vecs, meta):
                    episode = Episode(vec, item['text'], item['c'], item.get('metadata', {}))
                    self.episodes.append(episode)
                
                # Load or rebuild index
                if path.exists() and len(self.episodes) >= 2:
                    self.index = faiss.read_index(str(path))
                    self.is_trained = True
                else:
                    self._train_index()
                    
                logger.info(f"Loaded memory from {path} ({len(self.episodes)} episodes)")
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
        
        # Adjust cluster count for small datasets
        n_clusters = min(256, max(1, len(vecs) // 8))
        if hasattr(self.index, 'nlist'):
            self.index.nlist = n_clusters
            
        try:
            self.index.train(vecs)
            self.index.add(vecs)
            self.is_trained = True
            logger.info(f"Trained index with {len(self.episodes)} episodes")
            
        except Exception as e:
            logger.error(f"Index training failed: {e}")
            # Fallback to flat index
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(vecs)
            self.is_trained = True

    def _add_to_index(self, episode: Episode):
        """Add a single episode to the trained index."""
        if self.is_trained:
            self.index.add(episode.vec.reshape(1, -1))

    def _linear_search(self, query_vec: np.ndarray, k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Fallback linear search for small datasets."""
        results = []
        
        for i, episode in enumerate(self.episodes):
            # Cosine similarity
            sim = float(np.dot(query_vec, episode.vec))
            if sim < min_similarity:
                continue
                
            weighted_score = sim * (episode.c ** self.gamma)
            results.append({
                'text': episode.text,
                'similarity': sim,
                'c_value': episode.c,
                'weighted_score': weighted_score,
                'metadata': episode.metadata,
                'index': i
            })
        
        results.sort(key=lambda x: x['weighted_score'], reverse=True)
        return results[:k]

    @classmethod
    def build_from_documents(cls, docs: List[str], batch_size: int = 32) -> 'L2MemoryManager':
        """Build memory manager from a list of documents."""
        model = get_model()
        vecs = model.encode(docs, batch_size=batch_size, show_progress_bar=True, 
                          convert_to_numpy=True, normalize_embeddings=True)
        
        memory = cls(dim=vecs.shape[1])
        for vec, text in zip(vecs, docs):
            episode = Episode(vec, text)
            memory.episodes.append(episode)
            
        memory._train_index()
        return memory
