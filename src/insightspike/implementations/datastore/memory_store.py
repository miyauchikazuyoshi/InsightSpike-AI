"""
In-Memory DataStore Implementation
=================================

Memory-based implementation for testing and development.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.base.datastore import DataStore

logger = logging.getLogger(__name__)


class InMemoryDataStore(DataStore):
    """In-memory data store implementation"""

    def __init__(self):
        """Initialize in-memory store"""
        self.data = defaultdict(dict)
        self.vectors = defaultdict(lambda: {"vectors": None, "metadata": []})

    def save_episodes(
        self, episodes: List[Dict[str, Any]], namespace: str = "default"
    ) -> bool:
        """Save episodes to memory"""
        try:
            self.data["episodes"][namespace] = episodes.copy()
            logger.info(f"Saved {len(episodes)} episodes to memory")
            return True
        except Exception as e:
            logger.error(f"Failed to save episodes: {e}")
            return False

    def load_episodes(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Load episodes from memory"""
        return self.data["episodes"].get(namespace, [])

    def save_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        namespace: str = "vectors",
    ) -> bool:
        """Save vectors to memory"""
        try:
            self.vectors[namespace] = {
                "vectors": vectors.copy(),
                "metadata": metadata.copy(),
            }
            return True
        except Exception as e:
            logger.error(f"Failed to save vectors: {e}")
            return False

    def search_vectors(
        self, query_vector: np.ndarray, k: int = 10, namespace: str = "vectors"
    ) -> Tuple[List[int], List[float]]:
        """Search vectors in memory"""
        try:
            if (
                namespace not in self.vectors
                or self.vectors[namespace]["vectors"] is None
            ):
                return [], []

            vectors = self.vectors[namespace]["vectors"]

            # Compute cosine similarities
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            vectors_norm = vectors / (
                np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            )
            similarities = np.dot(vectors_norm, query_norm)

            # Get top k
            k = min(k, len(vectors))
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            top_k_sim = similarities[top_k_idx]

            return top_k_idx.tolist(), (1 - top_k_sim).tolist()

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return [], []

    def save_graph(
        self, graph_data: Any, graph_id: str, namespace: str = "graphs"
    ) -> bool:
        """Save graph to memory"""
        try:
            if namespace not in self.data:
                self.data[namespace] = {}
            self.data[namespace][graph_id] = graph_data
            return True
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False

    def load_graph(self, graph_id: str, namespace: str = "graphs") -> Optional[Any]:
        """Load graph from memory"""
        return self.data.get(namespace, {}).get(graph_id)

    def save_metadata(
        self, metadata: Dict[str, Any], key: str, namespace: str = "metadata"
    ) -> bool:
        """Save metadata to memory"""
        try:
            if namespace not in self.data:
                self.data[namespace] = {}
            self.data[namespace][key] = metadata.copy()
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False

    def load_metadata(
        self, key: str, namespace: str = "metadata"
    ) -> Optional[Dict[str, Any]]:
        """Load metadata from memory"""
        return self.data.get(namespace, {}).get(key)

    def delete(self, key: str, namespace: str) -> bool:
        """Delete from memory"""
        try:
            if namespace in self.data and key in self.data[namespace]:
                del self.data[namespace][key]
            return True
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            return False

    def list_keys(self, namespace: str, pattern: Optional[str] = None) -> List[str]:
        """List keys in namespace"""
        if namespace not in self.data:
            return []

        keys = list(self.data[namespace].keys())
        if pattern:
            keys = [k for k in keys if pattern in k]

        return sorted(keys)

    def clear_namespace(self, namespace: str) -> bool:
        """Clear namespace"""
        try:
            if namespace in self.data:
                self.data[namespace].clear()
            if namespace in self.vectors:
                self.vectors[namespace] = {"vectors": None, "metadata": []}
            return True
        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {}

        for namespace, data in self.data.items():
            stats[namespace] = {
                "item_count": len(data),
                "estimated_size_mb": self._estimate_size(data) / (1024 * 1024),
            }

        for namespace, vec_data in self.vectors.items():
            if vec_data["vectors"] is not None:
                stats[f"vectors_{namespace}"] = {
                    "vector_count": len(vec_data["vectors"]),
                    "dimension": vec_data["vectors"].shape[1]
                    if len(vec_data["vectors"]) > 0
                    else 0,
                }

        return stats

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        import sys

        return sys.getsizeof(obj)
