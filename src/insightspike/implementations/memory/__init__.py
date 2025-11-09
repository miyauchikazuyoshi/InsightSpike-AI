"""
Compatibility memory module exposing a simple `Memory` interface expected by older tests.

Provides minimal add_episode/retrieve built on top of a DataStore.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..datastore.memory_store import InMemoryDataStore
from ...core.base.datastore import DataStore
from ...processing.embedder import EmbeddingManager

__all__ = ["KnowledgeGraphMemory", "Memory"]


class Memory:
    """Legacy-compatible memory facade backed by a DataStore.

    - add_episode accepts dicts with keys: text, embedding (or vec), c_value.
    - retrieve performs a lightweight cosine search over stored embeddings.
    """

    def __init__(self, datastore: Optional[DataStore] = None, namespace: str = "episodes"):
        self.datastore = datastore or InMemoryDataStore()
        self.namespace = namespace
        self.embedder = EmbeddingManager()

    def _load_all(self) -> List[Dict[str, Any]]:
        try:
            return self.datastore.load_episodes(namespace=self.namespace) or []
        except Exception:
            return []

    def _save_all(self, episodes: List[Dict[str, Any]]) -> bool:
        try:
            return self.datastore.save_episodes(episodes, namespace=self.namespace)
        except Exception:
            return False

    def add_episode(self, episode: Dict[str, Any]) -> int:
        """Append a single episode dict to the datastore.

        Normalizes keys to match datastore expectations.
        """
        eps = self._load_all()
        # Normalize keys: embedding->vec; include c_value default
        ep = dict(episode)
        if "vec" not in ep and "embedding" in ep:
            ep["vec"] = ep["embedding"]
        ep.setdefault("c_value", ep.get("c", 0.5))
        eps.append(ep)
        self._save_all(eps)
        return len(eps) - 1

    def retrieve(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k results with basic cosine similarity over stored vectors."""
        episodes = self._load_all()
        if not episodes:
            return []
        qv = self.embedder.get_embedding(query_text)
        # Normalize shapes
        qv = np.array(qv, dtype=np.float32).reshape(-1)
        scored: List[Dict[str, Any]] = []
        for idx, ep in enumerate(episodes):
            vec = ep.get("vec", ep.get("embedding"))
            if vec is None:
                continue
            v = np.array(vec, dtype=np.float32).reshape(-1)
            # cosine
            denom = (np.linalg.norm(qv) * np.linalg.norm(v)) + 1e-8
            sim = float(np.dot(qv, v) / denom)
            scored.append(
                {
                    "text": ep.get("text", ""),
                    "similarity": sim,
                    "index": idx,
                    "c_value": ep.get("c_value", ep.get("c", 0.5)),
                    "embedding": v,
                }
            )
        scored.sort(key=lambda d: d["similarity"], reverse=True)
        return scored[:top_k]
