"""DataStore-backed VectorIndex stub (Phase 4 - DataStoreIndex PENDING part).

This provides a minimal wrapper adapting an existing DataStore (if present in
upstream project) to the VectorIndex protocol used by MazeNavigator.

Current behavior:
- Stores vectors in-memory (mirrors InMemoryIndex) while also optionally
  forwarding persistence calls to a provided DataStore object if it exposes
  save_vectors / load_vectors style APIs (best-effort, silent on failure).
- Search is linear scan (L2) to maintain behavioral parity.

Future (Phase 5/6):
- Replace in-memory storage with streaming / partial load from DataStore.
- Introduce ANN acceleration (Faiss / hnswlib) behind same protocol.
"""
from __future__ import annotations
from typing import Optional, Sequence, List, Tuple, Any
import numpy as np

try:
    # Attempt to import a generic DataStore factory / base if available
    from insightspike.implementations.datastore.factory import DataStoreFactory  # type: ignore
except Exception:  # pragma: no cover - optional dependency not required for tests
    DataStoreFactory = None  # type: ignore

class DataStoreIndex:
    """Linear scan index with optional persistence side-effects via DataStore.

    Args:
        datastore: Optional existing datastore instance (duck-typed). If None and
                   `datastore_type` is provided and factory import succeeded,
                   attempts to create via factory.
        datastore_type: String key for factory creation (e.g. 'filesystem').
        namespace: Logical namespace under which vectors may be persisted.
    """
    def __init__(self, datastore: Any | None = None, datastore_type: str | None = None, namespace: str = "maze_index"):
        if datastore is None and datastore_type and DataStoreFactory:
            try:
                datastore = DataStoreFactory.create(datastore_type)
            except Exception:
                datastore = None
        self.datastore = datastore
        self.namespace = namespace
        self._vectors: dict[int, np.ndarray] = {}
        self._dim: Optional[int] = None
        # Attempt lazy load if datastore offers a method
        if self.datastore is not None:
            self._maybe_load_existing()

    # VectorIndex protocol methods -------------------------------------------------
    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        if self._dim is None:
            self._dim = vectors.shape[1]
        for i, vid in enumerate(ids):
            self._vectors[int(vid)] = vectors[i]
        self._maybe_persist(ids, vectors)

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if top_k <= 0 or not self._vectors:
            return []
        # Defensive dimension match
        if self._dim is not None and query.shape[0] != self._dim:
            # Try to coerce if possible (truncate/pad) – conservative fallback
            if query.shape[0] > self._dim:
                query = query[: self._dim]
            else:
                query = np.pad(query, (0, self._dim - query.shape[0]))
        res: List[Tuple[int, float]] = []
        for vid, vec in self._vectors.items():
            if self._dim is not None and vec.shape[0] != self._dim:
                continue
            dist = float(np.linalg.norm(query - vec))
            res.append((vid, dist))
        res.sort(key=lambda x: x[1])
        return res[:top_k] if len(res) > top_k else res

    def __len__(self) -> int:
        return len(self._vectors)

    def remove(self, ids: Sequence[int]) -> None:
        """Remove vectors from in-memory map and forward delete to datastore if supported.

        Best-effort; silently ignores errors or missing ids.
        """
        for vid in ids:
            self._vectors.pop(int(vid), None)
        if self.datastore is None:
            return
        try:
            # Prefer explicit vector removal API if present
            if hasattr(self.datastore, 'delete_vectors'):
                self.datastore.delete_vectors([int(v) for v in ids], namespace=self.namespace)  # type: ignore[arg-type]
            elif hasattr(self.datastore, 'delete_episodes'):
                self.datastore.delete_episodes([int(v) for v in ids], namespace=self.namespace)  # type: ignore[arg-type]
        except Exception:
            pass

    # Phase5/6 helper: optional single-vector lazy fetch (used for rehydration if not cached)
    def get(self, vid: int) -> np.ndarray | None:
        vec = self._vectors.get(int(vid))
        if vec is not None:
            return vec
        if self.datastore is None:
            return None
        try:
            if hasattr(self.datastore, 'load_vector'):
                rec = self.datastore.load_vector(int(vid), namespace=self.namespace)
                if rec and 'vector' in rec:
                    arr = np.asarray(rec['vector'], dtype=float)
                    if self._dim is None:
                        self._dim = arr.shape[0]
                    if self._dim == arr.shape[0]:
                        self._vectors[int(vid)] = arr
                        return arr
            elif hasattr(self.datastore, 'load_episode'):
                rec = self.datastore.load_episode(int(vid), namespace=self.namespace)
                if rec and 'vector' in rec:
                    arr = np.asarray(rec['vector'], dtype=float)
                    if self._dim is None:
                        self._dim = arr.shape[0]
                    if self._dim == arr.shape[0]:
                        self._vectors[int(vid)] = arr
                        return arr
        except Exception:
            return None
        return None

    # Internal helpers -------------------------------------------------------------
    def _maybe_persist(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        if self.datastore is None:
            return
        # Best-effort: look for a generic save method
        try:
            if hasattr(self.datastore, 'save_vectors'):
                payload = [ {'id': int(ids[i]), 'vector': vectors[i].tolist()} for i in range(len(ids)) ]
                self.datastore.save_vectors(payload, namespace=self.namespace)  # type: ignore[arg-type]
            elif hasattr(self.datastore, 'save_episodes'):
                # Fallback: store as pseudo-episodes (id only, no metadata)
                payload = [ {'episode_id': int(ids[i]), 'vector': vectors[i].tolist()} for i in range(len(ids)) ]
                self.datastore.save_episodes(payload, namespace=self.namespace)  # type: ignore[arg-type]
        except Exception:
            pass  # Silently ignore persistence issues (non-critical for Phase 4)

    def _maybe_load_existing(self) -> None:
        if self.datastore is None:
            return
        try:
            vecs = None
            if hasattr(self.datastore, 'load_vectors'):
                vecs = self.datastore.load_vectors(namespace=self.namespace)  # type: ignore[assignment]
            elif hasattr(self.datastore, 'load_episodes'):
                # Attempt to interpret stored episodes with 'vector'
                epis = self.datastore.load_episodes(namespace=self.namespace)
                vecs = [ {'id': e.get('episode_id') or e.get('id'), 'vector': e.get('vector')} for e in epis if 'vector' in e ]
            if not vecs:
                return
            for rec in vecs:
                vid = rec.get('id')
                v = rec.get('vector')
                if vid is None or v is None:
                    continue
                arr = np.asarray(v, dtype=float)
                if self._dim is None:
                    self._dim = arr.shape[0]
                if self._dim == arr.shape[0]:
                    self._vectors[int(vid)] = arr
        except Exception:
            pass

__all__ = ["DataStoreIndex"]
