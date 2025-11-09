"""
FileSystem DataStore Implementation
==================================

File-based implementation of DataStore interface.
Maintains backward compatibility with existing file formats.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...config.constants import FILE_FORMAT_MAPPING, DataType, FileFormat
from ...utils.path_utils import resolve_project_relative
from ...core.base.datastore import DataStore, VectorIndex
from ...core.exceptions import (
    DataStoreLoadError,
    DataStoreNotFoundError,
    DataStorePermissionError,
    DataStoreSaveError,
)

logger = logging.getLogger(__name__)


class FileSystemDataStore(DataStore):
    """File system based data store implementation

    Accepts both `base_path` (legacy/internal) and `root_path` (config-facing)
    for backwards/forwards compatibility with different config schemas.
    """

    def __init__(self, base_path: str = "data", root_path: Optional[str] = None):
        """Initialize filesystem store

        Args:
            base_path: Base directory for all data (legacy/internal name)
            root_path: Base directory (alias used by public/config models)
        """
        # Prefer explicitly provided root_path if available
        effective_base = root_path if root_path is not None else base_path
        # Normalize relative paths against project root to prevent nested growth
        self.base_path = Path(resolve_project_relative(effective_base))
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create namespace directories following proposed structure
        self.namespaces = {
            "episodes": self.base_path / "core",
            "vectors": self.base_path / "core",
            "graphs": self.base_path / "core",
            "metadata": self.base_path / "core",
            "db": self.base_path / "db",
            "learning": self.base_path / "learning",
            "experiments": self.base_path / "experiments",
            "cache": self.base_path / "cache",
            "temp": self.base_path / "temp",
        }

        for namespace_dir in self.namespaces.values():
            namespace_dir.mkdir(exist_ok=True)

        # Vector indices for each namespace
        self.vector_indices: Dict[str, "FAISSVectorIndex"] = {}

    def _get_namespace_path(self, namespace: str) -> Path:
        """Get path for namespace, creating if needed"""
        if namespace not in self.namespaces:
            namespace_path = self.base_path / namespace
            namespace_path.mkdir(exist_ok=True)
            self.namespaces[namespace] = namespace_path
        return self.namespaces[namespace]

    def save_episodes(
        self, episodes: List[Dict[str, Any]], namespace: str = "default"
    ) -> bool:
        """Save episodes to JSON file"""
        try:
            namespace_path = self._get_namespace_path("episodes")
            file_path = namespace_path / f"{namespace}.json"

            # Convert numpy arrays to lists for JSON serialization without copying
            def serialize_episode(ep):
                # Create new dict with only necessary fields to avoid deep copy
                serialized = {
                    "text": ep["text"],
                    "c_value": ep.get("c_value", 0.5),
                    "timestamp": ep.get("timestamp"),
                    "metadata": ep.get("metadata", {}),
                }

                # Handle vector conversion efficiently
                if "vec" in ep:
                    vec = ep["vec"]
                    serialized["vec"] = (
                        vec.tolist() if isinstance(vec, np.ndarray) else vec
                    )

                # Include any other fields that aren't in the standard set
                for key, value in ep.items():
                    if key not in serialized and key != "vec":
                        serialized[key] = value

                return serialized

            # Load existing episodes (append behavior) if file exists
            existing: List[Dict[str, Any]] = []
            if file_path.exists():
                try:
                    with open(file_path, "r") as rf:
                        existing = json.load(rf)
                except Exception:
                    existing = []

            # Use list comprehension for efficiency and normalize new entries
            serializable_episodes = [serialize_episode(ep) for ep in episodes]

            # Append to existing list
            combined = existing + serializable_episodes

            with open(file_path, "w") as f:
                json.dump(combined, f, indent=2)

            logger.info(f"Saved {len(episodes)} episodes to {file_path}")
            return True

        except IOError as e:
            raise DataStoreSaveError(
                f"Failed to write episodes file: {e}", details={"path": file_path}
            ) from e
        except (TypeError, ValueError) as e:
            raise DataStoreSaveError(
                f"Failed to serialize episodes: {e}",
                details={"num_episodes": len(episodes)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error saving episodes: {e}")
            raise DataStoreSaveError(f"Failed to save episodes: {e}") from e

    def load_episodes(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Load episodes from JSON file"""
        try:
            namespace_path = self._get_namespace_path("episodes")
            file_path = namespace_path / f"{namespace}.json"

            if not file_path.exists():
                return []

            with open(file_path, "r") as f:
                episodes = json.load(f)

            # Convert lists back to numpy arrays
            for ep in episodes:
                if "vec" in ep and isinstance(ep["vec"], list):
                    ep["vec"] = np.array(ep["vec"], dtype=np.float32)

            logger.info(f"Loaded {len(episodes)} episodes from {file_path}")
            return episodes

        except FileNotFoundError:
            logger.info(f"No episodes file found at {file_path}")
            return []  # This is not an error - just no data yet
        except json.JSONDecodeError as e:
            raise DataStoreLoadError(
                f"Episodes file is corrupted: {e}", details={"path": file_path}
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error loading episodes: {e}")
            raise DataStoreLoadError(f"Failed to load episodes: {e}") from e

    def save_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        namespace: str = "vectors",
    ) -> bool:
        """Save vectors and metadata"""
        try:
            namespace_path = self._get_namespace_path("vectors")

            # Save vectors as numpy file
            vectors_path = namespace_path / f"{namespace}_vectors.npy"
            np.save(vectors_path, vectors)

            # Save metadata as JSON
            metadata_path = namespace_path / f"{namespace}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Update FAISS index if available
            if namespace not in self.vector_indices:
                self.vector_indices[namespace] = FAISSVectorIndex(vectors.shape[1])

            self.vector_indices[namespace].add_vectors(vectors)

            logger.info(f"Saved {len(vectors)} vectors to {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to save vectors: {e}")
            return False

    def search_vectors(
        self, query_vector: np.ndarray, k: int = 10, namespace: str = "vectors"
    ) -> Tuple[List[int], List[float]]:
        """Search for similar vectors"""
        try:
            if namespace in self.vector_indices:
                distances, indices = self.vector_indices[namespace].search(
                    query_vector.reshape(1, -1), k
                )
                return indices[0].tolist(), distances[0].tolist()

            # Fallback to brute force search
            namespace_path = self._get_namespace_path("vectors")
            vectors_path = namespace_path / f"{namespace}_vectors.npy"

            if not vectors_path.exists():
                return [], []

            vectors = np.load(vectors_path)

            # Use _normalize_vectors for consistent normalization
            query_norm = self._normalize_vectors(query_vector.reshape(1, -1))[0]
            vectors_norm = self._normalize_vectors(vectors)

            # Compute cosine similarities
            similarities = np.dot(vectors_norm, query_norm)

            # Get top k (highest similarities)
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            top_k_sim = similarities[top_k_idx]

            # Convert similarities to distances for consistency
            return top_k_idx.tolist(), (1 - top_k_sim).tolist()

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return [], []

    def save_graph(
        self, graph_data: Any, graph_id: str, namespace: str = "graphs"
    ) -> bool:
        """Save graph data using pickle or torch"""
        try:
            namespace_path = self._get_namespace_path("graphs")
            file_path = namespace_path / f"{graph_id}.pkl"

            # Try torch save if available and graph is a torch object
            try:
                import torch

                if (
                    hasattr(graph_data, "__module__")
                    and "torch" in graph_data.__module__
                ):
                    torch_path = namespace_path / f"{graph_id}.pt"
                    torch.save(graph_data, torch_path)
                    logger.info(f"Saved graph {graph_id} as torch file")
                    return True
            except ImportError:
                pass

            # Fallback to pickle
            with open(file_path, "wb") as f:
                pickle.dump(graph_data, f)

            logger.info(f"Saved graph {graph_id} as pickle file")
            return True

        except IOError as e:
            raise DataStoreSaveError(
                f"Failed to write graph file: {e}", details={"graph_id": graph_id}
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error saving graph: {e}")
            raise DataStoreSaveError(f"Failed to save graph: {e}") from e

    def load_graph(self, graph_id: str, namespace: str = "graphs") -> Optional[Any]:
        """Load graph data"""
        try:
            namespace_path = self._get_namespace_path("graphs")

            # Try torch load first
            torch_path = namespace_path / f"{graph_id}.pt"
            if torch_path.exists():
                try:
                    import torch

                    return torch.load(torch_path)
                except ImportError:
                    logger.warning("PyTorch not available, trying pickle")

            # Try pickle
            pickle_path = namespace_path / f"{graph_id}.pkl"
            if pickle_path.exists():
                with open(pickle_path, "rb") as f:
                    return pickle.load(f)

            return None

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return None

    def save_metadata(
        self, metadata: Dict[str, Any], key: str, namespace: str = "metadata"
    ) -> bool:
        """Save metadata as JSON"""
        try:
            namespace_path = self._get_namespace_path(namespace)
            file_path = namespace_path / f"{key}.json"

            with open(file_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False

    def load_metadata(
        self, key: str, namespace: str = "metadata"
    ) -> Optional[Dict[str, Any]]:
        """Load metadata from JSON"""
        try:
            namespace_path = self._get_namespace_path(namespace)
            file_path = namespace_path / f"{key}.json"

            if not file_path.exists():
                return None

            with open(file_path, "r") as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None

    def delete(self, key: str, namespace: str) -> bool:
        """Delete file by key"""
        try:
            namespace_path = self._get_namespace_path(namespace)

            # Try all possible extensions from FileFormat enum
            all_extensions = {fmt.value for fmt in FileFormat}
            for ext in all_extensions:
                file_path = namespace_path / f"{key}{ext}"
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted {file_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    def list_keys(self, namespace: str, pattern: Optional[str] = None) -> List[str]:
        """List keys in namespace"""
        try:
            namespace_path = self._get_namespace_path(namespace)

            if not namespace_path.exists():
                return []

            keys = []
            for file_path in namespace_path.iterdir():
                if file_path.is_file():
                    key = file_path.stem  # Remove extension
                    if pattern is None or pattern in key:
                        keys.append(key)

            return sorted(keys)

        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []

    def clear_namespace(self, namespace: str) -> bool:
        """Clear all files in namespace"""
        try:
            namespace_path = self._get_namespace_path(namespace)

            if namespace_path.exists():
                for file_path in namespace_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()

            logger.info(f"Cleared namespace {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {}

        for name, path in self.namespaces.items():
            if path.exists():
                file_count = len(list(path.glob("*")))
                total_size = sum(
                    f.stat().st_size for f in path.glob("*") if f.is_file()
                )
                stats[name] = {
                    "file_count": file_count,
                    "total_size_mb": total_size / (1024 * 1024),
                }

        stats["vector_indices"] = {
            name: idx.get_size() for name, idx in self.vector_indices.items()
        }

        return stats

    def save_queries(
        self, queries: List[Dict[str, Any]], namespace: str = "queries"
    ) -> bool:
        """Save query records to storage"""
        try:
            namespace_path = self._get_namespace_path(namespace)
            
            # Load existing queries
            existing_queries = []
            queries_file = namespace_path / "queries.json"
            if queries_file.exists():
                with open(queries_file, "r") as f:
                    existing_queries = json.load(f)
            
            # Append new queries
            existing_queries.extend(queries)
            
            # Save back to file
            with open(queries_file, "w") as f:
                json.dump(existing_queries, f, indent=2, default=str)
            
            logger.info(f"Saved {len(queries)} queries to {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save queries: {e}")
            return False

    def load_queries(
        self, 
        namespace: str = "queries",
        has_spike: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load query records from storage"""
        try:
            namespace_path = self._get_namespace_path(namespace)
            queries_file = namespace_path / "queries.json"
            
            if not queries_file.exists():
                logger.info(f"No queries found in {namespace}")
                return []
            
            with open(queries_file, "r") as f:
                all_queries = json.load(f)
            
            # Filter by has_spike if specified
            if has_spike is not None:
                filtered_queries = [
                    q for q in all_queries 
                    if q.get("has_spike", False) == has_spike
                ]
            else:
                filtered_queries = all_queries
            
            # Sort by timestamp (newest first)
            filtered_queries.sort(
                key=lambda x: x.get("timestamp", 0), 
                reverse=True
            )
            
            # Apply limit if specified
            if limit is not None:
                filtered_queries = filtered_queries[:limit]
            
            logger.info(f"Loaded {len(filtered_queries)} queries from {namespace}")
            return filtered_queries
            
        except Exception as e:
            logger.error(f"Failed to load queries: {e}")
            return []


class FAISSVectorIndex(VectorIndex):
    """FAISS-based vector index implementation"""

    def __init__(self, dimension: int):
        """Initialize FAISS index

        Args:
            dimension: Vector dimension
        """
        self.dimension = dimension
        self.vectors = []
        self.index = None

        try:
            import faiss

            self.faiss = faiss
            # Use inner product (IP) index for cosine similarity with normalized vectors
            self.index = faiss.IndexFlatIP(dimension)
            self.use_faiss = True
            logger.info("Using FAISS IndexFlatIP for cosine similarity search")
        except ImportError:
            logger.warning("FAISS not available, using numpy for vector search")
            self.use_faiss = False

    def add_vectors(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> bool:
        """Add vectors to index"""
        try:
            vectors = np.asarray(vectors, dtype=np.float32)

            # Normalize vectors for cosine similarity
            vectors = self._normalize_vectors(vectors)

            if self.use_faiss and self.index is not None:
                self.index.add(vectors)
            else:
                self.vectors.append(vectors)

            return True

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors using cosine similarity"""
        try:
            query_vectors = np.asarray(query_vectors, dtype=np.float32)

            # Normalize query vectors
            query_vectors = self._normalize_vectors(query_vectors)

            if self.use_faiss and self.index is not None:
                # FAISS returns inner product (which equals cosine similarity for normalized vectors)
                # We convert to distance: distance = 1 - similarity
                similarities, indices = self.index.search(query_vectors, k)
                distances = 1 - similarities
                return distances, indices

            # Numpy fallback - cosine similarity
            if not self.vectors:
                return np.array([[]]), np.array([[]])

            all_vectors = np.vstack(self.vectors)
            # Compute cosine similarities (dot product of normalized vectors)
            similarities = np.dot(query_vectors, all_vectors.T)

            # Get top k (highest similarities)
            indices = np.argsort(-similarities, axis=1)[:, :k]
            sorted_similarities = np.take_along_axis(similarities, indices, axis=1)

            # Convert to distances
            distances = 1 - sorted_similarities

            return distances, indices

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return np.array([[]]), np.array([[]])

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity"""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # Calculate L2 norms
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        # Normalize
        return vectors / norms

    def remove_vectors(self, ids: List[int]) -> bool:
        """Remove vectors not supported in basic implementation"""
        logger.warning("Vector removal not supported in FileSystem implementation")
        return False

    def save_index(self, path: str) -> bool:
        """Save index to disk"""
        try:
            if self.use_faiss and self.index is not None:
                self.faiss.write_index(self.index, path)
            else:
                np.save(path, np.vstack(self.vectors) if self.vectors else np.array([]))
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Load index from disk"""
        try:
            if self.use_faiss:
                self.index = self.faiss.read_index(path)
            else:
                vectors = np.load(path)
                if len(vectors) > 0:
                    self.vectors = [vectors]
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def get_size(self) -> int:
        """Get number of vectors"""
        if self.use_faiss and self.index is not None:
            return self.index.ntotal
        return sum(len(v) for v in self.vectors)
