"""
Data Store Interface
===================

Abstract interface for data persistence layer.
Allows swapping between different storage backends (filesystem, databases, vector DBs, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DataStore(ABC):
    """Abstract interface for data persistence"""

    @abstractmethod
    def save_episodes(
        self, episodes: List[Dict[str, Any]], namespace: str = "default"
    ) -> bool:
        """Save episodes to storage

        Args:
            episodes: List of episode dictionaries with 'text', 'vec', 'c', 'metadata'
            namespace: Namespace/collection for organizing data

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def load_episodes(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Load episodes from storage

        Args:
            namespace: Namespace/collection to load from

        Returns:
            List of episode dictionaries
        """
        pass

    @abstractmethod
    def save_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        namespace: str = "vectors",
    ) -> bool:
        """Save vectors with metadata

        Args:
            vectors: Numpy array of vectors (N x D)
            metadata: List of metadata dicts for each vector
            namespace: Namespace for the vectors

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def search_vectors(
        self, query_vector: np.ndarray, k: int = 10, namespace: str = "vectors"
    ) -> Tuple[List[int], List[float]]:
        """Search for similar vectors

        Args:
            query_vector: Query vector
            k: Number of results
            namespace: Namespace to search in

        Returns:
            Tuple of (indices, distances)
        """
        pass

    @abstractmethod
    def save_graph(
        self, graph_data: Any, graph_id: str, namespace: str = "graphs"
    ) -> bool:
        """Save graph data

        Args:
            graph_data: Graph data (format depends on implementation)
            graph_id: Unique identifier for the graph
            namespace: Namespace for graphs

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def load_graph(self, graph_id: str, namespace: str = "graphs") -> Optional[Any]:
        """Load graph data

        Args:
            graph_id: Graph identifier
            namespace: Namespace for graphs

        Returns:
            Graph data or None if not found
        """
        pass

    @abstractmethod
    def save_metadata(
        self, metadata: Dict[str, Any], key: str, namespace: str = "metadata"
    ) -> bool:
        """Save arbitrary metadata

        Args:
            metadata: Dictionary of metadata
            key: Storage key
            namespace: Namespace for metadata

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def load_metadata(
        self, key: str, namespace: str = "metadata"
    ) -> Optional[Dict[str, Any]]:
        """Load metadata

        Args:
            key: Storage key
            namespace: Namespace for metadata

        Returns:
            Metadata dict or None if not found
        """
        pass

    @abstractmethod
    def delete(self, key: str, namespace: str) -> bool:
        """Delete data by key

        Args:
            key: Data key
            namespace: Namespace

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def list_keys(self, namespace: str, pattern: Optional[str] = None) -> List[str]:
        """List keys in namespace

        Args:
            namespace: Namespace to list
            pattern: Optional pattern to filter keys

        Returns:
            List of keys
        """
        pass

    @abstractmethod
    def clear_namespace(self, namespace: str) -> bool:
        """Clear all data in a namespace

        Args:
            namespace: Namespace to clear

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics

        Returns:
            Dictionary of statistics
        """
        pass


class VectorIndex(ABC):
    """Abstract interface for vector indexing"""

    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> bool:
        """Add vectors to index

        Args:
            vectors: Vectors to add (N x D)
            ids: Optional IDs for vectors

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors

        Args:
            query_vectors: Query vectors (M x D)
            k: Number of neighbors

        Returns:
            Tuple of (distances, indices) both of shape (M x k)
        """
        pass

    @abstractmethod
    def remove_vectors(self, ids: List[int]) -> bool:
        """Remove vectors by ID

        Args:
            ids: Vector IDs to remove

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def save_index(self, path: str) -> bool:
        """Save index to disk

        Args:
            path: Save path

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def load_index(self, path: str) -> bool:
        """Load index from disk

        Args:
            path: Load path

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Get number of vectors in index

        Returns:
            Number of vectors
        """
        pass
