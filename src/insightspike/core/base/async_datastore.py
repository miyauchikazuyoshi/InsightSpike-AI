"""
Async Data Store Interface
==========================

Asynchronous extension of DataStore interface for scalable operations.
Supports non-blocking I/O and efficient batch operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .datastore import DataStore


class AsyncDataStore(DataStore):
    """Extended DataStore interface with async operations for scalability"""

    # ========== Async Episode Operations ==========

    @abstractmethod
    async def search_episodes_by_vector(
        self,
        query_vector: np.ndarray,
        k: int = 20,
        threshold: float = 0.7,
        namespace: str = "default",
    ) -> List[Dict[str, Any]]:
        """Search for episodes by vector similarity (async)

        Args:
            query_vector: Query vector for similarity search
            k: Number of results to return
            threshold: Minimum similarity threshold
            namespace: Namespace to search in

        Returns:
            List of episode dictionaries sorted by similarity
        """
        pass

    @abstractmethod
    async def get_episodes_by_ids(
        self, ids: List[str], namespace: str = "default"
    ) -> List[Dict[str, Any]]:
        """Get multiple episodes by their IDs (async)

        Args:
            ids: List of episode IDs
            namespace: Namespace to search in

        Returns:
            List of episode dictionaries
        """
        pass

    @abstractmethod
    async def add_episode(
        self, episode: Dict[str, Any], namespace: str = "default"
    ) -> str:
        """Add a single episode (async)

        Args:
            episode: Episode dictionary with 'text', 'vec', 'c', 'metadata'
            namespace: Namespace to add to

        Returns:
            Episode ID
        """
        pass

    @abstractmethod
    async def update_episode(
        self, episode_id: str, updates: Dict[str, Any], namespace: str = "default"
    ) -> bool:
        """Update an existing episode (async)

        Args:
            episode_id: ID of episode to update
            updates: Dictionary of fields to update
            namespace: Namespace of the episode

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def batch_add_episodes(
        self, episodes: List[Dict[str, Any]], namespace: str = "default"
    ) -> List[str]:
        """Add multiple episodes in batch (async)

        Args:
            episodes: List of episode dictionaries
            namespace: Namespace to add to

        Returns:
            List of episode IDs
        """
        pass

    # ========== Async Graph Operations ==========

    @abstractmethod
    async def get_graph_neighbors(
        self, node_id: str, hop: int = 1, namespace: str = "graphs"
    ) -> Dict[str, List[str]]:
        """Get neighboring nodes in graph without loading entire graph (async)

        Args:
            node_id: ID of the center node
            hop: Number of hops to traverse (1 = direct neighbors)
            namespace: Graph namespace

        Returns:
            Dictionary mapping hop distance to list of node IDs
            Example: {1: ['node1', 'node2'], 2: ['node3', 'node4']}
        """
        pass

    @abstractmethod
    async def update_graph_edges(
        self,
        edges_to_add: List[Tuple[str, str, Dict[str, Any]]],
        edges_to_remove: List[Tuple[str, str]],
        namespace: str = "graphs",
    ) -> bool:
        """Update graph edges incrementally (async)

        Args:
            edges_to_add: List of (source, target, attributes) tuples
            edges_to_remove: List of (source, target) tuples
            namespace: Graph namespace

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def get_subgraph(
        self, node_ids: List[str], namespace: str = "graphs"
    ) -> Dict[str, Any]:
        """Get subgraph containing specified nodes (async)

        Args:
            node_ids: List of node IDs to include
            namespace: Graph namespace

        Returns:
            Subgraph data including nodes and edges
        """
        pass

    # ========== Async Vector Operations ==========

    @abstractmethod
    async def search_vectors_batch(
        self, query_vectors: np.ndarray, k: int = 10, namespace: str = "vectors"
    ) -> List[Tuple[List[int], List[float]]]:
        """Batch vector similarity search (async)

        Args:
            query_vectors: Multiple query vectors (N x D)
            k: Number of neighbors per query
            namespace: Vector namespace

        Returns:
            List of (indices, distances) tuples for each query
        """
        pass

    # ========== Transaction Support ==========

    @abstractmethod
    async def begin_transaction(self) -> str:
        """Begin a new transaction

        Returns:
            Transaction ID
        """
        pass

    @abstractmethod
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction

        Args:
            transaction_id: ID of transaction to commit

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a transaction

        Args:
            transaction_id: ID of transaction to rollback

        Returns:
            Success status
        """
        pass

    # ========== Streaming Operations ==========

    @abstractmethod
    async def stream_episodes(self, batch_size: int = 100, namespace: str = "default"):
        """Stream episodes in batches (async generator)

        Args:
            batch_size: Number of episodes per batch
            namespace: Namespace to stream from

        Yields:
            Batches of episode dictionaries
        """
        pass

    @abstractmethod
    async def stream_search_results(
        self,
        query_vector: np.ndarray,
        max_results: int = 1000,
        batch_size: int = 100,
        namespace: str = "default",
    ):
        """Stream search results in batches (async generator)

        Args:
            query_vector: Query vector for similarity search
            max_results: Maximum total results
            batch_size: Results per batch
            namespace: Namespace to search in

        Yields:
            Batches of search results
        """
        pass
