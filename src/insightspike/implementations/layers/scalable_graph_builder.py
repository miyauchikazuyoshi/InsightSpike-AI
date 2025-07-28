"""
Scalable Graph Builder using FAISS for efficient nearest neighbor search.
Supports large-scale graph construction with O(n log n) complexity.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports with fallback
try:
    import torch
    from torch_geometric.data import Data

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Data = None

# Removed get_config import - now passed via constructor
from ...config.legacy_adapter import LegacyConfigAdapter
from ...monitoring import GraphOperationMonitor, MonitoredOperation

logger = logging.getLogger(__name__)

# Import vector index factory for backend-agnostic implementation
from ...vector_index import VectorIndexFactory


class ScalableGraphBuilder:
    # Added to store previous graph edges for incremental updates
    """
    Build graphs efficiently using FAISS for nearest neighbor search.

    Features:
    - Uses FAISS IndexFlatIP for fast inner product search
    - Configurable top-k neighbors per node
    - Batch processing support
    - Incremental graph updates
    """

    def __init__(self, config=None, monitor: Optional[GraphOperationMonitor] = None):
        """Initialize with configuration object.

        Args:
            config: Configuration object. If None, defaults will be used.
            monitor: Optional GraphOperationMonitor for tracking operations.
        """
        if config is None:
            # Create minimal default config
            logger.warning("No config provided to ScalableGraphBuilder, using defaults")
            from ...config import InsightSpikeConfig
            config = InsightSpikeConfig()
        
        self.config = LegacyConfigAdapter.ensure_pydantic(config)
        
        # Use Pydantic config attributes
        self.similarity_threshold = self.config.graph.similarity_threshold
        # Note: scalable_graph settings might need to be added to GraphConfig
        # For now, use defaults
        self.top_k = 50  # TODO: Add to GraphConfig
        self.batch_size = 1000  # TODO: Add to GraphConfig

        # FAISS index for efficient similarity search
        self.index = None
        # Use Pydantic config
        self.dimension = self.config.embedding.dimension

        # Track document metadata
        self.documents = []
        self.embeddings = None

        # Monitoring
        self.monitor = monitor
        
        # Store previous edges for incremental updates
        self.previous_edge_index = None

    def build_graph(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        incremental: bool = False,
    ) -> Data:
        """
        Build a graph from documents using FAISS for efficient neighbor search.

        Args:
            documents: List of document dictionaries
            embeddings: Optional precomputed embeddings
            incremental: If True, add to existing graph instead of rebuilding

        Returns:
            PyTorch Geometric Data object
        """
        if not documents:
            return self._empty_graph()

        # Get current graph state for monitoring
        def get_graph_state():
            return {
                "nodes": len(self.documents) if self.documents else 0,
                "edges": self.embeddings.shape[0] * self.top_k
                if self.embeddings is not None
                else 0,
            }

        # Use monitoring if available
        if self.monitor:
            with MonitoredOperation(
                self.monitor,
                "build_graph",
                get_graph_state,
                {"incremental": incremental, "num_docs": len(documents)},
            ):
                return self._build_graph_internal(documents, embeddings, incremental)
        else:
            return self._build_graph_internal(documents, embeddings, incremental)

    def _build_graph_internal(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray],
        incremental: bool,
    ) -> Data:
        """Internal graph building logic."""
        try:
            # Extract embeddings if not provided
            if embeddings is None:
                embeddings = self._get_embeddings(documents)

            # Normalize embeddings for cosine similarity via inner product
            embeddings = self._normalize_embeddings(embeddings)

            if incremental and self.index is not None:
                # Add new embeddings to existing index
                new_edges = self._incremental_update(embeddings, len(self.documents))
                
                # Preserve existing edges
                if self.previous_edge_index is not None:
                    edge_index = self.previous_edge_index + new_edges
                    logger.info(f"Incremental update: {len(self.previous_edge_index)} existing + {len(new_edges)} new = {len(edge_index)} total edges")
                else:
                    edge_index = new_edges
                
                self.documents.extend(documents)
                self.embeddings = np.vstack([self.embeddings, embeddings])
            else:
                # Build new index from scratch
                edge_index = self._build_from_scratch(embeddings)
                self.documents = documents
                self.embeddings = embeddings

            # Convert to PyG format
            if incremental and self.embeddings is not None:
                x = torch.tensor(self.embeddings, dtype=torch.float)
            else:
                x = torch.tensor(embeddings, dtype=torch.float)

            if edge_index:
                edge_index_tensor = (
                    torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                )
            else:
                edge_index_tensor = torch.empty(2, 0, dtype=torch.long)

            graph = Data(x=x, edge_index=edge_index_tensor)
            graph.num_nodes = x.size(0)
            graph.documents = self.documents if incremental else documents

            logger.info(
                f"Built graph with {graph.num_nodes} nodes, "
                f"{graph.edge_index.size(1)} edges"
            )
            
            # Store edge index for next incremental update
            if edge_index:
                self.previous_edge_index = edge_index

            return graph

        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            return self._empty_graph()

    def _build_from_scratch(self, embeddings: np.ndarray) -> List[List[int]]:
        """Build graph from scratch using vector index."""
        n_samples = embeddings.shape[0]

        # Get vector search backend from config
        backend = "auto"
        if hasattr(self.config, "vector_search") and hasattr(self.config.vector_search, "backend"):
            backend = self.config.vector_search.backend
        elif hasattr(self.config, "graph") and hasattr(self.config.graph, "use_faiss"):
            backend = "faiss" if self.config.graph.use_faiss else "numpy"
        
        # Initialize vector index using factory
        self.index = VectorIndexFactory.create_index(
            dimension=self.dimension,
            index_type=backend
        )
        self.index.add(embeddings.astype(np.float32))

        # Find k nearest neighbors for each node
        edge_list = []

        # Process in batches for memory efficiency
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_embeddings = embeddings[start_idx:end_idx]

            # Search for top-k neighbors (including self)
            distances, neighbors = self.index.search(
                batch_embeddings.astype(np.float32), min(self.top_k + 1, n_samples)
            )

            # Convert to edge list
            for i, (dists, neighs) in enumerate(zip(distances, neighbors)):
                node_idx = start_idx + i
                # Process all neighbors
                for j, (dist, neigh) in enumerate(zip(dists, neighs)):
                    # Skip self-connections and apply similarity threshold
                    # Note: FAISS IndexFlatIP returns inner product (higher = more similar)
                    if (
                        neigh != -1
                        and neigh != node_idx
                        and dist > self.similarity_threshold
                    ):
                        edge_list.append([node_idx, neigh])

        return edge_list

    def _incremental_update(
        self, new_embeddings: np.ndarray, offset: int
    ) -> List[List[int]]:
        """Add new nodes incrementally without rebuilding entire graph."""
        # Add new embeddings to index
        self.index.add(new_embeddings.astype(np.float32))

        edge_list = []
        n_new = new_embeddings.shape[0]
        n_total = offset + n_new

        # Find neighbors for new nodes
        distances, neighbors = self.index.search(
            new_embeddings.astype(np.float32), min(self.top_k + 1, n_total)
        )

        for i, (dists, neighs) in enumerate(zip(distances, neighbors)):
            new_node_idx = offset + i

            for dist, neigh in zip(dists, neighs):
                # Note: FAISS IndexFlatIP returns inner product (higher = more similar)
                if (
                    neigh != -1
                    and neigh != new_node_idx
                    and dist > self.similarity_threshold
                ):
                    edge_list.append([new_node_idx, neigh])
                    # Add reverse edge for undirected graph
                    edge_list.append([neigh, new_node_idx])

        return edge_list

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity via inner product."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def _get_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Extract embeddings from documents."""
        embeddings = []

        for doc in documents:
            if "embedding" in doc:
                emb = doc["embedding"]
                # Convert to numpy array if needed
                if isinstance(emb, list):
                    emb = np.array(emb)
                elif isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                embeddings.append(emb)
            else:
                # Fallback: create random embedding
                logger.warning(
                    "Document missing embedding, using random initialization"
                )
                embeddings.append(np.random.randn(self.dimension))

        return np.array(embeddings, dtype=np.float32)

    def _empty_graph(self) -> Data:
        """Create an empty graph for error cases."""
        return Data(
            x=torch.empty(0, self.dimension),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            num_nodes=0,
        )

    def get_neighbors(
        self, node_idx: int, k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get k nearest neighbors for a specific node.

        Args:
            node_idx: Index of the query node
            k: Number of neighbors to return (default: self.top_k)

        Returns:
            Tuple of (distances, neighbor_indices)
        """
        if self.index is None or self.embeddings is None:
            return np.array([]), np.array([])

        # Monitor search operation
        if self.monitor:

            def get_graph_state():
                return {"nodes": len(self.documents), "edges": 0}

            with MonitoredOperation(
                self.monitor,
                "search",
                get_graph_state,
                {"node_idx": node_idx, "k": k or self.top_k},
            ):
                return self._get_neighbors_internal(node_idx, k)
        else:
            return self._get_neighbors_internal(node_idx, k)

    def _get_neighbors_internal(
        self, node_idx: int, k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Internal neighbor search logic."""
        k = k or self.top_k
        query_embedding = self.embeddings[node_idx : node_idx + 1]

        distances, neighbors = self.index.search(
            query_embedding.astype(np.float32), min(k + 1, len(self.documents))
        )

        # Remove self from results
        return distances[0][1:], neighbors[0][1:]

    def update_similarity_threshold(self, new_threshold: float):
        """Update similarity threshold for edge creation."""
        self.similarity_threshold = new_threshold
        logger.info(f"Updated similarity threshold to {new_threshold}")
