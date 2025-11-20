"""
Scalable Graph Manager with FAISS Integration
============================================

Manages knowledge graph operations with O(n log n) scalability using FAISS.
Supports dynamic graph growth, conflict-based splitting, and graph-based importance.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from ...vector_index import VectorIndexFactory

logger = logging.getLogger(__name__)


class ScalableGraphManager:
    """
    Scalable graph management with FAISS-based similarity search.

    Features:
    - O(n log n) complexity for graph operations
    - Dynamic top-k neighbor connections
    - Conflict detection and episode splitting
    - Graph-based importance calculation
    - Incremental graph updates
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        similarity_threshold: float = 0.3,
        top_k: int = 50,
        conflict_threshold: float = 0.8,
    ):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.conflict_threshold = conflict_threshold

        # Vector index for efficient similarity search
        self.vector_index = None
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

        # Current graph state
        self.graph = None
        self._initialize_empty_graph()

        # Conflict tracking
        self.conflict_history: List[Dict[str, Any]] = []

    def _initialize_empty_graph(self):
        """Initialize an empty graph."""
        graph = Data(
            x=torch.empty((0, self.embedding_dim)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
        graph.num_nodes = 0
        self.graph = graph
        return graph

    def add_episode_node(
        self,
        embedding: np.ndarray,
        index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a node with FAISS-based neighbor search.

        Returns:
            Dict with added edges, conflicts detected, and performance metrics
        """
        try:
            start_time = time.time()
            embedding = embedding.astype(np.float32)

            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Initialize FAISS index if needed
            if self.vector_index is None:
                self.vector_index = VectorIndexFactory.create_index(
                    dimension=self.embedding_dim,
                    index_type="auto"
                )

            # Store embedding and metadata
            self.embeddings.append(embedding)
            self.metadata.append(metadata or {})
            node_id = len(self.embeddings) - 1

            # Add to FAISS index
            self.vector_index.add(embedding.reshape(1, -1))

            # Update graph node features
            new_x = torch.tensor(embedding).view(1, -1)
            if self.graph.x.numel() == 0:
                self.graph.x = new_x
            else:
                self.graph.x = torch.cat([self.graph.x, new_x], dim=0)

            # Find neighbors and create edges
            edges_added = []
            conflicts_detected = []

            if node_id > 0:
                # Search for top-k similar nodes
                k = min(self.top_k, node_id)
                distances, neighbors = self.vector_index.search(
                    embedding.reshape(1, -1), k + 1  # +1 to include self
                )

                # Process neighbors (skip self)
                for dist, neigh in zip(distances[0][1:], neighbors[0][1:]):
                    if neigh != -1 and dist > self.similarity_threshold:
                        # Add bidirectional edges
                        edges_added.extend([[node_id, neigh], [neigh, node_id]])

                        # Check for conflicts
                        if dist > self.conflict_threshold:
                            conflict = self._detect_conflict(node_id, neigh, dist)
                            if conflict:
                                conflicts_detected.append(conflict)

                # Update edge index
                if edges_added:
                    new_edges = torch.tensor(edges_added, dtype=torch.long).t()
                    if self.graph.edge_index.numel() == 0:
                        self.graph.edge_index = new_edges
                    else:
                        self.graph.edge_index = torch.cat(
                            [self.graph.edge_index, new_edges], dim=1
                        )

            # Update graph metadata
            self.graph.num_nodes = len(self.embeddings)

            # Calculate graph-based importance
            importance = self._calculate_node_importance(node_id)

            build_time = time.time() - start_time

            return {
                "success": True,
                "node_id": node_id,
                "edges_added": len(edges_added) // 2,  # Count undirected edges
                "conflicts": conflicts_detected,
                "importance": importance,
                "build_time": build_time,
                "graph_stats": {
                    "nodes": self.graph.num_nodes,
                    "edges": self.graph.edge_index.size(1) // 2,
                },
            }

        except Exception as e:
            logger.error(f"Failed to add episode node: {e}")
            return {
                "success": False,
                "error": str(e),
                "node_id": -1,
                "edges_added": 0,
                "conflicts": [],
                "importance": 0.0,
                "build_time": 0.0,
            }

    def _detect_conflict(
        self, node1: int, node2: int, similarity: float
    ) -> Optional[Dict[str, Any]]:
        """Detect semantic conflicts between highly similar nodes."""
        try:
            # Get metadata for both nodes
            meta1 = self.metadata[node1]
            meta2 = self.metadata[node2]

            # Check for directional conflicts (opposing information)
            text1 = meta1.get("text", "")
            text2 = meta2.get("text", "")

            # Simple conflict detection heuristics
            conflict_indicators = [
                ("increase", "decrease"),
                ("positive", "negative"),
                ("success", "failure"),
                ("true", "false"),
                ("yes", "no"),
            ]

            for pos, neg in conflict_indicators:
                if (pos in text1.lower() and neg in text2.lower()) or (
                    neg in text1.lower() and pos in text2.lower()
                ):
                    return {
                        "type": "directional",
                        "nodes": [node1, node2],
                        "similarity": similarity,
                        "indicator": (pos, neg),
                    }

            # Check for temporal conflicts
            time1 = meta1.get("timestamp")
            time2 = meta2.get("timestamp")
            if time1 and time2 and abs(time1 - time2) < 1.0:
                # Very close in time but different content
                return {
                    "type": "temporal",
                    "nodes": [node1, node2],
                    "similarity": similarity,
                    "time_diff": abs(time1 - time2),
                }

        except Exception as e:
            logger.warning(f"Conflict detection failed: {e}")

        return None

    def _calculate_node_importance(self, node_id: int) -> float:
        """Calculate graph-based importance for a node."""
        try:
            # Degree centrality as a simple importance metric
            if self.graph.edge_index.numel() == 0:
                return 0.0

            # Count edges for this node
            edge_mask = (self.graph.edge_index[0] == node_id) | (
                self.graph.edge_index[1] == node_id
            )
            degree = edge_mask.sum().item()

            # Normalize by max possible degree
            max_degree = (self.graph.num_nodes - 1) * 2  # Bidirectional edges
            if max_degree > 0:
                return degree / max_degree

            return 0.0

        except Exception as e:
            logger.warning(f"Importance calculation failed: {e}")
            return 0.0

    def get_current_graph(self) -> Data:
        """Get the current graph state."""
        return self.graph

    def get_subgraph(self, indices: List[int]) -> Data:
        """Get subgraph for specified nodes using FAISS for edge discovery."""
        if not indices or self.graph.x.numel() == 0:
            return self._initialize_empty_graph()

        try:
            # Get node features
            node_tensor = torch.tensor(indices, dtype=torch.long)
            x = self.graph.x[node_tensor]

            # Find edges within subgraph
            edge_list = []
            for i, idx in enumerate(indices):
                # Get embedding
                emb = self.embeddings[idx]

                # Search only within subgraph nodes
                subgraph_embeddings = np.array([self.embeddings[j] for j in indices])

                # Create temporary index for subgraph
                temp_index = VectorIndexFactory.create_index(
                    dimension=self.embedding_dim,
                    index_type="auto"
                )
                temp_index.add(subgraph_embeddings.astype(np.float32))

                # Find neighbors
                k = min(self.top_k, len(indices))
                distances, neighbors = temp_index.search(
                    emb.reshape(1, -1).astype(np.float32), k
                )

                # Add edges above threshold
                for dist, neigh in zip(distances[0], neighbors[0]):
                    if neigh != i and dist > self.similarity_threshold:
                        edge_list.append([i, neigh])

            # Create edge tensor
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            sub = Data(x=x, edge_index=edge_index)
            sub.num_nodes = x.size(0)
            return sub

        except Exception as e:
            logger.error(f"Subgraph extraction failed: {e}")
            return self._initialize_empty_graph()

    def should_split_episode(self, conflicts: List[Dict[str, Any]]) -> bool:
        """Determine if conflicts warrant episode splitting."""
        if not conflicts:
            return False

        # Count serious conflicts
        serious_conflicts = sum(
            1
            for c in conflicts
            if c.get("type") == "directional" or c.get("similarity", 0) > 0.9
        )

        # Split if multiple serious conflicts
        return serious_conflicts >= 2

    def get_split_candidates(self, node_id: int) -> List[Tuple[int, float]]:
        """Get candidate nodes for splitting based on conflicts."""
        if not self.conflict_history:
            return []

        # Find nodes involved in conflicts with this node
        candidates = []
        for conflict in self.conflict_history[-10:]:  # Last 10 conflicts
            if node_id in conflict.get("nodes", []):
                other_node = [n for n in conflict["nodes"] if n != node_id][0]
                similarity = conflict.get("similarity", 0)
                candidates.append((other_node, similarity))

        # Sort by similarity (higher similarity = stronger candidate)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:3]  # Top 3 candidates

    def update_from_episodes(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rebuild graph from updated episode list."""
        try:
            # Clear current state
            self._initialize_empty_graph()
            self.vector_index = None
            self.embeddings = []
            self.metadata = []

            # Rebuild from episodes
            total_edges = 0
            total_conflicts = 0

            for i, episode in enumerate(episodes):
                embedding = episode.get("embedding", episode.get("vector"))
                if embedding is not None:
                    result = self.add_episode_node(
                        embedding=np.array(embedding),
                        index=i,
                        metadata={
                            "text": episode.get("text", ""),
                            "timestamp": episode.get("timestamp"),
                            "c_value": episode.get("c_value", 0.5),
                        },
                    )

                    if result["success"]:
                        total_edges += result["edges_added"]
                        total_conflicts += len(result["conflicts"])

            return {
                "success": True,
                "graph_stats": {
                    "nodes": self.graph.num_nodes,
                    "edges": total_edges,
                    "conflicts": total_conflicts,
                    "density": total_edges
                    / (self.graph.num_nodes * (self.graph.num_nodes - 1))
                    if self.graph.num_nodes > 1
                    else 0,
                },
            }

        except Exception as e:
            logger.error(f"Failed to update from episodes: {e}")
            return {"success": False, "error": str(e)}

    def save_index(self, path: str):
        """Save FAISS index to disk."""
        # Vector index persistence is not implemented yet; stub for future extension.
        logger.warning("Vector index persistence not implemented; skipping save_index call")

    def load_index(self, path: str):
        """Load FAISS index from disk."""
        try:
            # Vector index loading is not implemented; keep this as a no-op stub.
            logger.warning("Vector index loading not implemented; skipping load_index call")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.vector_index = None
