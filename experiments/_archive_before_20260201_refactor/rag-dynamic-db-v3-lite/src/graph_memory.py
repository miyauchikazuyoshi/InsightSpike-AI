"""Graph memory that exposes networkx snapshots for geDIG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .retriever import RetrievalHit


@dataclass
class MemoryNode:
    node_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: int = 0
    activation: float = 1.0


@dataclass
class MemoryEdge:
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "retrieval"
    metadata: Dict[str, str] = field(default_factory=dict)


class GraphMemory:
    """Dynamic graph memory with helper exports for geDIG."""

    def __init__(self, latent_dim: int = 384) -> None:
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, MemoryNode] = {}
        self.latent_dim = latent_dim
        self.timestamp = 0

    def _ensure_embedding(self, embedding: Optional[np.ndarray]) -> np.ndarray:
        if embedding is None:
            return np.zeros(self.latent_dim, dtype=np.float32)
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        size = arr.size
        if size == self.latent_dim:
            return arr
        if size > self.latent_dim:
            return arr[: self.latent_dim]
        return np.pad(arr, (0, self.latent_dim - size), mode="constant")

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            content=node.content,
            feature=node.embedding,
            activation=node.activation,
            timestamp=node.timestamp,
            metadata=node.metadata,
        )

    def add_edge(self, edge: MemoryEdge) -> None:
        self.graph.add_edge(
            edge.source,
            edge.target,
            weight=edge.weight,
            edge_type=edge.edge_type,
            metadata=edge.metadata,
        )

    def update_from_retrieval(
        self,
        query: str,
        retrieved: Iterable[RetrievalHit],
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[str, List[str]]:
        hits = list(retrieved)
        query_id = f"query_{self.timestamp}"
        query_vec = self._ensure_embedding(query_embedding)
        query_node = MemoryNode(
            node_id=query_id,
            content=query,
            embedding=query_vec,
            metadata={"type": "query"},
            timestamp=self.timestamp,
        )
        self.add_node(query_node)

        added_docs: List[str] = []

        for hit in hits:
            doc = hit.document
            doc_embedding = self._ensure_embedding(doc.embedding)
            metadata = {"type": "document"}
            if doc.metadata:
                for key, value in doc.metadata.items():
                    metadata[key] = value
            metadata.update(
                {
                    "score": f"{hit.score:.6f}",
                    "bm25_score": f"{hit.bm25_score:.6f}",
                    "embedding_score": f"{hit.embedding_score:.6f}",
                    "rank": str(hit.rank),
                }
            )
            if doc.doc_id not in self.nodes:
                doc_node = MemoryNode(
                    node_id=doc.doc_id,
                    content=doc.text,
                    embedding=doc_embedding,
                    metadata=metadata,
                    timestamp=self.timestamp,
                    activation=1.0,
                )
                self.add_node(doc_node)
                added_docs.append(doc.doc_id)
            else:
                existing = self.nodes[doc.doc_id]
                existing.activation = float(min(existing.activation * 0.9 + max(hit.score, 0.1), 5.0))
                existing.metadata.update(metadata)
                self.graph.nodes[doc.doc_id]["activation"] = existing.activation
                self.graph.nodes[doc.doc_id]["feature"] = doc_embedding
                self.graph.nodes[doc.doc_id]["metadata"] = existing.metadata

            self.add_edge(
                MemoryEdge(
                    source=query_id,
                    target=doc.doc_id,
                    weight=max(0.05, float(hit.score)),
                    edge_type="retrieval",
                    metadata={
                        "score": f"{hit.score:.6f}",
                        "bm25_score": f"{hit.bm25_score:.6f}",
                        "embedding_score": f"{hit.embedding_score:.6f}",
                    },
                )
            )

        # strengthen co-occurrence edges
        doc_ids = [hit.document.doc_id for hit in hits]
        for i, src in enumerate(doc_ids):
            for dst in doc_ids[i + 1 :]:
                if self.graph.has_edge(src, dst):
                    self.graph[src][dst]["weight"] *= 1.1
                else:
                    self.add_edge(MemoryEdge(source=src, target=dst, weight=0.5, edge_type="co_occurrence"))

        self.timestamp += 1
        return query_id, doc_ids

    def compute_metrics(self) -> Dict[str, float]:
        if self.graph.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "clustering": 0.0,
            }

        undirected = self.graph.to_undirected()
        avg_degree = float(np.mean([d for _, d in self.graph.degree()]))
        clustering = float(nx.average_clustering(undirected)) if undirected.number_of_nodes() > 1 else 0.0

        return {
            "num_nodes": float(self.graph.number_of_nodes()),
            "num_edges": float(self.graph.number_of_edges()),
            "density": float(nx.density(self.graph)),
            "avg_degree": avg_degree,
            "clustering": clustering,
        }

    def export_for_gedig(self) -> Tuple[nx.Graph, np.ndarray]:
        undirected = self.graph.to_undirected()
        features: List[np.ndarray] = []
        for node in undirected.nodes():
            data = self.graph.nodes[node]
            feature_vec = data.get("feature")
            if feature_vec is None:
                feature_vec = np.zeros(self.latent_dim, dtype=np.float32)
            features.append(np.asarray(feature_vec, dtype=np.float32))
        feature_matrix = np.vstack(features) if features else np.zeros((0, self.latent_dim), dtype=np.float32)
        return undirected.copy(), feature_matrix

    def decay(self, factor: float = 0.95) -> None:
        for node_id, node in self.nodes.items():
            node.activation *= factor
            if node.activation < 0.01:
                node.activation = 0.01
            self.graph.nodes[node_id]["activation"] = node.activation
