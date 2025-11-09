"""Minimal dynamic graph memory for geDIG export (self-contained)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from .retriever import RetrievalHit


@dataclass
class MemoryNode:
    node_id: str
    text: str
    embedding: np.ndarray
    activation: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)


class GraphMemory:
    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.nodes: Dict[str, MemoryNode] = {}

    def decay(self, factor: float = 0.95) -> None:
        for node in self.nodes.values():
            node.activation *= factor
            data = self.graph.nodes.get(node.node_id, {})
            if isinstance(data, dict):
                data["activation"] = node.activation

    def update_from_retrieval(
        self,
        query: str,
        retrieved: Iterable[RetrievalHit],
        query_embedding: Optional[np.ndarray] = None,
    ) -> None:
        hits = list(retrieved)
        qid = f"q::{abs(hash(query)) % (10**9)}"
        qvec = np.asarray(query_embedding if query_embedding is not None else np.zeros((384,), dtype=np.float32), dtype=np.float32)
        if qid not in self.nodes:
            qnode = MemoryNode(node_id=qid, text=query, embedding=qvec, activation=1.0, metadata={"type": "query"})
            self.nodes[qid] = qnode
            self.graph.add_node(qid, feature=qvec, activation=1.0, metadata=qnode.metadata)
        for hit in hits:
            doc = hit.document
            if doc.doc_id not in self.nodes:
                meta = dict(doc.metadata)
                if "role" not in meta:
                    meta["role"] = "support"
                mnode = MemoryNode(node_id=doc.doc_id, text=doc.text, embedding=doc.embedding, activation=1.0, metadata=meta)
                self.nodes[doc.doc_id] = mnode
                self.graph.add_node(doc.doc_id, feature=doc.embedding, activation=1.0, metadata=meta)
            # link query to doc
            self.graph.add_edge(qid, doc.doc_id, weight=max(0.0, hit.score))

    def export_for_gedig(self) -> Tuple[nx.Graph, np.ndarray]:
        # features: stack node feature in node order
        feats: List[np.ndarray] = []
        for node_id in self.graph.nodes:
            data = self.graph.nodes.get(node_id, {})
            vec = data.get("feature")
            if vec is None:
                vec = np.zeros((384,), dtype=np.float32)
            feats.append(np.asarray(vec, dtype=np.float32))
        features = np.stack(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32)
        return self.graph.copy(), features

