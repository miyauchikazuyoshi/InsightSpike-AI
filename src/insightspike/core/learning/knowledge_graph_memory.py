"""Knowledge Graph Memory using PyTorch Geometric."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try importing torch-geometric, with fallback for environments where it's not available
try:
    from torch_geometric.data import Data
    from torch_geometric.utils import subgraph
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch-geometric not available, using basic torch functionality")
    
    # Fallback implementation for when torch-geometric is not available
    class Data:
        """Fallback Data class for basic graph functionality without torch-geometric."""
        def __init__(self, x=None, edge_index=None):
            self.x = x if x is not None else torch.empty((0, 0))
            self.edge_index = edge_index if edge_index is not None else torch.empty((2, 0), dtype=torch.long)
            self.num_nodes = self.x.size(0) if self.x.numel() > 0 else 0
    
    def subgraph(node_tensor, edge_index, relabel_nodes=True, num_nodes=None):
        """Fallback subgraph function for basic functionality without torch-geometric."""
        # Simple fallback: return empty edge_index for subgraph
        return torch.empty((2, 0), dtype=torch.long), None


class KnowledgeGraphMemory:
    """Persistent knowledge graph storing episode embeddings."""

    def __init__(self, embedding_dim: int, similarity_threshold: float = 0.2):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.embeddings: List[np.ndarray] = []
        self.graph = Data(x=torch.empty((0, embedding_dim)), edge_index=torch.empty((2, 0), dtype=torch.long))

    def add_episode_node(self, embedding: np.ndarray, index: int) -> None:
        """Add a node for a memory episode and connect it to similar nodes."""
        try:
            embedding = embedding.astype(np.float32)
            new_x = torch.tensor(embedding).view(1, -1)
            if self.graph.x.numel() == 0:
                self.graph.x = new_x
                self.graph.edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # append node feature
                self.graph.x = torch.cat([self.graph.x, new_x], dim=0)

            self.embeddings.append(embedding)
            node_id = len(self.embeddings) - 1

            # compute similarity to existing nodes
            if node_id > 0:
                existing = np.vstack(self.embeddings[:-1])
                sims = existing @ embedding / (np.linalg.norm(existing, axis=1) * np.linalg.norm(embedding) + 1e-8)
                edges = []
                for i, sim in enumerate(sims):
                    if sim > self.similarity_threshold:
                        edges.extend([[i, node_id], [node_id, i]])
                if edges:
                    new_edges = torch.tensor(edges, dtype=torch.long).t()
                    if self.graph.edge_index.numel() == 0:
                        self.graph.edge_index = new_edges
                    else:
                        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edges], dim=1)
        except Exception as e:
            logger.error(f"Failed to add episode node: {e}")

    def remove_episode_node(self, index: int) -> None:
        """Remove a node by rebuilding the graph without it."""
        if 0 <= index < len(self.embeddings):
            del self.embeddings[index]
            # rebuild graph
            self.graph = Data(x=torch.empty((0, self.embedding_dim)), edge_index=torch.empty((2, 0), dtype=torch.long))
            for i, emb in enumerate(self.embeddings):
                self.add_episode_node(emb, i)

    def get_subgraph(self, indices: List[int]) -> Data:
        """Return an induced subgraph of the given node indices."""
        if not indices or self.graph.x.numel() == 0:
            return Data(x=torch.empty((0, self.embedding_dim)), edge_index=torch.empty((2, 0), dtype=torch.long))
        
        node_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Handle case where there are no edges
        if self.graph.edge_index.numel() == 0:
            x = self.graph.x[node_tensor]
            sub = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
            sub.num_nodes = x.size(0)
            return sub
        
        edge_index, _ = subgraph(node_tensor, self.graph.edge_index, relabel_nodes=True, num_nodes=self.graph.x.size(0))
        x = self.graph.x[node_tensor]
        sub = Data(x=x, edge_index=edge_index)
        sub.num_nodes = x.size(0)
        return sub
