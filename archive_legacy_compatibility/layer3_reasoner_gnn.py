"""
L3 GraphSAGE Retrieval - Compatibility Layer
==========================================

This module provides backward compatibility for the old GraphSAGE functions.
New code should use: from insightspike.core.layers.layer3_graph_reasoner import L3GraphReasoner

DEPRECATED: This file will be removed in a future version.
Use the new structured approach in core/layers/ instead.
"""
from __future__ import annotations
import warnings

# Issue deprecation warning
warnings.warn(
    "insightspike.layer3_reasoner_gnn is deprecated. "
    "Use insightspike.core.layers.layer3_graph_reasoner instead.",
    DeprecationWarning,
    stacklevel=2
)
import faiss, numpy as np, torch
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

from .embedder            import get_model
from .layer3_graph_pyg    import load_graph
from .loader              import load_corpus
from .config              import LAYER3_TOP_K

__all__ = ["retrieve_gnn"]

class _GnnEnc(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = SAGEConv(dim, dim)
    def forward(self, d: Data):
        return torch.nn.functional.normalize(self.conv(d.x, d.edge_index))

def retrieve_gnn(question: str):
    model = get_model()
    q_vec = model.encode([question], normalize_embeddings=True)
    data = load_graph()
    enc = _GnnEnc(data.num_node_features).eval()
    with torch.no_grad():
        doc_vecs = enc(data).cpu().numpy()
    idx = faiss.IndexFlatIP(doc_vecs.shape[1]); idx.add(doc_vecs)
    scores, ids = idx.search(q_vec.astype(np.float32), LAYER3_TOP_K)
    corpus = load_corpus()
    return ids[0].tolist(), scores[0].tolist(), corpus
