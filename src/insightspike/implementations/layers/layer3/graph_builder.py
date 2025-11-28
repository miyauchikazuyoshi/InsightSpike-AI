"""GraphBuilder for constructing PyG graphs from documents."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch  # type: ignore  # noqa: F401
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover
    class Data:  # minimal fallback
        def __init__(self, x=None, edge_index=None, **kwargs):
            self.x = x
            self.edge_index = edge_index
            self.num_nodes = getattr(x, "shape", [0])[0] if x is not None else 0

from ....config import get_config
from ....config.legacy_adapter import LegacyConfigAdapter

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    if b is None:
        b = a
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class GraphBuilder:
    """Build and manage PyTorch Geometric graphs from documents."""

    def __init__(self, config=None):
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        self.similarity_threshold = self.config.graph.similarity_threshold

    def build_graph(
        self, documents: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None
    ) -> Data:
        """Build a graph from documents and their embeddings."""
        if not documents:
            return self._empty_graph()

        try:
            if embeddings is None:
                embeddings = self._get_embeddings(documents)

            if len(documents) < 3:
                edge_list = []
                if len(documents) == 2:
                    edge_list = [[0, 1], [1, 0]]
                elif len(documents) == 1:
                    edge_list = [[0, 0]]
            else:
                sim_matrix = _cosine_similarity(embeddings)

                edge_list: List[List[int]] = []
                for i in range(len(documents)):
                    for j in range(i + 1, len(documents)):
                        if sim_matrix[i, j] > self.similarity_threshold:
                            edge_list.extend([[i, j], [j, i]])

                if not edge_list:
                    edge_list = [[i, i + 1] for i in range(len(documents) - 1)]
                    edge_list.extend([[i + 1, i] for i in range(len(documents) - 1)])

            if not edge_list and len(documents) > 0:
                edge_list = [[0, 0]]

            import torch as _torch
            from torch_geometric.data import Data as _Data

            edge_index = _torch.tensor(edge_list, dtype=_torch.long).t().contiguous()
            x = _torch.tensor(embeddings, dtype=_torch.float)

            graph = _Data(x=x, edge_index=edge_index)
            graph.num_nodes = len(documents)
            graph.documents = documents

            logger.debug(
                "Built graph with %s nodes, %s edges",
                graph.num_nodes,
                graph.edge_index.size(1),
            )
            return graph

        except Exception as e:  # noqa: BLE001
            logger.error("Graph building failed: %s", e)
            return self._empty_graph()

    def _get_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Extract or compute embeddings for documents."""
        embeddings: List[np.ndarray] = []
        for doc in documents:
            if "embedding" in doc:
                embeddings.append(doc["embedding"])
            else:
                simple_emb = np.random.random(384)
                embeddings.append(simple_emb)
        return np.array(embeddings)

    def _empty_graph(self) -> Data:
        """Create an empty graph for error cases."""
        try:
            import torch as _torch
            from torch_geometric.data import Data as _Data

            return _Data(
                x=_torch.empty(0, 384),
                edge_index=_torch.empty(2, 0, dtype=_torch.long),
            )
        except Exception:  # pragma: no cover
            import numpy as _np
            from torch_geometric.data import Data as _Data

            return _Data(
                x=_np.empty((0, 384)), edge_index=_np.empty((2, 0), dtype=int)
            )


__all__ = ["GraphBuilder"]
