"""GNN helpers for Layer3 (optional, falls back to None when torch/PyG unavailable)."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_simple_gnn(input_dim: int, hidden_dim: int):
    """Return a simple 3-layer GCN or None if torch/PyG missing."""
    try:
        import torch  # type: ignore
        from torch_geometric.nn import GCNConv  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("GNN unavailable (torch/PyG import failed): %s", exc)
        return None

    class _SimpleGNN(torch.nn.Module):
        def __init__(self, in_dim: int, hid: int):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hid)
            self.act1 = torch.nn.ReLU()
            self.conv2 = GCNConv(hid, hid)
            self.act2 = torch.nn.ReLU()
            self.conv3 = GCNConv(hid, in_dim)

        def forward(self, x, edge_index):  # type: ignore[override]
            x = self.conv1(x, edge_index)
            x = self.act1(x)
            x = self.conv2(x, edge_index)
            x = self.act2(x)
            x = self.conv3(x, edge_index)
            return x

    return _SimpleGNN(input_dim, hidden_dim)


__all__ = ["build_simple_gnn"]
