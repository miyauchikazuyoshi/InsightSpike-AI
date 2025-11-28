"""Conflict scoring utilities for Layer3 graph reasoner."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

try:
    import torch  # type: ignore  # noqa: F401
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover
    class Data:  # minimal fallback for type compatibility
        def __init__(self, x=None, edge_index=None, **kwargs):
            self.x = x
            self.edge_index = edge_index
            self.num_nodes = getattr(x, "shape", [0])[0] if x is not None else 0

from ....config import get_config
from ....config.legacy_adapter import LegacyConfigAdapter

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(a_norm @ b_norm.T)  # shape (1,1) expected in this module


class ConflictScore:
    """Conflict detection and scoring for graph reasoning."""

    def __init__(self, config=None):
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        self.conflict_threshold = self.config.graph.conflict_threshold

    def calculate_conflict(
        self, graph_old: Data, graph_new: Data, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate conflict scores between graphs."""
        try:
            structural_conflict = self._structural_conflict(graph_old, graph_new)
            semantic_conflict = self._semantic_conflict(graph_old, graph_new)
            temporal_conflict = self._temporal_conflict(context)
            total = (structural_conflict + semantic_conflict + temporal_conflict) / 3
            return {
                "structural": float(structural_conflict),
                "semantic": float(semantic_conflict),
                "temporal": float(temporal_conflict),
                "total": float(total),
            }
        except Exception as e:  # noqa: BLE001
            logger.error("Conflict calculation failed: %s", e)
            return {"structural": 0.0, "semantic": 0.0, "temporal": 0.0, "total": 0.0}

    def _structural_conflict(self, graph_old: Data, graph_new: Data) -> float:
        """Calculate structural differences between graphs."""
        if graph_old is None or graph_new is None:
            return 0.0

        edge_diff = abs(graph_old.edge_index.size(1) - graph_new.edge_index.size(1))
        node_diff = abs(graph_old.x.size(0) - graph_new.x.size(0))

        max_edges = max(graph_old.edge_index.size(1), graph_new.edge_index.size(1), 1)
        max_nodes = max(graph_old.x.size(0), graph_new.x.size(0), 1)

        return (edge_diff / max_edges + node_diff / max_nodes) / 2

    def _semantic_conflict(self, graph_old: Data, graph_new: Data) -> float:
        """Calculate semantic differences in node features."""
        if graph_old is None or graph_new is None:
            return 0.0

        try:
            old_features = (
                graph_old.x.cpu().numpy()
                if hasattr(graph_old.x, "cpu")
                else graph_old.x.numpy()
            )
            new_features = (
                graph_new.x.cpu().numpy()
                if hasattr(graph_new.x, "cpu")
                else graph_new.x.numpy()
            )

            if old_features.size == 0 or new_features.size == 0:
                return 0.0

            if old_features.shape[1] == new_features.shape[1]:
                old_mean = np.mean(old_features, axis=0, keepdims=True)
                new_mean = np.mean(new_features, axis=0, keepdims=True)

                old_norm = np.linalg.norm(old_mean)
                new_norm = np.linalg.norm(new_mean)

                if old_norm == 0 or new_norm == 0:
                    return 0.0

                similarity = _cosine_similarity(old_mean, new_mean)
                if not np.isfinite(similarity):
                    return 0.0

                return float(1.0 - similarity)

        except Exception as e:  # noqa: BLE001
            logger.warning("Semantic conflict calculation failed: %s", e)

        return 0.0

    def _temporal_conflict(self, context: Dict[str, Any]) -> float:
        """Calculate temporal inconsistencies."""
        if "previous_confidence" in context and "current_confidence" in context:
            conf_diff = abs(
                context["previous_confidence"] - context["current_confidence"]
            )
            return min(conf_diff, 1.0)
        return 0.0


__all__ = ["ConflictScore"]
