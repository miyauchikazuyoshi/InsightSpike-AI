"""Controller for message passing and edge reevaluation (Layer3 refactor)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ....config import get_config
from ....config.legacy_adapter import LegacyConfigAdapter
from .message_passing import EdgeReevaluator, MessagePassing

logger = logging.getLogger(__name__)


class MessagePassingController:
    """Initialize and hold message passing components based on config."""

    def __init__(self, config=None, original_config: Optional[Dict[str, Any]] = None):
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        self._original_config = original_config or {}
        self.message_passing_enabled = False
        self.message_passing: Optional[Any] = None
        self.edge_reevaluator: Optional[Any] = None

    def initialize(self) -> None:
        """Configure message passing and edge reevaluation."""
        try:
            # Config flags (dict or pydantic)
            if isinstance(self._original_config, dict):
                self.message_passing_enabled = self._original_config.get("graph", {}).get(
                    "enable_message_passing", False
                )
                mp_config = self._original_config.get("graph", {}).get("message_passing", {})
                er_config = self._original_config.get("graph", {}).get("edge_reevaluation", {})
            else:
                self.message_passing_enabled = getattr(
                    getattr(self.config, "graph", None), "enable_message_passing", False
                )
                mp_config = getattr(getattr(self.config, "graph", None), "message_passing", {}) or {}
                er_config = getattr(getattr(self.config, "graph", None), "edge_reevaluation", {}) or {}

            if not self.message_passing_enabled:
                logger.info("Message passing disabled in config")
                return

            # Choose implementation
            use_optimized = mp_config.get("enable_batch_computation", True)
            max_hops = mp_config.get("max_hops", 1)

            if use_optimized:
                from ...graph.message_passing_optimized import OptimizedMessagePassing

                self.message_passing = OptimizedMessagePassing(
                    alpha=mp_config.get("alpha", 0.3),
                    iterations=mp_config.get("iterations", 2),
                    max_hops=max_hops,
                    aggregation=mp_config.get("aggregation", "weighted_mean"),
                    self_loop_weight=mp_config.get("self_loop_weight", 0.5),
                    decay_factor=mp_config.get("decay_factor", 0.8),
                    convergence_threshold=mp_config.get("convergence_threshold", 1e-4),
                    similarity_threshold=mp_config.get("similarity_threshold", 0.3),
                )
                logger.info("Using OptimizedMessagePassing with max_hops=%s", max_hops)
            else:
                self.message_passing = MessagePassing(
                    alpha=mp_config.get("alpha", 0.3),
                    iterations=mp_config.get("iterations", 2),
                    aggregation=mp_config.get("aggregation", "weighted_mean"),
                    self_loop_weight=mp_config.get("self_loop_weight", 0.5),
                    decay_factor=mp_config.get("decay_factor", 0.8),
                )

            self.edge_reevaluator = EdgeReevaluator(
                similarity_threshold=er_config.get("similarity_threshold", 0.7),
                new_edge_threshold=er_config.get("new_edge_threshold", 0.8),
                max_new_edges_per_node=er_config.get("max_new_edges_per_node", 5),
                edge_decay_factor=er_config.get("edge_decay_factor", 0.9),
            )
            logger.info("Message passing components initialized")

        except Exception as exc:  # noqa: BLE001
            logger.warning("Message passing initialization failed: %s", exc)
            self.message_passing_enabled = False
            self.message_passing = None
            self.edge_reevaluator = None

    def apply(self, graph: Any, query_vector: Any):
        """Run message passing + edge reevaluation if enabled; otherwise return original graph."""
        if not self.message_passing_enabled or self.message_passing is None:
            return graph
        if query_vector is None:
            return graph
        updated_representations = None
        try:
            updated_representations = self.message_passing.forward(graph, query_vector)
        except Exception as exc:  # pragma: no cover
            logger.warning("Message passing forward failed: %s", exc)
            return graph

        # Edge reevaluation is optional; if unavailable, just return graph
        if self.edge_reevaluator is None:
            return graph
        try:
            return self.edge_reevaluator.reevaluate(
                graph, updated_representations, query_vector, return_edge_scores=True
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Edge reevaluation failed: %s", exc)
            return graph


__all__ = ["MessagePassingController"]
