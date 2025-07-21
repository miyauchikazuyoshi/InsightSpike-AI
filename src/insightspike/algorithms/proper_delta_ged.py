"""
Proper ΔGED Implementation for InsightSpike
==========================================

This module provides the corrected ΔGED calculation that maintains
a persistent reference graph for proper insight detection.

Key fix: ΔGED = GED(current, initial) - GED(previous, initial)
Where initial is the first graph in the reasoning sequence.
"""

import logging
from typing import Any, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class ProperDeltaGED:
    """
    Maintains reference graph state for proper ΔGED calculation.

    The key insight is that we need to track distance from an initial
    reference state, not just compare consecutive states.
    """

    def __init__(self):
        self.initial_graph = None
        self.previous_graph = None

    def calculate_delta_ged(
        self,
        current_graph: Any,
        previous_graph: Optional[Any] = None,
        reference_graph: Optional[Any] = None,
    ) -> float:
        """
        Calculate proper ΔGED.

        Args:
            current_graph: Current graph state
            previous_graph: Previous graph state (uses stored if not provided)
            reference_graph: Reference graph (uses initial if not provided)

        Returns:
            float: ΔGED value (negative indicates insight/simplification)
        """
        # Handle reference graph
        if reference_graph is not None:
            self.initial_graph = reference_graph
        elif self.initial_graph is None:
            # First call - use previous as initial
            self.initial_graph = previous_graph or current_graph
            self.previous_graph = self.initial_graph
            return 0.0  # No change on first call

        # Handle previous graph
        if previous_graph is None:
            if self.previous_graph is None:
                self.previous_graph = self.initial_graph
            previous_graph = self.previous_graph

        try:
            # Calculate GED from current to initial
            ged_current = self._calculate_ged(current_graph, self.initial_graph)

            # Calculate GED from previous to initial
            ged_previous = self._calculate_ged(previous_graph, self.initial_graph)

            # ΔGED = GED(current, initial) - GED(previous, initial)
            delta_ged = float(ged_current - ged_previous)

            logger.info(
                f"Proper ΔGED: current→initial={ged_current}, "
                f"previous→initial={ged_previous}, ΔGED={delta_ged}"
            )

            # Update state for next calculation
            self.previous_graph = current_graph

            return delta_ged

        except Exception as e:
            logger.error(f"ΔGED calculation failed: {e}")
            return 0.0

    def _calculate_ged(self, g1: Any, g2: Any) -> float:
        """Calculate GED between two graphs."""
        # For small graphs, try exact calculation
        if (
            hasattr(g1, "number_of_nodes")
            and hasattr(g2, "number_of_nodes")
            and g1.number_of_nodes() <= 10
            and g2.number_of_nodes() <= 10
        ):
            try:
                # Use generator to get first (best) result
                ged_gen = nx.optimize_graph_edit_distance(g1, g2, upper_bound=100)
                return float(next(ged_gen))
            except:
                pass

        # Fallback to simple approximation
        return self._approximate_ged(g1, g2)

    def _approximate_ged(self, g1: Any, g2: Any) -> float:
        """Simple GED approximation based on node/edge differences."""
        try:
            n1 = g1.number_of_nodes() if hasattr(g1, "number_of_nodes") else 0
            n2 = g2.number_of_nodes() if hasattr(g2, "number_of_nodes") else 0
            e1 = g1.number_of_edges() if hasattr(g1, "number_of_edges") else 0
            e2 = g2.number_of_edges() if hasattr(g2, "number_of_edges") else 0

            node_diff = abs(n1 - n2)
            edge_diff = abs(e1 - e2)

            return float(node_diff + edge_diff)
        except:
            return 0.0

    def reset(self):
        """Reset the reference state."""
        self.initial_graph = None
        self.previous_graph = None


# Global instance for stateful calculations
_proper_ged_instance = ProperDeltaGED()


def proper_delta_ged(
    current_graph: Any,
    previous_graph: Optional[Any] = None,
    reference_graph: Optional[Any] = None,
) -> float:
    """
    Calculate proper ΔGED with persistent reference state.

    This is a convenience function that uses a global instance
    to maintain state across calls.
    """
    return _proper_ged_instance.calculate_delta_ged(
        current_graph, previous_graph, reference_graph
    )


def reset_ged_state():
    """Reset the global GED calculator state."""
    _proper_ged_instance.reset()
