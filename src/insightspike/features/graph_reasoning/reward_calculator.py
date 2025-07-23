"""
Reward Calculator
=================

Calculates rewards for graph reasoning and memory updates.
Separated from L3GraphReasoner to follow Single Responsibility Principle.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Calculates reward signals for memory updates."""

    def __init__(self, config=None):
        self.config = config or {}

        # Import defaults
        from ...config.constants import Defaults

        # Extract weights from config with defaults from constants
        if hasattr(config, "graph"):
            self.weights = {
                "ged": getattr(config.graph, "weight_ged", Defaults.REWARD_WEIGHT_GED),
                "ig": getattr(config.graph, "weight_ig", Defaults.REWARD_WEIGHT_IG),
            }
            # Temperature parameter for geDIG formula
            self.temperature = getattr(
                config.graph, "temperature", Defaults.REWARD_TEMPERATURE
            )
        else:
            self.weights = {
                "ged": Defaults.REWARD_WEIGHT_GED,
                "ig": Defaults.REWARD_WEIGHT_IG,
            }
            self.temperature = Defaults.REWARD_TEMPERATURE

        # Optimal graph size for structure reward
        self.optimal_graph_size = (
            getattr(config.graph, "optimal_graph_size", Defaults.OPTIMAL_GRAPH_SIZE)
            if hasattr(config, "graph")
            else Defaults.OPTIMAL_GRAPH_SIZE
        )

    def calculate_reward(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate multi-component reward signal using geDIG formula.

        geDIG formula: F = w1*ΔGED - kT*ΔIG

        This implements the brain-inspired trade-off between structural
        simplification (ΔGED) and information gain (ΔIG), where temperature
        controls the exploration-exploitation balance.
        """
        # Base reward calculation: F = w1*ΔGED - kT*ΔIG
        # Note: kT = weight_ig * temperature in our parameterization
        base_reward = self.weights["ged"] * metrics.get("delta_ged", 0) - self.weights[
            "ig"
        ] * self.temperature * metrics.get("delta_ig", 0)

        # Additional reward components
        structure_reward = self._calculate_structure_reward(metrics)
        novelty_reward = self._calculate_novelty_reward(metrics, conflicts)

        return {
            "base": float(base_reward),
            "structure": float(structure_reward),
            "novelty": float(novelty_reward),
            "total": float(base_reward + structure_reward + novelty_reward),
        }

    def _calculate_structure_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for good graph structure (not too sparse, not too dense)."""
        current_size = metrics.get("graph_size_current", 0)
        if current_size == 0:
            return 0.0

        # Penalize deviation from optimal size
        size_penalty = (
            abs(current_size - self.optimal_graph_size) / self.optimal_graph_size
        )
        return max(0.0, 1.0 - size_penalty)

    def _calculate_novelty_reward(
        self, metrics: Dict[str, float], conflicts: Dict[str, float]
    ) -> float:
        """Reward for novel insights."""
        novelty = metrics.get("delta_ig", 0)
        # Conflicts parameter kept for backward compatibility but ignored
        return max(0.0, novelty)
