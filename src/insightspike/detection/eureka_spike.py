"""EurekaSpike detection logic - the core insight mechanism."""

from __future__ import annotations

import importlib.util
import logging

# Import from the legacy config.py file
import os
from typing import Any, Dict, List, Tuple

try:
    # Import from the unified config system
    from ..config import get_legacy_config

    legacy_config = get_legacy_config()
    SPIKE_GED = legacy_config.get("SPIKE_GED", 0.5)
    SPIKE_IG = legacy_config.get("SPIKE_IG", 0.2)
    ETA_SPIKE = legacy_config.get("ETA_SPIKE", 0.2)
except (ImportError, AttributeError):
    # Fallback for testing or if config is not available
    SPIKE_GED = 0.5
    SPIKE_IG = 0.2
    ETA_SPIKE = 0.2

__all__ = ["EurekaDetector", "detect_eureka_spike"]

logger = logging.getLogger(__name__)


class EurekaDetector:
    """Detects insight spikes based on ΔGED and ΔIG patterns."""

    def __init__(
        self,
        ged_threshold: float = SPIKE_GED,
        ig_threshold: float = SPIKE_IG,
        eta_spike: float = ETA_SPIKE,
    ):
        """
        Initialize EurekaSpike detector.

        Args:
            ged_threshold: ΔGED threshold for spike detection (default: 0.5)
            ig_threshold: ΔIG threshold for spike detection (default: 0.2)
            eta_spike: Learning rate for C-value updates during spikes (default: 0.2)
        """
        self.ged_threshold = ged_threshold
        self.ig_threshold = ig_threshold
        self.eta_spike = eta_spike

        # History for pattern analysis
        self.ged_history: List[float] = []
        self.ig_history: List[float] = []
        self.spike_history: List[bool] = []

    def detect_spike(self, delta_ged: float, delta_ig: float) -> Dict[str, Any]:
        """
        Detect EurekaSpike based on ΔGED and ΔIG values.

        EurekaSpike fires when:
        - ΔGED drops ≥ threshold (structural simplification)
        - ΔIG rises ≥ threshold (information gain)

        Args:
            delta_ged: Graph edit distance change
            delta_ig: Information gain change

        Returns:
            Dict containing spike detection results
        """
        # Record history
        self.ged_history.append(delta_ged)
        self.ig_history.append(delta_ig)

        # Core spike detection logic
        ged_drops = delta_ged <= -self.ged_threshold  # ΔGED drops significantly
        ig_rises = delta_ig >= self.ig_threshold  # ΔIG increases significantly

        eureka_spike = ged_drops and ig_rises
        self.spike_history.append(eureka_spike)

        # Calculate spike intensity
        spike_intensity = 0.0
        if eureka_spike:
            # Intensity based on how much thresholds are exceeded
            ged_excess = (
                abs(delta_ged) / self.ged_threshold if self.ged_threshold > 0 else 1.0
            )
            ig_excess = delta_ig / self.ig_threshold if self.ig_threshold > 0 else 1.0
            spike_intensity = min(1.0, (ged_excess + ig_excess) / 2.0)

        # Generate reward signal for memory updates
        reward = spike_intensity * self.eta_spike if eureka_spike else 0.0

        result = {
            "eureka_spike": eureka_spike,
            "spike_intensity": spike_intensity,
            "reward": reward,
            "metrics": {
                "delta_ged": delta_ged,
                "delta_ig": delta_ig,
                "ged_drops": ged_drops,
                "ig_rises": ig_rises,
            },
            "thresholds": {
                "ged_threshold": self.ged_threshold,
                "ig_threshold": self.ig_threshold,
            },
        }

        if eureka_spike:
            logger.info(
                f"🎆 EUREKA SPIKE detected! Intensity: {spike_intensity:.3f}, "
                f"ΔGED: {delta_ged:.3f}, ΔIG: {delta_ig:.3f}"
            )

        return result

    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze historical patterns for insights."""
        if len(self.ged_history) < 2:
            return {"status": "insufficient_data"}

        # Calculate recent trends
        recent_window = min(10, len(self.ged_history))
        recent_ged = self.ged_history[-recent_window:]
        recent_ig = self.ig_history[-recent_window:]
        recent_spikes = self.spike_history[-recent_window:]

        return {
            "total_spikes": sum(self.spike_history),
            "spike_rate": sum(recent_spikes) / len(recent_spikes),
            "avg_ged": sum(recent_ged) / len(recent_ged),
            "avg_ig": sum(recent_ig) / len(recent_ig),
            "ged_trend": "decreasing"
            if recent_ged[-1] < recent_ged[0]
            else "increasing",
            "ig_trend": "increasing" if recent_ig[-1] > recent_ig[0] else "decreasing",
            "history_length": len(self.ged_history),
        }


# Convenience function for simple spike detection
def detect_eureka_spike(delta_ged: float, delta_ig: float, **kwargs) -> bool:
    """
    Simple function to detect EurekaSpike.

    Args:
        delta_ged: Graph edit distance change
        delta_ig: Information gain change
        **kwargs: Optional threshold overrides

    Returns:
        True if EurekaSpike detected, False otherwise
    """
    detector = EurekaDetector(
        ged_threshold=kwargs.get("ged_threshold", SPIKE_GED),
        ig_threshold=kwargs.get("ig_threshold", SPIKE_IG),
    )
    result = detector.detect_spike(delta_ged, delta_ig)
    return result["eureka_spike"]
