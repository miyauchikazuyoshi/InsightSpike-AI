"""EurekaSpike detection logic - the core insight mechanism."""

from __future__ import annotations

import importlib.util
import logging

# Import from the legacy config.py file
import os
from typing import Any, Dict, List, Tuple

try:
    # Import from the modern config system directly
    from ..config import get_config

    config = get_config()
    SPIKE_GED = config.spike.spike_ged
    SPIKE_IG = config.spike.spike_ig
    ETA_SPIKE = config.spike.eta_spike
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
        self.llm = None  # Optional LLM for reasoning

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

    def detect(self, text: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect if the given text represents an insight spike.

        This is a simplified interface for DataStoreMainAgent integration.

        Args:
            text: Input text to analyze
            context: List of related episodes with similarity scores

        Returns:
            Dict with 'has_spike', 'confidence', and other metadata
        """
        # Simple heuristic based on context similarity
        if not context:
            # No context = potentially new insight
            return {
                "has_spike": True,
                "confidence": 0.8,
                "reason": "Novel input with no similar context",
            }

        # Check similarity of top result
        top_similarity = context[0].get("similarity", 0) if context else 0

        if top_similarity > 0.95:
            # Too similar = not an insight
            return {
                "has_spike": False,
                "confidence": 0.9,
                "reason": "Very similar to existing knowledge",
            }
        elif top_similarity > 0.7:
            # Moderately similar = potential insight
            return {
                "has_spike": True,
                "confidence": 0.6,
                "reason": "Related but distinct from existing knowledge",
            }
        else:
            # Low similarity = likely insight
            return {
                "has_spike": True,
                "confidence": 0.85,
                "reason": "Novel concept with low similarity to existing knowledge",
            }

    def set_llm(self, llm):
        """Set LLM for enhanced insight detection."""
        self.llm = llm

    async def detect_with_llm(
        self, text: str, context: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced insight detection using LLM reasoning.

        Args:
            text: Input text to analyze
            context: List of related episodes

        Returns:
            Dict with detailed spike analysis
        """
        # First use heuristic detection
        basic_result = self.detect(text, context)

        if not self.llm:
            return basic_result

        # Prepare context for LLM
        context_texts = []
        if context:
            for ep in context[:5]:  # Top 5 related episodes
                context_texts.append(
                    f"- {ep.get('text', '')} (similarity: {ep.get('similarity', 0):.2f})"
                )

        prompt = f"""Analyze if this represents a genuine insight spike:

Input: {text}

Related knowledge:
{chr(10).join(context_texts) if context_texts else 'No related knowledge found'}

Initial assessment:
- Has spike: {basic_result['has_spike']}
- Confidence: {basic_result['confidence']}
- Reason: {basic_result['reason']}

Provide enhanced analysis:
1. Is this a genuine insight that bridges concepts?
2. What is the novelty score (0-1)?
3. What concepts does it connect?
4. Confidence in spike detection (0-1)?

Respond in JSON format.
"""

        try:
            # Call LLM for enhanced analysis
            llm_response = await self.llm.agenerate(prompt)

            # Parse response
            import json

            analysis = json.loads(llm_response)

            return {
                "has_spike": analysis.get(
                    "is_genuine_insight", basic_result["has_spike"]
                ),
                "confidence": analysis.get("confidence", basic_result["confidence"]),
                "novelty_score": analysis.get("novelty_score", 0.5),
                "connected_concepts": analysis.get("connected_concepts", []),
                "reason": analysis.get("reasoning", basic_result["reason"]),
                "basic_result": basic_result,
                "llm_enhanced": True,
            }

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return basic_result


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
