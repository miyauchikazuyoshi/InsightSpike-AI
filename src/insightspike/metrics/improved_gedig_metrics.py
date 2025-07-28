"""
Improved GEDIG Metrics
=====================

Separates GED (distance) from structural improvement for clearer semantics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GEDIGMetrics:
    """Complete GEDIG metrics with clear semantics."""
    
    # Raw metrics (always positive)
    ged: float  # Graph Edit Distance
    ig: float   # Information Gain
    
    # Derived metrics (can be negative)
    structural_improvement: float  # Positive = better structure
    knowledge_coherence: float     # 0-1, higher = more coherent
    
    # Composite scores
    insight_score: float          # Combined insight strength
    spike_detected: bool          # Binary spike detection
    spike_intensity: float        # 0-1, spike strength if detected
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for compatibility."""
        return {
            "ged": self.ged,
            "ig": self.ig,
            "structural_improvement": self.structural_improvement,
            "knowledge_coherence": self.knowledge_coherence,
            "insight_score": self.insight_score,
            "spike_detected": self.spike_detected,
            "spike_intensity": self.spike_intensity,
        }


class ImprovedGEDIGCalculator:
    """
    Calculates GEDIG metrics with proper separation of concerns.
    
    Key improvements:
    1. GED remains a distance (always positive)
    2. Structural improvement is a separate metric
    3. Clear composite scoring
    """
    
    def __init__(
        self,
        structure_weight: float = 0.5,
        knowledge_weight: float = 0.5,
        spike_threshold: float = 0.6,
    ):
        self.structure_weight = structure_weight
        self.knowledge_weight = knowledge_weight
        self.spike_threshold = spike_threshold
    
    def calculate(
        self,
        graph_before: Any,
        graph_after: Any,
        vectors_before: Any = None,
        vectors_after: Any = None,
    ) -> GEDIGMetrics:
        """
        Calculate comprehensive GEDIG metrics.
        
        Args:
            graph_before: Previous graph state
            graph_after: Current graph state
            vectors_before: Previous embedding vectors
            vectors_after: Current embedding vectors
            
        Returns:
            GEDIGMetrics with all calculated values
        """
        # Import dependencies
        from ..algorithms.graph_structure_analyzer import analyze_graph_structure
        from ..algorithms.information_gain import InformationGain
        
        # 1. Calculate structural metrics
        structure_analysis = analyze_graph_structure(graph_before, graph_after)
        
        ged = structure_analysis["ged"]
        structural_improvement = structure_analysis["structural_improvement"]
        
        # 2. Calculate information metrics
        ig_calculator = InformationGain()
        ig_result = ig_calculator.calculate(vectors_before, vectors_after)
        ig = ig_result.ig_value if hasattr(ig_result, 'ig_value') else 0.0
        
        # 3. Calculate knowledge coherence (0-1)
        coherence = self._calculate_coherence(
            structure_analysis,
            ig,
            vectors_after
        )
        
        # 4. Calculate composite insight score
        # Normalize components to [0, 1] range
        norm_structure = self._normalize_improvement(structural_improvement)
        norm_knowledge = self._normalize_ig(ig)
        
        insight_score = (
            self.structure_weight * norm_structure +
            self.knowledge_weight * norm_knowledge
        )
        
        # 5. Spike detection
        spike_detected = insight_score >= self.spike_threshold
        spike_intensity = min(1.0, insight_score / self.spike_threshold) if spike_detected else 0.0
        
        return GEDIGMetrics(
            ged=ged,
            ig=ig,
            structural_improvement=structural_improvement,
            knowledge_coherence=coherence,
            insight_score=insight_score,
            spike_detected=spike_detected,
            spike_intensity=spike_intensity,
        )
    
    def _calculate_coherence(
        self,
        structure_analysis: Dict,
        ig: float,
        vectors: Any
    ) -> float:
        """Calculate knowledge coherence score (0-1)."""
        # Factors for coherence
        scores = []
        
        # 1. Efficiency improvement contributes to coherence
        if "efficiency_change" in structure_analysis:
            eff_score = min(1.0, max(0.0, structure_analysis["efficiency_change"] + 0.5))
            scores.append(eff_score)
        
        # 2. Hub formation indicates organized knowledge
        if "hub_formation" in structure_analysis:
            hub_score = min(1.0, max(0.0, structure_analysis["hub_formation"]))
            scores.append(hub_score)
        
        # 3. Information gain without structure degradation
        if ig > 0 and structure_analysis.get("structural_improvement", 0) >= 0:
            scores.append(0.8)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _normalize_improvement(self, improvement: float) -> float:
        """Normalize structural improvement to [0, 1]."""
        # Map [-1, +1] to [0, 1]
        return (improvement + 1.0) / 2.0
    
    def _normalize_ig(self, ig: float) -> float:
        """Normalize information gain to [0, 1]."""
        # Typical IG range is [0, 3]
        return min(1.0, ig / 3.0)


def calculate_gedig_metrics(
    graph_before: Any,
    graph_after: Any,
    vectors_before: Any = None,
    vectors_after: Any = None,
    config: Optional[Dict] = None,
) -> GEDIGMetrics:
    """
    Convenience function for GEDIG calculation.
    
    Example:
        >>> metrics = calculate_gedig_metrics(g1, g2, v1, v2)
        >>> print(f"Spike detected: {metrics.spike_detected}")
        >>> print(f"Insight score: {metrics.insight_score:.3f}")
    """
    config = config or {}
    calculator = ImprovedGEDIGCalculator(
        structure_weight=config.get("structure_weight", 0.5),
        knowledge_weight=config.get("knowledge_weight", 0.5),
        spike_threshold=config.get("spike_threshold", 0.6),
    )
    return calculator.calculate(graph_before, graph_after, vectors_before, vectors_after)


# Backward compatibility wrapper
def compute_gedig_legacy(
    delta_ged: float,
    delta_ig: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Legacy compatibility wrapper that works with negative GED.
    
    Converts old-style metrics to new format internally.
    """
    weights = weights or {"ged": 0.5, "ig": 0.5}
    
    # Convert negative GED to structural improvement
    structural_improvement = -delta_ged if delta_ged < 0 else 0.0
    
    # Create synthetic metrics
    metrics = GEDIGMetrics(
        ged=abs(delta_ged),  # Make positive
        ig=delta_ig,
        structural_improvement=structural_improvement,
        knowledge_coherence=0.5,  # Default
        insight_score=weights["ged"] * structural_improvement + weights["ig"] * delta_ig,
        spike_detected=False,  # Will be set below
        spike_intensity=0.0,
    )
    
    # Legacy spike detection
    metrics.spike_detected = (
        delta_ged <= -0.5 and delta_ig >= 0.2
    )
    
    if metrics.spike_detected:
        ged_contrib = abs(delta_ged) / 0.5
        ig_contrib = delta_ig / 0.2
        metrics.spike_intensity = min(1.0, (ged_contrib + ig_contrib) / 2.0)
    
    return metrics.to_dict()