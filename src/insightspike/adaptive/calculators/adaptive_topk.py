"""
Adaptive TopK Calculator
=======================

Calculates dynamic topK values based on Layer 1 analysis.
Ported and refactored from old implementation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..core.interfaces import TopKCalculator

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveTopKConfig:
    """Configuration for adaptive topK algorithm"""
    
    # Base topK values for different layers
    base_layer1_k: int = 20
    base_layer2_k: int = 15
    base_layer3_k: int = 12
    
    # Scaling factors based on analysis
    synthesis_multiplier: float = 1.5      # Increase for synthesis tasks
    complexity_multiplier: float = 1.3     # Increase for complex queries
    unknown_ratio_multiplier: float = 1.2  # Increase for high unknown ratio
    
    # Bounds to prevent excessive resource usage
    max_layer1_k: int = 50
    max_layer2_k: int = 30
    max_layer3_k: int = 25
    min_k: int = 3
    
    # Confidence-based adjustments
    low_confidence_multiplier: float = 1.4
    confidence_threshold: float = 0.6


class AdaptiveTopKCalculator(TopKCalculator):
    """
    Calculates adaptive topK values based on Layer 1 analysis.
    
    Factors considered:
    - Synthesis requirement (queries requiring combination of concepts)
    - Query complexity (0-1 scale)
    - Unknown ratio (proportion of unknown elements)
    - Analysis confidence
    """
    
    def __init__(self, config: Optional[AdaptiveTopKConfig] = None):
        """Initialize with optional configuration"""
        self.config = config or AdaptiveTopKConfig()
        
    def calculate(self, l1_analysis: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate adaptive topK values for each layer.
        
        Args:
            l1_analysis: Layer 1 analysis results containing:
                - known_elements: List of known concepts
                - unknown_elements: List of unknown concepts
                - requires_synthesis: Boolean for synthesis requirement
                - query_complexity: Float 0-1 indicating complexity
                - analysis_confidence: Float 0-1 indicating confidence
                
        Returns:
            Dict with keys: layer1_k, layer2_k, layer3_k
        """
        # Extract analysis components
        known_count = len(l1_analysis.get("known_elements", []))
        unknown_count = len(l1_analysis.get("unknown_elements", []))
        requires_synthesis = l1_analysis.get("requires_synthesis", False)
        query_complexity = l1_analysis.get("query_complexity", 0.5)
        analysis_confidence = l1_analysis.get("analysis_confidence", 0.5)
        
        # Calculate unknown ratio
        total_concepts = known_count + unknown_count
        unknown_ratio = unknown_count / total_concepts if total_concepts > 0 else 0.5
        
        # Start with base values
        layer1_k = self.config.base_layer1_k
        layer2_k = self.config.base_layer2_k
        layer3_k = self.config.base_layer3_k
        
        # Apply synthesis multiplier
        if requires_synthesis:
            layer1_k = int(layer1_k * self.config.synthesis_multiplier)
            layer2_k = int(layer2_k * self.config.synthesis_multiplier)
            layer3_k = int(layer3_k * self.config.synthesis_multiplier)
            
            logger.debug(f"Applied synthesis multiplier: {self.config.synthesis_multiplier}")
        
        # Apply complexity multiplier
        complexity_factor = 1.0 + (query_complexity * (self.config.complexity_multiplier - 1.0))
        layer1_k = int(layer1_k * complexity_factor)
        layer2_k = int(layer2_k * complexity_factor)
        layer3_k = int(layer3_k * complexity_factor)
        
        # Apply unknown ratio multiplier
        unknown_factor = 1.0 + (unknown_ratio * (self.config.unknown_ratio_multiplier - 1.0))
        layer1_k = int(layer1_k * unknown_factor)
        layer2_k = int(layer2_k * unknown_factor)
        layer3_k = int(layer3_k * unknown_factor)
        
        # Apply confidence adjustment
        confidence_factor = 1.0
        if analysis_confidence < self.config.confidence_threshold:
            confidence_factor = self.config.low_confidence_multiplier
            layer1_k = int(layer1_k * confidence_factor)
            layer2_k = int(layer2_k * confidence_factor)
            layer3_k = int(layer3_k * confidence_factor)
            
            logger.debug(
                f"Low confidence ({analysis_confidence:.2f}), "
                f"applied multiplier: {confidence_factor}"
            )
        
        # Apply bounds
        layer1_k = max(self.config.min_k, min(layer1_k, self.config.max_layer1_k))
        layer2_k = max(self.config.min_k, min(layer2_k, self.config.max_layer2_k))
        layer3_k = max(self.config.min_k, min(layer3_k, self.config.max_layer3_k))
        
        result = {
            "layer1_k": layer1_k,
            "layer2_k": layer2_k,
            "layer3_k": layer3_k
        }
        
        logger.debug(
            f"Adaptive topK calculated: L1={layer1_k}, L2={layer2_k}, L3={layer3_k} "
            f"(synthesis={requires_synthesis}, complexity={query_complexity:.2f}, "
            f"unknown_ratio={unknown_ratio:.2f})"
        )
        
        return result


def estimate_chain_reaction_potential(
    l1_analysis: Dict[str, Any],
    adaptive_topk: Dict[str, int]
) -> float:
    """
    Estimate the potential for chain reaction insight improvement.
    
    Higher potential indicates likelihood of cascading insights
    when exploring with current parameters.
    
    Args:
        l1_analysis: Layer 1 analysis results
        adaptive_topk: Calculated topK values
        
    Returns:
        Float 0-1 indicating chain reaction potential
    """
    # Factors that increase chain reaction potential
    synthesis_required = l1_analysis.get("requires_synthesis", False)
    query_complexity = l1_analysis.get("query_complexity", 0.5)
    unknown_count = len(l1_analysis.get("unknown_elements", []))
    
    # TopK density factor (higher topK = more potential connections)
    avg_topk = (
        adaptive_topk["layer1_k"] + 
        adaptive_topk["layer2_k"] + 
        adaptive_topk["layer3_k"]
    ) / 3
    density_factor = min(1.0, avg_topk / 20.0)
    
    # Synthesis factor (synthesis queries have higher potential)
    synthesis_factor = 0.8 if synthesis_required else 0.3
    
    # Complexity factor (complex queries = more insight potential)
    complexity_factor = query_complexity
    
    # Unknown element factor (more unknowns = higher discovery potential)
    unknown_factor = min(1.0, unknown_count / 10.0)
    
    # Weighted combination
    chain_reaction_potential = (
        synthesis_factor * 0.4 +
        complexity_factor * 0.3 +
        density_factor * 0.2 +
        unknown_factor * 0.1
    )
    
    return min(1.0, chain_reaction_potential)