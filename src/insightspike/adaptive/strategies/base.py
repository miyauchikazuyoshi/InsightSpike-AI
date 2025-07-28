"""
Base strategy implementation with common functionality
"""

from typing import List
import logging

from ..core.interfaces import ExplorationStrategy, ExplorationParams, ExplorationResult
from ..calculators.adaptive_topk import AdaptiveTopKCalculator

logger = logging.getLogger(__name__)


class BaseStrategy(ExplorationStrategy):
    """Base class with common strategy functionality"""
    
    def __init__(
        self,
        initial_radius: float = 0.7,
        initial_temperature: float = 1.0,
        min_radius: float = 0.1,
        max_radius: float = 1.0,
        temperature_decay: float = 0.95,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize base strategy parameters.
        
        Args:
            initial_radius: Starting exploration radius
            initial_temperature: Starting temperature for exploration
            min_radius: Minimum exploration radius
            max_radius: Maximum exploration radius  
            temperature_decay: Temperature decay factor per attempt
            confidence_threshold: Confidence to consider stopping
        """
        self.initial_radius = initial_radius
        self.initial_temperature = initial_temperature
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.temperature_decay = temperature_decay
        self.confidence_threshold = confidence_threshold
        
        # TopK calculator
        self.topk_calculator = AdaptiveTopKCalculator()
        
    def get_initial_params(self) -> ExplorationParams:
        """Get initial exploration parameters"""
        # Use default L1 analysis for initial topK
        default_l1 = {
            "known_elements": [],
            "unknown_elements": ["unknown"],
            "requires_synthesis": False,
            "query_complexity": 0.5,
            "analysis_confidence": 0.5
        }
        
        topk_values = self.topk_calculator.calculate(default_l1)
        
        return ExplorationParams(
            radius=self.initial_radius,
            topk_l1=topk_values["layer1_k"],
            topk_l2=topk_values["layer2_k"],
            topk_l3=topk_values["layer3_k"],
            temperature=self.initial_temperature,
            attempt_number=0
        )
    
    def should_continue(self, results: List[ExplorationResult]) -> bool:
        """Decide whether to continue exploration"""
        if not results:
            return True
            
        # Stop if spike detected
        if any(r.spike_detected for r in results):
            return False
            
        # Stop if high confidence achieved
        latest_result = results[-1]
        if latest_result.confidence >= self.confidence_threshold:
            logger.debug(
                f"Stopping due to high confidence: {latest_result.confidence:.2f}"
            )
            return False
            
        # Stop if confidence plateaued (last 3 attempts)
        if len(results) >= 3:
            recent_confidences = [r.confidence for r in results[-3:]]
            confidence_change = max(recent_confidences) - min(recent_confidences)
            if confidence_change < 0.05:
                logger.debug("Stopping due to confidence plateau")
                return False
                
        return True
    
    def _calculate_temperature(self, attempt: int) -> float:
        """Calculate temperature for given attempt"""
        return self.initial_temperature * (self.temperature_decay ** attempt)
    
    def _bound_radius(self, radius: float) -> float:
        """Ensure radius is within bounds"""
        return max(self.min_radius, min(self.max_radius, radius))