"""
Alternating Strategy - Oscillate between broad and narrow
"""

import logging
from typing import List

from .base import BaseStrategy
from ..core.interfaces import ExplorationParams, ExplorationResult

logger = logging.getLogger(__name__)


class AlternatingStrategy(BaseStrategy):
    """
    Alternating exploration strategy.
    
    Oscillates between broad and narrow exploration to get benefits
    of both approaches. Good for comprehensive exploration.
    """
    
    def __init__(
        self,
        broad_radius: float = 0.8,
        narrow_radius: float = 0.3,
        **kwargs
    ):
        """
        Initialize alternating strategy.
        
        Args:
            broad_radius: Radius for broad exploration phases
            narrow_radius: Radius for narrow exploration phases
            **kwargs: Additional parameters for base strategy
        """
        # Start with broad radius
        super().__init__(initial_radius=broad_radius, **kwargs)
        self.broad_radius = broad_radius
        self.narrow_radius = narrow_radius
        
    def adjust_params(
        self, 
        attempt: int, 
        prev_result: ExplorationResult
    ) -> ExplorationParams:
        """
        Adjust parameters by alternating between broad and narrow.
        
        Odd attempts use narrow radius, even attempts use broad radius.
        This creates an oscillating pattern that explores different scales.
        """
        # Alternate between broad and narrow
        if attempt % 2 == 0:
            # Even attempt: broad exploration
            new_radius = self.broad_radius
            topk_multiplier = 1.2  # More results for broad search
            logger.debug(f"Alternating to BROAD exploration (attempt {attempt})")
        else:
            # Odd attempt: narrow exploration  
            new_radius = self.narrow_radius
            topk_multiplier = 0.8  # Fewer results for focused search
            logger.debug(f"Alternating to NARROW exploration (attempt {attempt})")
        
        # Slight decay over time
        decay_factor = 0.95 ** (attempt // 2)
        new_radius = self._bound_radius(new_radius * decay_factor)
        
        # Calculate new temperature
        new_temperature = self._calculate_temperature(attempt)
        
        # Recalculate topK based on previous metrics
        l1_metrics = prev_result.metrics
        l1_analysis = {
            "known_elements": ["known"] * int(l1_metrics.get("l1_known_ratio", 0.5) * 10),
            "unknown_elements": ["unknown"] * int((1 - l1_metrics.get("l1_known_ratio", 0.5)) * 10),
            "requires_synthesis": attempt > 2,  # Assume synthesis after initial attempts
            "query_complexity": 0.6,  # Medium complexity
            "analysis_confidence": 1.0 - l1_metrics.get("l1_uncertainty", 0.5)
        }
        
        topk_values = self.topk_calculator.calculate(l1_analysis)
        
        params = ExplorationParams(
            radius=new_radius,
            topk_l1=int(topk_values["layer1_k"] * topk_multiplier),
            topk_l2=int(topk_values["layer2_k"] * topk_multiplier),
            topk_l3=int(topk_values["layer3_k"] * topk_multiplier),
            temperature=new_temperature,
            attempt_number=attempt
        )
        
        logger.debug(
            f"Alternating strategy: radius={new_radius:.2f}, "
            f"topK_l2={params.topk_l2}, temperature={new_temperature:.2f}"
        )
        
        return params