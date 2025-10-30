"""
Narrowing Strategy - Start broad and focus down
"""

import logging
from typing import List

from .base import BaseStrategy
from ..core.interfaces import ExplorationParams, ExplorationResult

logger = logging.getLogger(__name__)


class NarrowingStrategy(BaseStrategy):
    """
    Narrowing exploration strategy.
    
    Starts with a broad exploration radius and gradually narrows down
    to focus on specific areas. Good for finding specific insights.
    """
    
    def __init__(
        self,
        initial_radius: float = 0.8,
        radius_decay_factor: float = 0.8,
        **kwargs
    ):
        """
        Initialize narrowing strategy.
        
        Args:
            initial_radius: Starting exploration radius (broader)
            radius_decay_factor: Factor to reduce radius each attempt
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(initial_radius=initial_radius, **kwargs)
        self.radius_decay_factor = radius_decay_factor
        
    def adjust_params(
        self, 
        attempt: int, 
        prev_result: ExplorationResult
    ) -> ExplorationParams:
        """
        Adjust parameters by narrowing the exploration radius.
        
        The radius decreases each attempt to focus the search.
        TopK values are recalculated based on the previous L1 analysis.
        """
        # Calculate new radius (narrowing)
        new_radius = self.initial_radius * (self.radius_decay_factor ** attempt)
        new_radius = self._bound_radius(new_radius)
        
        # Calculate new temperature
        new_temperature = self._calculate_temperature(attempt)
        
        # Recalculate topK based on previous metrics
        l1_metrics = prev_result.metrics
        l1_analysis = {
            "known_elements": ["known"] * int(l1_metrics.get("l1_known_ratio", 0.5) * 10),
            "unknown_elements": ["unknown"] * int((1 - l1_metrics.get("l1_known_ratio", 0.5)) * 10),
            "requires_synthesis": l1_metrics.get("l1_uncertainty", 0.5) > 0.6,
            "query_complexity": 1.0 - prev_result.confidence,
            "analysis_confidence": 1.0 - l1_metrics.get("l1_uncertainty", 0.5)
        }
        
        topk_values = self.topk_calculator.calculate(l1_analysis)
        
        # Reduce topK slightly as we narrow (focus on quality over quantity)
        topk_reduction = 0.9 ** attempt
        
        params = ExplorationParams(
            radius=new_radius,
            topk_l1=int(topk_values["layer1_k"] * topk_reduction),
            topk_l2=int(topk_values["layer2_k"] * topk_reduction),
            topk_l3=int(topk_values["layer3_k"] * topk_reduction),
            temperature=new_temperature,
            attempt_number=attempt
        )
        
        logger.debug(
            f"Narrowing strategy: radius {self.initial_radius:.2f} → {new_radius:.2f}, "
            f"topK_l2 {topk_values['layer2_k']} → {params.topk_l2}"
        )
        
        return params