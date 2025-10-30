"""
Expanding Strategy - Start narrow and broaden out
"""

import logging
from typing import List

from .base import BaseStrategy
from ..core.interfaces import ExplorationParams, ExplorationResult

logger = logging.getLogger(__name__)


class ExpandingStrategy(BaseStrategy):
    """
    Expanding exploration strategy.
    
    Starts with a narrow exploration radius and gradually expands
    to explore broader connections. Good for connecting distant concepts.
    """
    
    def __init__(
        self,
        initial_radius: float = 0.3,
        radius_growth_factor: float = 1.3,
        **kwargs
    ):
        """
        Initialize expanding strategy.
        
        Args:
            initial_radius: Starting exploration radius (narrower)
            radius_growth_factor: Factor to increase radius each attempt
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(initial_radius=initial_radius, **kwargs)
        self.radius_growth_factor = radius_growth_factor
        
    def adjust_params(
        self, 
        attempt: int, 
        prev_result: ExplorationResult
    ) -> ExplorationParams:
        """
        Adjust parameters by expanding the exploration radius.
        
        The radius increases each attempt to explore broader connections.
        TopK values are increased to capture more distant relationships.
        """
        # Calculate new radius (expanding)
        new_radius = self.initial_radius * (self.radius_growth_factor ** attempt)
        new_radius = self._bound_radius(new_radius)
        
        # Calculate new temperature
        new_temperature = self._calculate_temperature(attempt)
        
        # Recalculate topK based on previous metrics
        l1_metrics = prev_result.metrics
        l1_analysis = {
            "known_elements": ["known"] * int(l1_metrics.get("l1_known_ratio", 0.5) * 10),
            "unknown_elements": ["unknown"] * int((1 - l1_metrics.get("l1_known_ratio", 0.5)) * 10),
            "requires_synthesis": True,  # Assume synthesis for expansion
            "query_complexity": min(0.8, 0.5 + 0.1 * attempt),  # Increase complexity
            "analysis_confidence": 1.0 - l1_metrics.get("l1_uncertainty", 0.5)
        }
        
        topk_values = self.topk_calculator.calculate(l1_analysis)
        
        # Increase topK as we expand (capture more connections)
        topk_growth = 1.1 ** attempt
        
        params = ExplorationParams(
            radius=new_radius,
            topk_l1=int(topk_values["layer1_k"] * topk_growth),
            topk_l2=int(topk_values["layer2_k"] * topk_growth),
            topk_l3=int(topk_values["layer3_k"] * topk_growth),
            temperature=new_temperature,
            attempt_number=attempt
        )
        
        logger.debug(
            f"Expanding strategy: radius {self.initial_radius:.2f} → {new_radius:.2f}, "
            f"topK_l2 {topk_values['layer2_k']} → {params.topk_l2}"
        )
        
        return params