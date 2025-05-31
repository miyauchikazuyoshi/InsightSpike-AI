"""
Layer 1: Error Monitor - Cerebellum Analog
==========================================

Implements uncertainty calculation and error monitoring for the InsightSpike architecture.
"""

import numpy as np
from typing import Sequence, List, Any, Dict
from ..interfaces import L1ErrorMonitorInterface, LayerInput, LayerOutput
from ...config import get_config


class ErrorMonitor(L1ErrorMonitorInterface):
    """
    Layer 1 implementation for error monitoring and uncertainty calculation.
    
    This layer acts as the cerebellum analog, providing uncertainty measures
    and prediction error calculations to guide the reasoning process.
    """
    
    def __init__(self, layer_id: str = "L1_ErrorMonitor", config: Dict[str, Any] = None):
        super().__init__(layer_id, config)
        self.global_config = get_config()
        self.error_history = []
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.5)
    
    def initialize(self) -> bool:
        """Initialize the error monitor"""
        try:
            self.error_history = []
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing L1 ErrorMonitor: {e}")
            return False
    
    def process(self, input_data: LayerInput) -> LayerOutput:
        """Process input through error monitor"""
        if 'scores' in input_data.data:
            uncertainty_value = self.calculate_uncertainty(input_data.data['scores'])
        else:
            uncertainty_value = 0.0
        
        return LayerOutput(
            result={'uncertainty': uncertainty_value},
            confidence=1.0 - uncertainty_value,
            metadata={'layer_id': self.layer_id},
            metrics={'uncertainty': uncertainty_value}
        )
    
    def calculate_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate prediction error between predicted and actual values.
        
        Args:
            predicted: Predicted values
            actual: Actual values
            
        Returns:
            float: Calculated error value
        """
        if len(predicted) != len(actual):
            raise ValueError("Predicted and actual arrays must have same length")
        
        # Mean squared error
        error = float(np.mean((predicted - actual) ** 2))
        self.error_history.append(error)
        
        # Keep only recent history
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        return error
    
    def get_uncertainty(self, input_sequence: List[Any]) -> float:
        """
        Get uncertainty measure for input sequence.
        
        Args:
            input_sequence: Input sequence to evaluate
            
        Returns:
            float: Uncertainty value (0.0 to 1.0)
        """
        if not input_sequence:
            return 1.0
        
        # Simple entropy-based uncertainty
        if isinstance(input_sequence[0], (int, float)):
            return self.calculate_uncertainty(input_sequence)
        
        # For non-numeric sequences, use length-based heuristic
        return min(1.0, 1.0 / (len(input_sequence) + 1))
    
    def calculate_uncertainty(self, scores: Sequence[float]) -> float:
        """
        Calculate uncertainty using entropy of probability distribution.
        
        Args:
            scores: Sequence of scores to evaluate
            
        Returns:
            float: Uncertainty value
        """
        probs = np.array(scores, dtype=float)
        
        # Normalize to probabilities
        probs = probs / (probs.sum() + 1e-9)
        
        # Calculate entropy
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        
        # Normalize to [0, 1] range
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
        normalized_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return min(1.0, max(0.0, normalized_uncertainty))
    
    def get_error_trend(self) -> str:
        """Get trend of recent errors"""
        if len(self.error_history) < 2:
            return "insufficient_data"
        
        recent_errors = self.error_history[-5:]
        if len(recent_errors) < 2:
            return "stable"
        
        trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def is_error_spike(self) -> bool:
        """Check if there's an error spike"""
        if len(self.error_history) < 3:
            return False
        
        current_error = self.error_history[-1]
        recent_mean = np.mean(self.error_history[-10:-1]) if len(self.error_history) > 10 else np.mean(self.error_history[:-1])
        
        return current_error > recent_mean * 2.0
    
    def cleanup(self):
        """Cleanup resources"""
        self.error_history.clear()
        self._is_initialized = False


# Backward compatibility functions
def uncertainty(scores: Sequence[float]) -> float:
    """Legacy uncertainty function for backward compatibility"""
    monitor = ErrorMonitor()
    monitor.initialize()
    return monitor.calculate_uncertainty(scores)


# Export main symbols
__all__ = ['ErrorMonitor', 'uncertainty']
