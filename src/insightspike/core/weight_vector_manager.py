"""
Weight Vector Manager
=====================

Simple vector weight management using element-wise multiplication.
"""

import numpy as np
import logging
from typing import Optional, List
from ..config.vector_weights import VectorWeightConfig, DEFAULT_PRESETS

logger = logging.getLogger(__name__)


class WeightVectorManager:
    """Simple vector weight manager.
    
    Applies element-wise multiplication to vectors based on configured weights.
    """
    
    def __init__(self, config: Optional[VectorWeightConfig] = None):
        """Initialize weight vector manager.
        
        Args:
            config: Vector weight configuration. If None, creates default (disabled).
        """
        self.config = config or VectorWeightConfig()
        self._weight_vector = None
        
        # Add default presets if not present
        for name, preset in DEFAULT_PRESETS.items():
            if name not in self.config.presets:
                self.config.presets[name] = preset
        
        self._initialize_weights()
        
        if self.config.enabled:
            logger.info(f"WeightVectorManager initialized (enabled={self.config.enabled})")
            if self._weight_vector:
                logger.debug(f"Weight vector: {self._weight_vector[:10]}..." 
                           if len(self._weight_vector) > 10 else f"Weight vector: {self._weight_vector}")
    
    def _initialize_weights(self):
        """Initialize the weight vector from config."""
        if not self.config.enabled:
            return
        
        # Use preset if specified
        if self.config.active_preset and self.config.active_preset in self.config.presets:
            self._weight_vector = self.config.presets[self.config.active_preset]
            logger.info(f"Using preset: {self.config.active_preset}")
        # Use direct weights
        elif self.config.weights:
            self._weight_vector = self.config.weights
            logger.info(f"Using direct weights (dim={len(self.config.weights)})")
        else:
            self._weight_vector = None
            logger.debug("No weights configured")
    
    def apply_weights(self, vector: np.ndarray) -> np.ndarray:
        """Apply weights to a vector using element-wise multiplication.
        
        Args:
            vector: Input vector
            
        Returns:
            Weighted vector (or original if weights not applicable)
        """
        # Feature disabled or no weights
        if not self.config.enabled or self._weight_vector is None:
            return vector
        
        # Dimension check
        if len(vector) != len(self._weight_vector):
            logger.warning(
                f"Dimension mismatch: vector has {len(vector)} dims, "
                f"weight has {len(self._weight_vector)} dims. Skipping weights."
            )
            return vector
        
        # Element-wise multiplication
        weighted = vector * np.array(self._weight_vector, dtype=vector.dtype)
        return weighted
    
    def apply_to_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Apply weights to a batch of vectors.
        
        Args:
            vectors: Batch of vectors (N x D) or single vector (D,)
            
        Returns:
            Weighted vectors
        """
        if len(vectors.shape) == 1:
            # Single vector
            return self.apply_weights(vectors)
        
        # Batch processing
        return np.array([self.apply_weights(v) for v in vectors])
    
    def switch_preset(self, preset_name: str):
        """Switch to a different preset.
        
        Args:
            preset_name: Name of preset to activate
            
        Raises:
            ValueError: If preset doesn't exist
        """
        if preset_name not in self.config.presets:
            available = list(self.config.presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        self.config.active_preset = preset_name
        self._initialize_weights()
        logger.info(f"Switched to preset: {preset_name}")
    
    def set_weights(self, weights: List[float]):
        """Set weight vector directly.
        
        Args:
            weights: New weight vector
        """
        self._weight_vector = weights
        self.config.weights = weights
        self.config.active_preset = None  # Clear preset when setting direct weights
        logger.info(f"Set weights directly (dim={len(weights)})")
    
    def get_weights(self) -> Optional[List[float]]:
        """Get current weight vector.
        
        Returns:
            Current weight vector or None if not configured
        """
        return self._weight_vector
    
    def is_enabled(self) -> bool:
        """Check if weight application is enabled.
        
        Returns:
            True if enabled and weights are configured
        """
        return self.config.enabled and self._weight_vector is not None