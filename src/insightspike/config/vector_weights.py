"""
Vector Weight Configuration
============================

Simple vector weight configuration for dimension-specific scaling.
Allows element-wise multiplication of vectors for task optimization.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict


class VectorWeightConfig(BaseModel):
    """Simple vector weight configuration.
    
    Allows direct specification of weights for each dimension,
    or selection from predefined presets.
    """
    
    enabled: bool = Field(
        False,
        description="Enable/disable vector weighting feature"
    )
    
    weights: Optional[List[float]] = Field(
        None,
        description="Weight vector (one weight per dimension)"
    )
    
    presets: Dict[str, Optional[List[float]]] = Field(
        default_factory=dict,
        description="Named weight presets"
    )
    
    active_preset: Optional[str] = Field(
        None,
        description="Name of preset to use (if None, use weights directly)"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "enabled": True,
            "weights": [1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3],
            "presets": {
                "maze_8d": [1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3],
                "maze_aggressive": [2.0, 2.0, 0.05, 0.05, 0.3, 0.3, 0.3, 0.1]
            }
        }
    })


# Default presets for common use cases
DEFAULT_PRESETS = {
    "maze_8d": [1.0, 1.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.3],
    "maze_aggressive": [2.0, 2.0, 0.05, 0.05, 0.3, 0.3, 0.3, 0.1],
    "language_384d": None,  # No weighting for language tasks
}