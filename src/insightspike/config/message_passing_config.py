"""
Message Passing Configuration Models
===================================

Pydantic models for message passing configuration with validation.
"""

from pydantic import BaseModel, Field
from typing import Literal


class MessagePassingConfig(BaseModel):
    """Configuration for question-aware message passing"""
    
    # Core parameters
    alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for question influence in node updates"
    )
    
    iterations: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of message passing iterations (reduced default from 3 to 2)"
    )
    
    max_hops: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Maximum hops from query-relevant nodes (limits computation)"
    )
    
    # Aggregation settings
    aggregation: Literal["weighted_mean", "max", "attention"] = Field(
        default="weighted_mean",
        description="Message aggregation method"
    )
    
    self_loop_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for self-loop in propagation"
    )
    
    # Propagation control
    decay_factor: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Decay factor for question relevance over distance"
    )
    
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity to maintain edges during propagation"
    )
    
    # Optimization settings
    convergence_threshold: float = Field(
        default=1e-4,
        gt=0.0,
        le=0.01,
        description="Threshold for early stopping when converged"
    )
    
    enable_batch_computation: bool = Field(
        default=True,
        description="Enable batch similarity computation for better performance"
    )
    
    cache_similarities: bool = Field(
        default=True,
        description="Cache similarity computations within a forward pass"
    )
    
    # Sparse graph control
    top_k_relevance_percentile: int = Field(
        default=75,
        ge=50,
        le=95,
        description="Percentile threshold for selecting starting nodes (e.g., 75 = top 25%)"
    )
    
    class Config:
        """Pydantic config"""
        schema_extra = {
            "example": {
                "alpha": 0.3,
                "iterations": 2,
                "max_hops": 1,
                "aggregation": "weighted_mean",
                "self_loop_weight": 0.5,
                "decay_factor": 0.8,
                "similarity_threshold": 0.3,
                "convergence_threshold": 0.0001,
                "enable_batch_computation": True,
                "cache_similarities": True,
                "top_k_relevance_percentile": 75
            }
        }


def get_default_message_passing_config() -> dict:
    """Get default message passing configuration as dict"""
    return MessagePassingConfig().dict()


def get_performance_optimized_config() -> dict:
    """Get performance-optimized configuration"""
    return MessagePassingConfig(
        iterations=1,
        max_hops=1,
        similarity_threshold=0.5,
        top_k_relevance_percentile=85
    ).dict()


def get_quality_optimized_config() -> dict:
    """Get quality-optimized configuration (slower but more thorough)"""
    return MessagePassingConfig(
        alpha=0.5,
        iterations=3,
        max_hops=3,
        similarity_threshold=0.2,
        top_k_relevance_percentile=60
    ).dict()