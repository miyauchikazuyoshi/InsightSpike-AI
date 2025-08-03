"""
Wake-Sleep cycle configuration models.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class SphereSearchConfig(BaseModel):
    """Configuration for sphere/donut search in wake mode.
    
    IMPORTANT: For high-dimensional spaces (e.g., 768-dim embeddings),
    use intuitive_radius instead of radius for 3D-like behavior.
    """
    
    # Search method selection
    method: Literal["cosine", "sphere", "donut"] = Field(
        default="sphere",
        description="Search method: cosine (legacy), sphere, or donut"
    )
    
    # Direct radius (use only if you know what you're doing)
    radius: float = Field(
        default=0.995,  # Equivalent to intuitive 0.5 in 768D
        ge=0.1,
        le=2.0,
        description="Direct radius. For 768D: 0.98-0.999 typical. Use intuitive_radius instead!"
    )
    
    # Intuitive radius (RECOMMENDED) - preserves 3D volume intuition
    intuitive_radius: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="""
        3D-intuitive radius that auto-scales to any dimension.
        Recommended ranges by task:
        - 0.2-0.3: Very strict (medical, legal) - finds near-duplicates
        - 0.3-0.4: Strict (academic) - closely related concepts only  
        - 0.4-0.6: Balanced (general QA) - standard semantic search
        - 0.6-0.8: Creative (brainstorming) - includes tangential concepts
        
        Examples:
        - 0.25 → 1.6% volume (very selective)
        - 0.50 → 12.5% volume (1/8, balanced)
        - 0.75 → 42.2% volume (inclusive)
        """
    )
    
    # Donut search parameters
    inner_radius: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Direct inner radius for donut search. Use intuitive_inner_radius instead!"
    )
    
    intuitive_inner_radius: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="""
        Inner radius for filtering known/redundant information.
        Recommended: 0.1-0.25
        - 0.1: Minimal filtering (keeps most similar)
        - 0.2: Standard filtering (removes obvious duplicates)
        - 0.25: Aggressive filtering (only novel info)
        """
    )
    
    intuitive_outer_radius: float = Field(
        default=0.6,
        ge=0.2,
        le=1.0,
        description="""
        Outer radius for relevance boundary.
        Should be larger than inner_radius.
        Recommended: 0.4-0.7
        - 0.4: Very focused (high precision)
        - 0.5-0.6: Balanced (good precision/recall)
        - 0.7: Broad (high recall)
        """
    )
    
    # Search constraints
    max_neighbors: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of neighbors to retrieve"
    )
    
    # Backend selection
    use_faiss: bool = Field(
        default=True,
        description="Use FAISS for large-scale search (falls back to numpy if unavailable)"
    )
    
    # Dimension-aware search
    use_dimension_aware_scaling: bool = Field(
        default=True,
        description="Automatically scale intuitive radius based on vector dimension"
    )

    class Config:
        schema_extra = {
            "example": {
                "method": "donut",
                "intuitive_radius": 0.5,  # Use intuitive values!
                "intuitive_inner_radius": 0.2,
                "intuitive_outer_radius": 0.6,
                "max_neighbors": 20,
                "use_faiss": True,
                "use_dimension_aware_scaling": True
            },
            "example_strict": {
                # For medical/legal applications
                "method": "sphere",
                "intuitive_radius": 0.3,
                "max_neighbors": 10
            },
            "example_creative": {
                # For brainstorming/exploration
                "method": "donut",
                "intuitive_inner_radius": 0.1,
                "intuitive_outer_radius": 0.7,
                "max_neighbors": 50
            }
        }


class WakeModeConfig(BaseModel):
    """Configuration for wake mode (query processing)."""
    
    # Optimization objective
    objective: Literal["minimize_cost", "maximize_reward"] = Field(
        default="minimize_cost",
        description="Wake mode should minimize cost for efficiency"
    )
    
    # Search configuration
    search: SphereSearchConfig = Field(
        default_factory=SphereSearchConfig,
        description="Search configuration for finding relevant episodes"
    )
    
    # Cost parameters
    node_cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of activating a node"
    )
    
    edge_cost: float = Field(
        default=0.5,
        ge=0.0,
        description="Cost of traversing an edge"
    )
    
    # Processing parameters
    combination_search_method: Literal["exhaustive", "beam", "greedy"] = Field(
        default="beam",
        description="Method for searching optimal node combinations"
    )
    
    beam_width: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Beam width for beam search (if used)"
    )


class SleepModeConfig(BaseModel):
    """Configuration for sleep mode (consolidation)."""
    
    # Optimization objective
    objective: Literal["maximize_reward", "minimize_cost"] = Field(
        default="maximize_reward",
        description="Sleep mode should maximize reward for quality"
    )
    
    # Consolidation parameters
    contradiction_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for detecting contradictions"
    )
    
    merge_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for merging episodes"
    )
    
    prune_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Importance threshold below which to prune"
    )
    
    # Reward parameters (for geDIG calculation)
    lambda_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for information gain (ΔIG)"
    )
    
    mu_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for graph edit distance (ΔGED)"
    )


class WakeSleepConfig(BaseModel):
    """Complete wake-sleep cycle configuration."""
    
    # Mode selection
    mode: Literal["wake", "sleep", "auto"] = Field(
        default="wake",
        description="Current operating mode or auto-switching"
    )
    
    # Mode-specific configurations
    wake: WakeModeConfig = Field(
        default_factory=WakeModeConfig,
        description="Wake mode configuration"
    )
    
    sleep: SleepModeConfig = Field(
        default_factory=SleepModeConfig,
        description="Sleep mode configuration"
    )
    
    # Auto-switching parameters (when mode="auto")
    wake_duration: int = Field(
        default=100,
        ge=1,
        description="Number of queries before switching to sleep"
    )
    
    sleep_duration: int = Field(
        default=20,
        ge=1,
        description="Number of consolidation cycles in sleep"
    )
    
    # Transition triggers
    switch_on_low_performance: bool = Field(
        default=True,
        description="Switch to sleep if performance drops"
    )
    
    performance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Performance threshold for mode switching"
    )


# Radius recommendations based on vector space properties
RADIUS_RECOMMENDATIONS = {
    "intuitive_3d_based": {
        # RECOMMENDED: Use these with intuitive_radius parameters
        # These values work the same way regardless of dimension (768D, 384D, etc.)
        "sphere": {
            "very_strict": 0.2,      # ~0.8% volume - Near duplicates only
            "strict": 0.3,           # ~2.7% volume - Very similar concepts
            "balanced": 0.5,         # ~12.5% volume - Good precision/recall
            "inclusive": 0.7,        # ~34% volume - Broader search
            "exploratory": 0.8       # ~51% volume - Very inclusive
        },
        "donut": {
            "inner": {
                "minimal": 0.1,      # Filter almost nothing
                "standard": 0.2,     # Remove obvious duplicates  
                "aggressive": 0.25   # Remove all very similar
            },
            "outer": {
                "focused": 0.4,      # High precision search
                "balanced": 0.6,     # Good balance
                "broad": 0.7         # High recall search
            }
        },
        "task_specific": {
            "medical_legal": {"inner": 0.1, "outer": 0.35},      # Very precise
            "academic": {"inner": 0.15, "outer": 0.45},          # Precise
            "general_qa": {"inner": 0.2, "outer": 0.6},         # Balanced
            "creative": {"inner": 0.1, "outer": 0.7},           # Exploratory
            "brainstorming": {"inner": 0.05, "outer": 0.8}      # Very broad
        }
    },
    "direct_radius_768d": {
        # Only use these if you're setting radius directly (not recommended)
        # These are the actual radius values for 768-dimensional space
        "sphere": {
            "tight": 0.984,      # Equivalent to intuitive 0.25
            "normal": 0.9954,    # Equivalent to intuitive 0.5
            "broad": 0.9985      # Equivalent to intuitive 0.75
        },
        "note": "Use intuitive_radius instead! These values are dimension-specific."
    },
    "volume_fraction_guide": {
        # Understanding volume fractions
        "intuitive_0.2": "0.8% of space",
        "intuitive_0.3": "2.7% of space",
        "intuitive_0.4": "6.4% of space",
        "intuitive_0.5": "12.5% of space (1/8)",
        "intuitive_0.6": "21.6% of space",
        "intuitive_0.7": "34.3% of space",
        "intuitive_0.8": "51.2% of space"
    }
}