"""
Algorithms Module
================

Core algorithms for InsightSpike-AI's geDIG technology.
Provides implementations of Graph Edit Distance (GED) and Information Gain (IG) calculations.

This module serves as the foundation for:
- ΔGED computation: Measuring structural changes in knowledge graphs
- ΔIG computation: Measuring information gain during learning
- EurekaSpike detection: Combining ΔGED and ΔIG for insight detection

External Research API:
    The algorithms in this module are designed to be easily used by external researchers
    for replicating and extending InsightSpike-AI's core functionality.
"""

from .graph_edit_distance import (
    GraphEditDistance,
    OptimizationLevel,
    GEDResult,
    compute_graph_edit_distance,
    compute_delta_ged
)

from .information_gain import (
    InformationGain,
    EntropyMethod,
    IGResult,
    compute_shannon_entropy,
    compute_information_gain,
    compute_delta_ig
)

# Primary algorithm classes for external research use
__all__ = [
    # Graph Edit Distance components
    "GraphEditDistance",
    "OptimizationLevel", 
    "GEDResult",
    "compute_graph_edit_distance",
    "compute_delta_ged",
    
    # Information Gain components  
    "InformationGain",
    "EntropyMethod",
    "IGResult", 
    "compute_shannon_entropy",
    "compute_information_gain",
    "compute_delta_ig"
]

# Algorithm metadata for external researchers
ALGORITHM_INFO = {
    "graph_edit_distance": {
        "description": "Measures structural changes in knowledge graphs",
        "mathematical_form": "ΔGED = GED(G_after, G_ref) - GED(G_before, G_ref)",
        "insight_threshold": -0.5,
        "optimization_levels": ["fast", "standard", "precise"],
        "complexity": {
            "fast": "O(n²)",
            "standard": "O(n³) for small graphs",
            "precise": "O(n!) - use with caution"
        }
    },
    "information_gain": {
        "description": "Measures information gain during learning processes", 
        "mathematical_form": "ΔIG = H(S_after) - H(S_before)",
        "insight_threshold": 0.2,
        "entropy_methods": ["shannon", "clustering", "mutual_info", "feature_entropy"],
        "complexity": "O(n log n) for most methods"
    },
    "fusion_scheme": {
        "description": "Combined insight detection using ΔGED and ΔIG",
        "mathematical_form": "R(w₁,w₂,w₃) = w₁×ΔGED + w₂×ΔIG - w₃×ConflictScore",
        "default_weights": {"ged": 0.4, "ig": 0.3, "conflict": 0.3},
        "eureka_conditions": "ΔGED ≤ -0.5 AND ΔIG ≥ 0.2"
    }
}


def get_algorithm_info() -> dict:
    """
    Get comprehensive information about available algorithms.
    
    Returns:
        dict: Algorithm metadata and usage information
    """
    return ALGORITHM_INFO


def create_default_ged_calculator(**kwargs) -> GraphEditDistance:
    """
    Create a GED calculator with recommended default settings.
    
    Args:
        **kwargs: Override parameters
        
    Returns:
        GraphEditDistance: Configured calculator
    """
    defaults = {
        "optimization_level": OptimizationLevel.STANDARD,
        "node_cost": 1.0,
        "edge_cost": 1.0,
        "timeout_seconds": 5.0
    }
    defaults.update(kwargs)
    return GraphEditDistance(**defaults)


def create_default_ig_calculator(**kwargs) -> InformationGain:
    """
    Create an IG calculator with recommended default settings.
    
    Args:
        **kwargs: Override parameters
        
    Returns:
        InformationGain: Configured calculator
    """
    defaults = {
        "method": EntropyMethod.CLUSTERING,
        "k_clusters": 8,
        "min_samples": 2
    }
    defaults.update(kwargs)
    return InformationGain(**defaults)