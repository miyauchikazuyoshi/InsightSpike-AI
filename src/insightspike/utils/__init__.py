"""Utility modules for InsightSpike-AI"""

# Export commonly used utilities
from .text_utils import clean_text, iter_text

# Export new visualization utilities - moved to visualization.dashboards
try:
    from ..visualization.dashboards import (
        InsightSpikeVisualizer,
        quick_comparison,
        quick_performance_chart,
        quick_progress_chart,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Re-export from reorganized modules for backward compatibility
try:
    from ..metrics.graph_metrics import delta_ged, delta_ig
    from ..processing.embedder import get_model

    if VISUALIZATION_AVAILABLE:
        __all__ = [
            "get_model",
            "clean_text",
            "iter_text",
            "delta_ged",
            "delta_ig",
            "InsightSpikeVisualizer",
            "quick_performance_chart",
            "quick_comparison",
            "quick_progress_chart",
        ]
    else:
        __all__ = ["get_model", "clean_text", "iter_text", "delta_ged", "delta_ig"]
except ImportError as e:
    print(f"Warning: Some utilities could not be imported: {e}")
    if VISUALIZATION_AVAILABLE:
        __all__ = [
            "clean_text",
            "iter_text",
            "InsightSpikeVisualizer",
            "quick_performance_chart",
            "quick_comparison",
            "quick_progress_chart",
        ]
    else:
        __all__ = ["clean_text", "iter_text"]

# Export graph importance utilities - moved to algorithms
try:
    from ..algorithms.graph_importance import DynamicImportanceTracker, GraphImportanceCalculator

    __all__.extend(["GraphImportanceCalculator", "DynamicImportanceTracker"])
except ImportError:
    pass


# Model availability utilities
def get_available_models():
    """Get list of available embedding models."""
    # Default models that are commonly available
    return [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1",
    ]


__all__.append("get_available_models")
