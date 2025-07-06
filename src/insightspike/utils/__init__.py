"""Utility modules for InsightSpike-AI"""

# Export commonly used utilities
from .prompt_builder import *
from .text_utils import clean_text, iter_text

# Export new visualization utilities
try:
    from .visualization import (
        InsightSpikeVisualizer, 
        quick_performance_chart, 
        quick_comparison, 
        quick_progress_chart
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization utilities not available: {e}")
    VISUALIZATION_AVAILABLE = False

# Re-export from reorganized modules for backward compatibility
try:
    from ..metrics.graph_metrics import delta_ged, delta_ig
    from ..processing.embedder import get_model

    if VISUALIZATION_AVAILABLE:
        __all__ = [
            "get_model", "clean_text", "iter_text", "delta_ged", "delta_ig",
            "InsightSpikeVisualizer", "quick_performance_chart", 
            "quick_comparison", "quick_progress_chart"
        ]
    else:
        __all__ = ["get_model", "clean_text", "iter_text", "delta_ged", "delta_ig"]
except ImportError as e:
    print(f"Warning: Some utilities could not be imported: {e}")
    if VISUALIZATION_AVAILABLE:
        __all__ = [
            "clean_text", "iter_text", "InsightSpikeVisualizer", 
            "quick_performance_chart", "quick_comparison", "quick_progress_chart"
        ]
    else:
        __all__ = ["clean_text", "iter_text"]

# Export graph importance utilities
try:
    from .graph_importance import GraphImportanceCalculator, DynamicImportanceTracker
    __all__.extend(["GraphImportanceCalculator", "DynamicImportanceTracker"])
except ImportError:
    pass
