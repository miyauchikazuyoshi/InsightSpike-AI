"""Utility modules for InsightSpike-AI"""

# Export commonly used utilities
from .prompt_builder import *
from .text_utils import clean_text, iter_text

# Re-export from reorganized modules for backward compatibility
try:
    from ..metrics.graph_metrics import delta_ged, delta_ig
    from ..processing.embedder import get_model

    __all__ = ["get_model", "clean_text", "iter_text", "delta_ged", "delta_ig"]
except ImportError as e:
    print(f"Warning: Some utilities could not be imported: {e}")
    __all__ = ["clean_text", "iter_text"]
