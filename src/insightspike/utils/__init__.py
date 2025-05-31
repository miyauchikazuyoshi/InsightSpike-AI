"""Utility modules for InsightSpike-AI"""

# Export commonly used utilities
from .embedder import get_model
from .text_utils import clean_text, iter_text

try:
    from .graph_metrics import delta_ged, delta_ig
    __all__ = ["get_model", "clean_text", "iter_text", "delta_ged", "delta_ig"]
except ImportError as e:
    print(f"Warning: Some utilities could not be imported: {e}")
    __all__ = ["get_model", "clean_text", "iter_text"]
