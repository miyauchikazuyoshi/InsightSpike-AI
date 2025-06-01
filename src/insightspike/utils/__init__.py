"""Utility modules for InsightSpike-AI"""

# Export commonly used utilities  
from .text_utils import clean_text, iter_text
from .prompt_builder import *

# Re-export from reorganized modules for backward compatibility
try:
    from ..processing.embedder import get_model
    from ..metrics.graph_metrics import delta_ged, delta_ig
    __all__ = ["get_model", "clean_text", "iter_text", "delta_ged", "delta_ig"]
except ImportError as e:
    print(f"Warning: Some utilities could not be imported: {e}")
    __all__ = ["clean_text", "iter_text"]
