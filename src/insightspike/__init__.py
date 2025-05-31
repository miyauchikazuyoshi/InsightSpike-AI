"""InsightSpike package metadata"""
import importlib.util
import os

class About:
    NAME = "InsightSpike-AI"
    VERSION = "0.7-Eureka"

# Export new main agent for easy access
# Check if we're in lite mode for CI testing
LITE_MODE = os.environ.get('INSIGHTSPIKE_LITE_MODE', '0') == '1'

if not LITE_MODE:
    try:
        from .core.agents.main_agent import MainAgent
    except ImportError:
        # Define a placeholder if main_agent is not available
        class MainAgent:
            def __init__(self):
                pass
            def initialize(self):
                return False
            def process_question(self, question, **kwargs):
                return {"response": "MainAgent not available", "success": False}
else:
    # In lite mode, always use placeholder
    class MainAgent:
        def __init__(self):
            pass
        def initialize(self):
            return False
        def process_question(self, question, **kwargs):
            return {"response": "MainAgent not available (lite mode)", "success": False}

# Legacy compatibility exports - import the config.py file specifically
from .config import get_config

# Import the legacy config.py module explicitly to avoid conflict with config/ directory
_config_file = os.path.join(os.path.dirname(__file__), 'config.py')
_spec = importlib.util.spec_from_file_location("legacy_config", _config_file)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

# Legacy module exports for compatibility
from . import embedder
from . import graph_metrics
from . import eureka_spike
from . import utils

# Version info
__version__ = About.VERSION

# Main exports
__all__ = ["MainAgent", "CycleResult", "get_config", "About", "embedder", "graph_metrics", "eureka_spike", "config", "utils"]
