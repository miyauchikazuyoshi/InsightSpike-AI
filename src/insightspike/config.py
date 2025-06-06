# Legacy config compatibility module
"""
Legacy configuration compatibility for InsightSpike-AI
====================================================

This module provides backward compatibility for the old config.py structure.
"""

from datetime import datetime

from insightspike.core.config import get_config, get_legacy_config

# Import all legacy config values
_legacy = get_legacy_config()

# Export legacy constants
ROOT_DIR = _legacy["ROOT_DIR"]
DATA_DIR = _legacy["DATA_DIR"]
LOG_DIR = _legacy["LOG_DIR"]
INDEX_FILE = _legacy["INDEX_FILE"]
GRAPH_FILE = _legacy["GRAPH_FILE"]
EMBED_MODEL_NAME = _legacy["EMBED_MODEL_NAME"]
LLM_NAME = _legacy["LLM_NAME"]
SIM_THRESHOLD = _legacy["SIM_THRESHOLD"]
TOP_K = _legacy["TOP_K"]
LAYER1_TOP_K = _legacy["LAYER1_TOP_K"]
LAYER2_TOP_K = _legacy["LAYER2_TOP_K"]
LAYER3_TOP_K = _legacy["LAYER3_TOP_K"]
SPIKE_GED = _legacy["SPIKE_GED"]
SPIKE_IG = _legacy["SPIKE_IG"]
ETA_SPIKE = _legacy["ETA_SPIKE"]
MERGE_GED = _legacy["MERGE_GED"]
SPLIT_IG = _legacy["SPLIT_IG"]
PRUNE_C = _legacy["PRUNE_C"]
INACTIVE_N = _legacy["INACTIVE_N"]


def timestamp() -> str:
    """Generate timestamp in the legacy format"""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# Ensure directories exist
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # Ignore errors if directories already exist or permissions issue
