"""
Constants and Enumerations
==========================

Central location for constants and enums used throughout InsightSpike.
"""

from enum import Enum
from typing import Dict


class FileFormat(Enum):
    """Supported file formats for data storage."""

    JSON = ".json"
    NUMPY = ".npy"
    PICKLE = ".pkl"
    PYTORCH = ".pt"
    TEXT = ".txt"
    YAML = ".yaml"


class DataType(Enum):
    """Types of data stored by the system."""

    EPISODES = "episodes"
    GRAPH = "graph"
    METADATA = "metadata"
    VECTORS = "vectors"
    CONFIG = "config"


# File format mappings for each data type
FILE_FORMAT_MAPPING: Dict[DataType, FileFormat] = {
    DataType.EPISODES: FileFormat.JSON,
    DataType.GRAPH: FileFormat.PYTORCH,
    DataType.METADATA: FileFormat.JSON,
    DataType.VECTORS: FileFormat.NUMPY,
    DataType.CONFIG: FileFormat.YAML,
}


# Default values for various components
class Defaults:
    """Default values used throughout the system."""

    # Episode defaults
    EPISODE_C_VALUE = 0.5
    EPISODE_MERGE_THRESHOLD = 0.8

    # Graph defaults
    OPTIMAL_GRAPH_SIZE = 10
    GRAPH_SIMILARITY_THRESHOLD = 0.3

    # Reward calculation weights
    REWARD_WEIGHT_GED = 0.5
    REWARD_WEIGHT_IG = 0.5
    REWARD_WEIGHT_CONFLICT = 0.0  # Deprecated, conflicts are no longer used in reward calculation

    # Spike detection thresholds
    SPIKE_GED_THRESHOLD = -0.5
    SPIKE_IG_THRESHOLD = 0.2
    SPIKE_CONFLICT_THRESHOLD = 0.5
