"""geDIG algorithms package (selector/orchestration/legacy).

Note: Prefer using `insightspike.algorithms.gedig.selector.compute_gedig` as the
canonical entrypoint from within this repository.
"""

from .types import (
    ProcessingMode,
    SpikeDetectionMode,
    HopResult,
    GeDIGResult,
    LinksetMetrics,
)
from .config import (
    GeDIGConfig,
    GeDIGPresets,
)
from .spike import (
    detect_spike,
    compute_rewards,
)
from .graph_utils import (
    graph_efficiency,
    spectral_score,
    avg_shortest_path_length_safe,
    compute_sp_gain_norm,
    extract_k_hop_subgraph,
    trim_terminal_edges,
    ensure_networkx,
    pyg_to_networkx,
    extract_features,
    filter_features,
    compute_ged_min_proxy,
)
from .monitor import GeDIGMonitor
from .logger import GeDIGLogger
from .linkset import compute_linkset_metrics
from .multihop import calculate_multihop

__all__ = [
    # Types
    "ProcessingMode",
    "SpikeDetectionMode",
    "HopResult",
    "GeDIGResult",
    "LinksetMetrics",
    # Config
    "GeDIGConfig",
    "GeDIGPresets",
    # Spike detection
    "detect_spike",
    "compute_rewards",
    # Graph utilities
    "graph_efficiency",
    "spectral_score",
    "avg_shortest_path_length_safe",
    "compute_sp_gain_norm",
    "extract_k_hop_subgraph",
    "trim_terminal_edges",
    "ensure_networkx",
    "pyg_to_networkx",
    "extract_features",
    "filter_features",
    "compute_ged_min_proxy",
    # Monitoring and Logging
    "GeDIGMonitor",
    "GeDIGLogger",
    # Linkset
    "compute_linkset_metrics",
    # Multihop
    "calculate_multihop",
]

