"""Algorithms Module (light-mode aware).

軽量化ポリシー:
 - 旧: ``INSIGHT_SPIKE_LIGHT_MODE=1``
 - 新: ``INSIGHTSPIKE_LITE_MODE=1`` または ``INSIGHTSPIKE_MIN_IMPORT=1`` (他モジュールと統一)

上記いずれかの環境変数が `"1"` の場合、torch 等の重量依存を伴う再エクスポートを完全にスキップし、
直接サブモジュールを import した際 (``from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger`` など)
に余計な初期化コストが掛からないようにする。

Direct submodule imports (e.g. ``from insightspike.algorithms.gedig_core import GeDIGCore``) remain
usable regardless of this file's light-mode path because they bypass the heavy re-export list.
"""

from __future__ import annotations
import os

# Recognize all supported lite-mode flags (backward + current unified flags)
LIGHT_MODE = (
    os.environ.get("INSIGHT_SPIKE_LIGHT_MODE") == "1"
    or os.environ.get("INSIGHTSPIKE_LITE_MODE") == "1"
    or os.environ.get("INSIGHTSPIKE_MIN_IMPORT") == "1"
)

if LIGHT_MODE:
    # Provide minimal surface; heavy algorithm imports skipped to avoid torch dependency.
    __all__ = [
        # Intentionally left almost empty; users should import submodules directly in light mode.
    ]
else:
    from .entropy_calculator import (
        ContentStructureSeparation,
        EntropyCalculator,
        EntropyResult,
    )
    from .graph_edit_distance import (
        GEDResult,
        GraphEditDistance,
        OptimizationLevel,
        compute_delta_ged,
        compute_graph_edit_distance,
    )
    from .information_gain import (
        EntropyMethod,
        IGResult,
        InformationGain,
        compute_delta_ig,
        compute_information_gain,
        compute_shannon_entropy,
    )
    from .structural_entropy import (
        clustering_coefficient_entropy,
        degree_distribution_entropy,
        path_length_entropy,
        structural_entropy,
        von_neumann_entropy,
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
        "compute_delta_ig",
        # Structural Entropy components
        "degree_distribution_entropy",
        "von_neumann_entropy",
        "structural_entropy",
        "clustering_coefficient_entropy",
        "path_length_entropy",
        # Unified Entropy Calculator
        "EntropyCalculator",
        "EntropyResult",
        "ContentStructureSeparation",
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
            "precise": "O(n!) - use with caution",
        },
    },
    "information_gain": {
        "description": "Measures information gain during learning processes",
        "mathematical_form": "ΔIG = H(S_before) - H(S_after)",
        "insight_threshold": 0.2,
        "entropy_methods": [
            "shannon",
            "clustering",
            "mutual_info",
            "feature_entropy",
            "structural",
            "degree_distribution",
            "von_neumann",
        ],
        "complexity": "O(n log n) for most methods",
    },
    "fusion_scheme": {
        "description": "Combined insight detection using ΔGED and ΔIG",
        "mathematical_form": "R = w₁×ΔGED + w₂×ΔIG",
        "default_weights": {"ged": 0.5, "ig": 0.5},
        "eureka_conditions": "ΔGED ≤ -0.5 AND ΔIG ≥ 0.2",
    },
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
        "timeout_seconds": 5.0,
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
    defaults = {"method": EntropyMethod.CLUSTERING, "k_clusters": 8, "min_samples": 2}
    defaults.update(kwargs)
    return InformationGain(**defaults)
