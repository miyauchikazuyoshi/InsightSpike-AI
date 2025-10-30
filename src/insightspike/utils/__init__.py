"""Utility modules for InsightSpike-AI

Diagnostic/minimal import fast-path:
 If INSIGHTSPIKE_DIAG_IMPORT=1 または INSIGHTSPIKE_MIN_IMPORT=1 が設定されている場合、
 重い可視化/metrics/graph 方面の import をスキップし、main_agent 初期 import の
 ハング原因となりうる循環依存を回避する。必要になったら個別モジュール側で
 遅延 import する運用に切り替える。
"""
from __future__ import annotations
import os

_FAST_PATH = os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1' or os.getenv('INSIGHTSPIKE_MIN_IMPORT') == '1'
if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
    print('[utils.__init__] import start (fast_path=%s)' % _FAST_PATH, flush=True)

# Always cheap essentials
from .text_utils import clean_text, iter_text  # noqa: E402

if _FAST_PATH:
    # Provide only minimal surface; skip heavy optional imports.
    def get_available_models():  # minimal duplicate (kept consistent with full path)
        return [
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
        ]
    __all__ = ["clean_text", "iter_text", "get_available_models"]
    # Fast path ends here to avoid triggering heavy sub-imports during diagnostics.
    if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
        print('[utils.__init__] fast-path exports ready', flush=True)
else:
    # Full (original) behavior below
    # Export new visualization utilities - moved to visualization.dashboards
    try:
        from ..visualization.dashboards import (  # noqa: E402
            InsightSpikeVisualizer,
            quick_comparison,
            quick_performance_chart,
            quick_progress_chart,
        )

        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False

    # Re-export from reorganized modules for backward compatibility
    try:  # noqa: E402
        from ..metrics.graph_metrics import delta_ged, delta_ig
        # Avoid importing embedder.get_model here to prevent circular import during CLI startup
        if VISUALIZATION_AVAILABLE:
            __all__ = [
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
            __all__ = ["clean_text", "iter_text", "delta_ged", "delta_ig"]
    except ImportError as e:  # pragma: no cover
        # Keep fallback minimal and silent to not pollute CLI stdout required by tests
        if 'VISUALIZATION_AVAILABLE' in globals() and VISUALIZATION_AVAILABLE:
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
    try:  # noqa: E402
        from ..algorithms.graph_importance import (  # noqa: F401
            DynamicImportanceTracker,
            GraphImportanceCalculator,
        )

        __all__.extend(["GraphImportanceCalculator", "DynamicImportanceTracker"])  # type: ignore
    except ImportError:
        pass

    # Model availability utilities
    def get_available_models():  # noqa: E402
        """Get list of available embedding models."""
        # Default models that are commonly available
        return [
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
        ]

    __all__.append("get_available_models")  # type: ignore
    if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
        print('[utils.__init__] full-path exports ready', flush=True)

    # Keep any previous late imports (visualization already handled).


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
    # Avoid importing embedder.get_model here to prevent circular import during CLI startup
    if VISUALIZATION_AVAILABLE:
        __all__ = [
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
        __all__ = ["clean_text", "iter_text", "delta_ged", "delta_ig"]
except ImportError as e:  # pragma: no cover
    # Keep fallback minimal and silent to not pollute CLI stdout required by tests
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
    from ..algorithms.graph_importance import (
        DynamicImportanceTracker,
        GraphImportanceCalculator,
    )

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
