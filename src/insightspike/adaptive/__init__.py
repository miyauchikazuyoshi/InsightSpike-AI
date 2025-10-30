"""
Adaptive Processing Module
=========================

Implements adaptive exploration loop for efficient spike detection
with minimal LLM API calls.
"""

from .core.adaptive_processor import AdaptiveProcessor
from .core.exploration_loop import ExplorationLoop
from .core.interfaces import (
    ExplorationParams,
    ExplorationResult,
    ExplorationStrategy,
    PatternLearner,
    TopKCalculator,
)

__all__ = [
    "AdaptiveProcessor",
    "ExplorationLoop",
    "ExplorationParams",
    "ExplorationResult",
    "ExplorationStrategy",
    "TopKCalculator",
    "PatternLearner",
]