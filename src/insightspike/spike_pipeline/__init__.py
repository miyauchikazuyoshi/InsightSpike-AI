"""Spike detection pipeline components.

This package provides a modular pipeline for spike detection,
breaking down the monolithic spike detection into composable stages.
"""

from .collector import SpikeDataCollector
from .analyzer import SpikeStatsAnalyzer  
from .detector import SpikeDecisionEngine, SpikeDecisionMode
from .processor import SpikePostProcessor
from .pipeline import SpikePipeline

__all__ = [
    'SpikeDataCollector',
    'SpikeStatsAnalyzer',
    'SpikeDecisionEngine',
    'SpikeDecisionMode', 
    'SpikePostProcessor',
    'SpikePipeline'
]