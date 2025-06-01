"""
Insight Detection Module
=======================

Eureka spike detection and insight fact registry.
"""

from .eureka_spike import EurekaDetector, detect_eureka_spike
from .insight_registry import InsightFactRegistry

__all__ = ["EurekaDetector", "detect_eureka_spike", "InsightFactRegistry"]
