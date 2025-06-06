"""
InsightSpike-AI: Production Monitoring Package
Comprehensive monitoring and alerting system for large-scale deployments.
"""

from .production_monitor import (
    ProductionMonitor,
    PerformanceMetrics,
    SystemAlert,
    MonitoringConfig
)

__version__ = "1.0.0"
__author__ = "InsightSpike-AI Team"

__all__ = [
    'ProductionMonitor',
    'PerformanceMetrics', 
    'SystemAlert',
    'MonitoringConfig'
]
