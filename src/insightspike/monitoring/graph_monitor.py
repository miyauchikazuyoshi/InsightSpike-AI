"""
Graph Operations Monitoring
==========================

Monitor and log performance metrics for scalable graph operations.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GraphOperationMetric:
    """Single metric for a graph operation."""
    operation: str
    timestamp: float
    duration: float
    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def nodes_added(self) -> int:
        return self.nodes_after - self.nodes_before
    
    @property
    def edges_added(self) -> int:
        return self.edges_after - self.edges_before
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "edges_before": self.edges_before,
            "edges_after": self.edges_after,
            "nodes_added": self.nodes_added,
            "edges_added": self.edges_added,
            "metadata": self.metadata
        }


class GraphOperationMonitor:
    """
    Monitor graph operations for performance and debugging.
    
    Features:
    - Operation timing
    - Graph size tracking
    - Performance metrics aggregation
    - Anomaly detection
    - Detailed logging
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_history: int = 1000,
        enable_file_logging: bool = True
    ):
        self.log_dir = log_dir or Path("data/logs/graph_operations")
        self.max_history = max_history
        self.enable_file_logging = enable_file_logging
        
        # Operation history
        self.metrics_history: deque = deque(maxlen=max_history)
        
        # Aggregated statistics
        self.operation_stats = defaultdict(lambda: {
            "count": 0,
            "total_duration": 0.0,
            "min_duration": float('inf'),
            "max_duration": 0.0,
            "total_nodes_added": 0,
            "total_edges_added": 0,
            "errors": 0
        })
        
        # Performance thresholds for warnings
        self.performance_thresholds = {
            "add_node": 0.1,  # 100ms
            "build_graph": 5.0,  # 5 seconds
            "search": 0.5,  # 500ms
            "update_importance": 1.0,  # 1 second
            "conflict_detection": 0.2,  # 200ms
        }
        
        # Ensure log directory exists
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def start_operation(self, operation: str, graph_state: Dict[str, int]) -> Dict[str, Any]:
        """Start monitoring an operation."""
        return {
            "operation": operation,
            "start_time": time.time(),
            "nodes_before": graph_state.get("nodes", 0),
            "edges_before": graph_state.get("edges", 0)
        }
    
    def end_operation(
        self,
        context: Dict[str, Any],
        graph_state: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> GraphOperationMetric:
        """End monitoring an operation and record metrics."""
        end_time = time.time()
        duration = end_time - context["start_time"]
        
        # Create metric
        metric = GraphOperationMetric(
            operation=context["operation"],
            timestamp=context["start_time"],
            duration=duration,
            nodes_before=context["nodes_before"],
            nodes_after=graph_state.get("nodes", 0),
            edges_before=context["edges_before"],
            edges_after=graph_state.get("edges", 0),
            metadata=metadata or {}
        )
        
        # Add error info if present
        if error:
            metric.metadata["error"] = str(error)
            metric.metadata["error_type"] = type(error).__name__
        
        # Record metric
        self._record_metric(metric, error is not None)
        
        # Check performance
        self._check_performance(metric)
        
        return metric
    
    def _record_metric(self, metric: GraphOperationMetric, is_error: bool = False):
        """Record a metric in history and update statistics."""
        # Add to history
        self.metrics_history.append(metric)
        
        # Update statistics
        stats = self.operation_stats[metric.operation]
        stats["count"] += 1
        stats["total_duration"] += metric.duration
        stats["min_duration"] = min(stats["min_duration"], metric.duration)
        stats["max_duration"] = max(stats["max_duration"], metric.duration)
        stats["total_nodes_added"] += metric.nodes_added
        stats["total_edges_added"] += metric.edges_added
        
        if is_error:
            stats["errors"] += 1
        
        # Log to file if enabled
        if self.enable_file_logging:
            self._log_to_file(metric)
    
    def _check_performance(self, metric: GraphOperationMetric):
        """Check if operation performance is within thresholds."""
        threshold = self.performance_thresholds.get(metric.operation)
        
        if threshold and metric.duration > threshold:
            logger.warning(
                f"Slow {metric.operation} operation: {metric.duration:.3f}s "
                f"(threshold: {threshold}s)"
            )
            
            # Additional context for debugging
            if metric.nodes_added > 100 or metric.edges_added > 1000:
                logger.warning(
                    f"Large graph change detected: "
                    f"+{metric.nodes_added} nodes, +{metric.edges_added} edges"
                )
    
    def _log_to_file(self, metric: GraphOperationMetric):
        """Log metric to file for analysis."""
        try:
            # Create daily log file
            date_str = datetime.fromtimestamp(metric.timestamp).strftime("%Y%m%d")
            log_file = self.log_dir / f"graph_operations_{date_str}.jsonl"
            
            # Append metric
            with open(log_file, "a") as f:
                json.dump(metric.to_dict(), f)
                f.write("\n")
                
        except Exception as e:
            logger.error(f"Failed to log metric to file: {e}")
    
    def get_operation_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for operations."""
        if operation:
            stats = self.operation_stats.get(operation, {})
            if stats.get("count", 0) > 0:
                return {
                    "operation": operation,
                    "count": stats["count"],
                    "avg_duration": stats["total_duration"] / stats["count"],
                    "min_duration": stats["min_duration"],
                    "max_duration": stats["max_duration"],
                    "avg_nodes_added": stats["total_nodes_added"] / stats["count"],
                    "avg_edges_added": stats["total_edges_added"] / stats["count"],
                    "error_rate": stats["errors"] / stats["count"]
                }
            return {}
        
        # Return all operation summaries
        summaries = {}
        for op, stats in self.operation_stats.items():
            if stats["count"] > 0:
                summaries[op] = {
                    "count": stats["count"],
                    "avg_duration": stats["total_duration"] / stats["count"],
                    "min_duration": stats["min_duration"],
                    "max_duration": stats["max_duration"],
                    "avg_nodes_added": stats["total_nodes_added"] / stats["count"],
                    "avg_edges_added": stats["total_edges_added"] / stats["count"],
                    "error_rate": stats["errors"] / stats["count"]
                }
        
        return summaries
    
    def get_recent_operations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent operations."""
        recent = list(self.metrics_history)[-n:]
        return [metric.to_dict() for metric in recent]
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous operations based on historical data."""
        anomalies = []
        
        for operation, stats in self.operation_stats.items():
            if stats["count"] < 10:  # Need enough data
                continue
            
            avg_duration = stats["total_duration"] / stats["count"]
            
            # Check recent operations for anomalies
            recent_ops = [
                m for m in self.metrics_history 
                if m.operation == operation
            ][-10:]
            
            for metric in recent_ops:
                # Duration anomaly (3x average)
                if metric.duration > 3 * avg_duration:
                    anomalies.append({
                        "type": "slow_operation",
                        "operation": operation,
                        "timestamp": metric.timestamp,
                        "duration": metric.duration,
                        "expected_duration": avg_duration,
                        "factor": metric.duration / avg_duration
                    })
                
                # Large graph change anomaly
                avg_edges = stats["total_edges_added"] / stats["count"]
                if avg_edges > 0 and metric.edges_added > 10 * avg_edges:
                    anomalies.append({
                        "type": "large_edge_change",
                        "operation": operation,
                        "timestamp": metric.timestamp,
                        "edges_added": metric.edges_added,
                        "expected_edges": avg_edges,
                        "factor": metric.edges_added / avg_edges
                    })
        
        return anomalies
    
    def export_metrics(self, output_file: Path) -> bool:
        """Export all metrics to a file for analysis."""
        try:
            data = {
                "export_time": time.time(),
                "metrics": [m.to_dict() for m in self.metrics_history],
                "statistics": dict(self.operation_stats),
                "anomalies": self.detect_anomalies()
            }
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported metrics to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def reset(self):
        """Reset all monitoring data."""
        self.metrics_history.clear()
        self.operation_stats.clear()
        logger.info("Graph monitor reset")


# Context manager for easy operation monitoring
class MonitoredOperation:
    """Context manager for monitoring graph operations."""
    
    def __init__(
        self,
        monitor: GraphOperationMonitor,
        operation: str,
        graph_state_func: callable,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.monitor = monitor
        self.operation = operation
        self.graph_state_func = graph_state_func
        self.metadata = metadata or {}
        self.context = None
        self.error = None
    
    def __enter__(self):
        self.context = self.monitor.start_operation(
            self.operation,
            self.graph_state_func()
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.error = exc_val
        self.monitor.end_operation(
            self.context,
            self.graph_state_func(),
            self.metadata,
            self.error
        )
        return False  # Don't suppress exceptions


def create_default_monitor() -> GraphOperationMonitor:
    """Create a monitor with default settings."""
    from ..core.config import get_config
    config = get_config()
    
    return GraphOperationMonitor(
        log_dir=config.paths.log_dir / "graph_operations",
        enable_file_logging=True
    )