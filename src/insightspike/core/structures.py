"""
Core Data Structures
===================

Common data structures used throughout the system.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class GraphMetrics:
    """Metrics from graph analysis."""
    
    ged_value: float = 0.0
    ig_value: float = 0.0
    entropy: float = 0.0
    clustering_coefficient: float = 0.0
    node_count: int = 0
    edge_count: int = 0


@dataclass
class CycleResult:
    """Result from a single reasoning cycle."""
    
    response: str
    reasoning_trace: str
    memory_used: List[Dict[str, Any]]
    spike_detected: bool
    graph_metrics: Dict[str, float]
    reasoning_quality: float
    convergence_score: float
    has_spike: bool  # For backward compatibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "reasoning_trace": self.reasoning_trace,
            "memory_used": self.memory_used,
            "spike_detected": self.spike_detected,
            "graph_metrics": self.graph_metrics,
            "reasoning_quality": self.reasoning_quality,
            "convergence_score": self.convergence_score,
            "has_spike": self.has_spike
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleResult":
        """Create from dictionary."""
        return cls(
            response=data.get("response", ""),
            reasoning_trace=data.get("reasoning_trace", ""),
            memory_used=data.get("memory_used", []),
            spike_detected=data.get("spike_detected", False),
            graph_metrics=data.get("graph_metrics", {}),
            reasoning_quality=data.get("reasoning_quality", 0.0),
            convergence_score=data.get("convergence_score", 0.0),
            has_spike=data.get("has_spike", False)
        )