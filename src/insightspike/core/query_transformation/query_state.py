"""
Query State Management
Tracks how a query evolves through the knowledge graph
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch
from datetime import datetime


@dataclass
class QueryState:
    """Represents the state of a query at a point in time"""
    
    # Core attributes
    text: str
    embedding: Optional[torch.Tensor] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Transformation tracking
    stage: str = "initial"  # initial -> exploring -> transforming -> insight -> complete
    confidence: float = 0.0
    color: str = "yellow"  # yellow -> orange -> green (visual metaphor)
    
    # Knowledge gained
    absorbed_concepts: List[str] = field(default_factory=list)
    discovered_connections: List[tuple] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    
    # Graph position
    graph_position: Optional[Dict[str, Any]] = None
    connected_nodes: List[str] = field(default_factory=list)
    edge_weights: Dict[str, float] = field(default_factory=dict)
    
    # Metrics
    transformation_magnitude: float = 0.0
    information_gain: float = 0.0
    structural_change: float = 0.0
    
    def update_color(self):
        """Update color based on transformation progress"""
        if self.confidence < 0.3:
            self.color = "yellow"
        elif self.confidence < 0.7:
            self.color = "orange"
        else:
            self.color = "green"
    
    def add_insight(self, insight: str):
        """Add a new insight and update metrics"""
        self.insights.append(insight)
        self.information_gain += 0.2
        self.confidence = min(1.0, self.confidence + 0.1)
        self.update_color()
    
    def absorb_concept(self, concept: str, strength: float = 1.0):
        """Absorb knowledge from a connected node"""
        self.absorbed_concepts.append(concept)
        self.transformation_magnitude += strength * 0.1
        self.confidence = min(1.0, self.confidence + strength * 0.05)
        self.update_color()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "stage": self.stage,
            "confidence": self.confidence,
            "color": self.color,
            "absorbed_concepts": self.absorbed_concepts,
            "insights": self.insights,
            "transformation_magnitude": self.transformation_magnitude,
            "information_gain": self.information_gain,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QueryTransformationHistory:
    """Tracks the complete transformation journey of a query"""
    
    initial_query: str
    states: List[QueryState] = field(default_factory=list)
    
    def add_state(self, state: QueryState):
        """Add a new state to the history"""
        self.states.append(state)
    
    def get_current_state(self) -> Optional[QueryState]:
        """Get the most recent state"""
        return self.states[-1] if self.states else None
    
    def get_transformation_path(self) -> List[str]:
        """Get the path of stages the query went through"""
        return [state.stage for state in self.states]
    
    def get_total_insights(self) -> List[str]:
        """Get all insights discovered during transformation"""
        insights = []
        for state in self.states:
            insights.extend(state.insights)
        return list(set(insights))  # Remove duplicates
    
    def get_confidence_trajectory(self) -> List[float]:
        """Get how confidence evolved over time"""
        return [state.confidence for state in self.states]
    
    def reached_insight(self) -> bool:
        """Check if the query reached an insight state"""
        return any(state.stage == "insight" for state in self.states)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "initial_query": self.initial_query,
            "num_states": len(self.states),
            "transformation_path": self.get_transformation_path(),
            "total_insights": self.get_total_insights(),
            "confidence_trajectory": self.get_confidence_trajectory(),
            "reached_insight": self.reached_insight(),
            "states": [state.to_dict() for state in self.states]
        }