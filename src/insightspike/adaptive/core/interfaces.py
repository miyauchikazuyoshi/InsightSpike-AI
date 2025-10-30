"""
Clean interfaces for adaptive processing components
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class ExplorationParams:
    """Parameters for a single exploration attempt"""
    radius: float
    topk_l1: int
    topk_l2: int
    topk_l3: int
    temperature: float
    attempt_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "radius": self.radius,
            "topk_l1": self.topk_l1,
            "topk_l2": self.topk_l2,
            "topk_l3": self.topk_l3,
            "temperature": self.temperature,
            "attempt_number": self.attempt_number
        }


@dataclass
class ExplorationResult:
    """Result of a single exploration attempt"""
    spike_detected: bool
    confidence: float
    retrieved_docs: List[Dict[str, Any]]
    graph_analysis: Dict[str, Any]
    metrics: Dict[str, float]
    params: ExplorationParams
    
    @property
    def has_spike(self) -> bool:
        """Alias for spike_detected"""
        return self.spike_detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "spike_detected": self.spike_detected,
            "confidence": self.confidence,
            "retrieved_doc_count": len(self.retrieved_docs),
            "graph_analysis": self.graph_analysis,
            "metrics": self.metrics,
            "params": self.params.to_dict()
        }


class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies"""
    
    @abstractmethod
    def get_initial_params(self) -> ExplorationParams:
        """Get initial exploration parameters"""
        pass
    
    @abstractmethod
    def adjust_params(
        self, 
        attempt: int, 
        prev_result: ExplorationResult
    ) -> ExplorationParams:
        """Adjust parameters based on previous result"""
        pass
    
    @abstractmethod
    def should_continue(
        self, 
        results: List[ExplorationResult]
    ) -> bool:
        """Decide whether to continue exploration"""
        pass


class TopKCalculator(ABC):
    """Abstract base class for topK calculation"""
    
    @abstractmethod
    def calculate(
        self, 
        l1_analysis: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Calculate adaptive topK values based on L1 analysis
        
        Returns:
            Dict with keys: layer1_k, layer2_k, layer3_k
        """
        pass


class PatternLearner(ABC):
    """Abstract base class for pattern learning"""
    
    @abstractmethod
    def record_success(
        self, 
        query: str, 
        exploration_path: List[ExplorationParams], 
        final_result: ExplorationResult
    ):
        """Record a successful exploration pattern"""
        pass
    
    @abstractmethod
    def suggest_params(
        self, 
        query: str
    ) -> Optional[ExplorationParams]:
        """
        Suggest exploration parameters based on learned patterns
        
        Returns None if no suitable pattern found
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        pass