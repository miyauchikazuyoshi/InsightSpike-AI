"""
InsightSpike-AI Core Layer Interfaces
===================================

Abstract base classes for all layers in the architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LayerInput:
    """Standard input format for all layers"""
    data: Any
    metadata: Dict[str, Any] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class LayerOutput:
    """Standard output format for all layers"""
    result: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    metrics: Dict[str, float] = None


class LayerInterface(ABC):
    """Base interface for all InsightSpike layers"""
    
    def __init__(self, layer_id: str, config: Dict[str, Any] = None):
        self.layer_id = layer_id
        self.config = config or {}
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the layer"""
        pass
    
    @abstractmethod
    def process(self, input_data: LayerInput) -> LayerOutput:
        """Process input through this layer"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def is_ready(self) -> bool:
        """Check if layer is ready for processing"""
        return self._is_initialized


class L1ErrorMonitorInterface(LayerInterface):
    """Layer 1: Error Monitor (Cerebellum analog)"""
    
    @abstractmethod
    def calculate_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate prediction error"""
        pass
    
    @abstractmethod
    def get_uncertainty(self, input_sequence: List[Any]) -> float:
        """Get uncertainty measure for input"""
        pass


class L2MemoryInterface(LayerInterface):
    """Layer 2: Memory Manager (LC + Hippocampus analog)"""
    
    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[int]]:
        """Search similar episodes in memory"""
        pass
    
    @abstractmethod
    def add_episode(self, vector: np.ndarray, text: str, c_value: float = 0.2) -> int:
        """Add new episode to memory"""
        pass
    
    @abstractmethod
    def update_c_values(self, episode_ids: List[int], rewards: List[float]):
        """Update C-values based on rewards"""
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        pass


class L3GraphReasonerInterface(LayerInterface):
    """Layer 3: Graph Reasoner (PFC analog)"""
    
    @abstractmethod
    def build_graph(self, vectors: np.ndarray) -> Any:
        """Build similarity graph from vectors"""
        pass
    
    @abstractmethod
    def calculate_ged(self, graph1: Any, graph2: Any) -> float:
        """Calculate graph edit distance"""
        pass
    
    @abstractmethod
    def calculate_ig(self, old_state: Any, new_state: Any) -> float:
        """Calculate information gain"""
        pass
    
    @abstractmethod
    def detect_eureka_spike(self, delta_ged: float, delta_ig: float) -> bool:
        """Detect if current state constitutes a eureka spike"""
        pass


class L4LLMInterface(LayerInterface):
    """Layer 4: Language Model Interface"""
    
    @abstractmethod
    def generate_response(self, context: str, question: str) -> str:
        """Generate response based on context and question"""
        pass
    
    @abstractmethod
    def format_context(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodes into context string"""
        pass


class AgentInterface(ABC):
    """Base interface for agents that coordinate layers"""
    
    def __init__(self, agent_id: str, layers: Dict[str, LayerInterface]):
        self.agent_id = agent_id
        self.layers = layers
    
    @abstractmethod
    def execute_cycle(self, input_data: Any) -> Dict[str, Any]:
        """Execute one reasoning cycle"""
        pass
    
    @abstractmethod
    def initialize_layers(self) -> bool:
        """Initialize all layers"""
        pass


# Export main interfaces
__all__ = [
    'LayerInput', 'LayerOutput', 'LayerInterface',
    'L1ErrorMonitorInterface', 'L2MemoryInterface', 
    'L3GraphReasonerInterface', 'L4LLMInterface',
    'AgentInterface'
]
