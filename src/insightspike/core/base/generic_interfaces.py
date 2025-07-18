"""
Generic Reusable Interfaces for InsightSpike Framework
====================================================

Abstract interfaces for making InsightSpike components reusable across
different domains and environments, not just maze environments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class TaskType(Enum):
    """Supported task types for insight detection"""

    NAVIGATION = "navigation"
    OPTIMIZATION = "optimization"
    CLASSIFICATION = "classification"
    SEQUENCE_PREDICTION = "sequence_prediction"
    GAME_PLAYING = "game_playing"
    CUSTOM = "custom"


@dataclass
class EnvironmentState:
    """Generic state representation for any environment"""

    # Core state data (flexible format)
    state_data: Union[np.ndarray, Dict[str, Any], List, Tuple]

    # Environment metadata
    environment_type: str
    task_type: TaskType
    state_shape: Optional[Tuple[int, ...]] = None

    # Normalization info for different environments
    state_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # Context information
    step_count: int = 0
    episode_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ActionSpace:
    """Generic action space definition"""

    action_type: str  # "discrete", "continuous", "mixed"
    action_dim: int
    action_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    discrete_actions: Optional[List[str]] = None  # Action names for discrete spaces


@dataclass
class InsightMoment:
    """Generic insight moment representation"""

    # Core insight data
    episode: int
    step: int
    insight_type: str
    description: str

    # Quantitative measures
    dged_value: float  # Δ Global Exploration Difficulty
    dig_value: float  # Δ Information Gain
    confidence: float = 0.0
    performance_impact: float = 0.0

    # Context-specific data
    state: Optional[EnvironmentState] = None
    action: Optional[Any] = None
    reward: Optional[float] = None

    # Additional metadata
    detection_method: str = "default"
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class EnvironmentInterface(ABC):
    """Generic environment interface for any domain"""

    @abstractmethod
    def get_state(self) -> EnvironmentState:
        """Get current environment state"""
        pass

    @abstractmethod
    def get_action_space(self) -> ActionSpace:
        """Get action space definition"""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """Execute action and return (state, reward, done, info)"""
        pass

    @abstractmethod
    def reset(self) -> EnvironmentState:
        """Reset environment and return initial state"""
        pass

    @abstractmethod
    def get_task_type(self) -> TaskType:
        """Get the type of task this environment represents"""
        pass


class RewardNormalizer(ABC):
    """Abstract reward normalizer for different environments"""

    @abstractmethod
    def normalize_reward(self, reward: float, context: Dict[str, Any]) -> float:
        """Normalize reward to standardized scale"""
        pass

    @abstractmethod
    def get_reward_bounds(self) -> Tuple[float, float]:
        """Get expected reward bounds for this environment"""
        pass


class StateEncoder(ABC):
    """Abstract state encoder for different state representations"""

    @abstractmethod
    def encode_state(self, state: EnvironmentState) -> np.ndarray:
        """Encode state to standardized vector representation"""
        pass

    @abstractmethod
    def get_encoding_dim(self) -> int:
        """Get dimensionality of encoded state"""
        pass

    @abstractmethod
    def decode_state(self, encoded_state: np.ndarray) -> EnvironmentState:
        """Decode state from vector back to environment state"""
        pass


class InsightDetectorInterface(ABC):
    """Generic interface for insight detection across domains"""

    def __init__(self, task_type: TaskType, config: Optional[Dict[str, Any]] = None):
        self.task_type = task_type
        self.config = config or {}
        self.insight_history: List[InsightMoment] = []

    @abstractmethod
    def detect_insight(
        self,
        current_state: EnvironmentState,
        action: Any,
        reward: float,
        next_state: EnvironmentState,
        context: Dict[str, Any],
    ) -> Optional[InsightMoment]:
        """Detect if current transition constitutes an insight moment"""
        pass

    @abstractmethod
    def calculate_dged(self, context: Dict[str, Any]) -> float:
        """Calculate Δ Global Exploration Difficulty for this domain"""
        pass

    @abstractmethod
    def calculate_dig(self, context: Dict[str, Any]) -> float:
        """Calculate Δ Information Gain for this domain"""
        pass

    @abstractmethod
    def update_context(self, state: EnvironmentState, action: Any, reward: float):
        """Update internal context for insight detection"""
        pass

    def get_insight_history(self) -> List[InsightMoment]:
        """Get history of detected insights"""
        return self.insight_history.copy()

    def clear_history(self):
        """Clear insight history"""
        self.insight_history.clear()


class GenericAgentInterface(ABC):
    """Generic agent interface that can work with any environment"""

    def __init__(
        self,
        agent_id: str,
        environment: EnvironmentInterface,
        insight_detector: InsightDetectorInterface,
        state_encoder: StateEncoder,
        reward_normalizer: RewardNormalizer,
    ):
        self.agent_id = agent_id
        self.environment = environment
        self.insight_detector = insight_detector
        self.state_encoder = state_encoder
        self.reward_normalizer = reward_normalizer

        # Generic tracking
        self.episode_count = 0
        self.step_count = 0
        self.performance_history: List[float] = []
        self.insight_moments: List[InsightMoment] = []

    @abstractmethod
    def select_action(self, state: EnvironmentState) -> Any:
        """Select action given current state"""
        pass

    @abstractmethod
    def update(
        self,
        state: EnvironmentState,
        action: Any,
        reward: float,
        next_state: EnvironmentState,
        done: bool,
    ):
        """Update agent based on transition"""
        pass

    @abstractmethod
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode and return performance metrics"""
        pass

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all episodes"""
        return {
            "total_episodes": self.episode_count,
            "total_insights": len(self.insight_moments),
            "avg_performance": np.mean(self.performance_history)
            if self.performance_history
            else 0.0,
            "insight_rate": len(self.insight_moments) / max(self.episode_count, 1),
            "task_type": self.environment.get_task_type().value,
        }


class MemoryManagerInterface(ABC):
    """Generic memory manager interface"""

    @abstractmethod
    def store_experience(
        self,
        state: EnvironmentState,
        action: Any,
        reward: float,
        next_state: EnvironmentState,
        insight: Optional[InsightMoment] = None,
    ):
        """Store experience in memory"""
        pass

    @abstractmethod
    def retrieve_similar(
        self, query_state: EnvironmentState, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve k most similar experiences"""
        pass

    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        pass


class ReasonerInterface(ABC):
    """Generic reasoner interface for insight analysis"""

    @abstractmethod
    def analyze_insight_pattern(self, insights: List[InsightMoment]) -> Dict[str, Any]:
        """Analyze patterns in insight sequences"""
        pass

    @abstractmethod
    def predict_next_insight(self, current_context: Dict[str, Any]) -> Dict[str, float]:
        """Predict likelihood of insight in different scenarios"""
        pass


# Export all interfaces
__all__ = [
    "TaskType",
    "EnvironmentState",
    "ActionSpace",
    "InsightMoment",
    "EnvironmentInterface",
    "RewardNormalizer",
    "StateEncoder",
    "InsightDetectorInterface",
    "GenericAgentInterface",
    "MemoryManagerInterface",
    "ReasonerInterface",
]
