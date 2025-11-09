"""
Episode Data Structure
=====================

Core data structure for memory episodes in InsightSpike.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Episode:
    """
    Single memory entry with vector embedding, text, and C-value.

    Attributes:
        text: The text content of the episode
        vec: Vector embedding (numpy array)
        c: C-value for confidence/importance (0.0 to 1.0)
        timestamp: Creation timestamp
        metadata: Additional metadata dictionary
        episode_type: Type of episode (experience, insight, contradiction)
        selection_count: Number of times selected in geDIG evaluation
        creation_time: Original creation timestamp (for pruning)
    """

    text: str
    vec: np.ndarray
    c: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_type: str = "experience"
    selection_count: int = 0
    creation_time: float = field(default_factory=time.time)
    # July 2025 compatibility: standardized confidence field (alias of c)
    confidence: Optional[float] = None

    def __post_init__(self):
        """Initialize episode with proper C-value based on type"""
        if isinstance(self.vec, np.ndarray):
            self.vec = self.vec.astype(np.float32)

        # If explicit confidence provided, use it as the primary field
        if self.confidence is not None:
            try:
                self.c = float(self.confidence)
            except Exception:
                pass

        # Set initial C-value based on episode type if not explicitly set
        if self.c == 0.5:  # Default value, auto-adjust based on type
            if self.episode_type == "experience":
                self.c = 0.5  # Experiences start at medium confidence
            elif self.episode_type == "insight":
                self.c = 0.3  # Insights start at low confidence
            elif self.episode_type == "contradiction":
                self.c = 0.4  # Contradictions need resolution
        
        # Ensure c-value is in valid range
        self.c = max(0.1, min(1.0, float(self.c)))
        # Keep alias in sync
        self.confidence = self.c

    def __repr__(self):
        """String representation"""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Episode(c={self.c:.3f}, text='{text_preview}')"

    def increment_confidence(self, amount: float = 0.1) -> None:
        """Increase confidence (max 1.0)"""
        self.c = min(1.0, self.c + amount)
        self.selection_count += 1
    
    def decay_confidence(self, amount: float = 0.05) -> None:
        """Decrease confidence (min 0.1)"""
        self.c = max(0.1, self.c - amount)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "c_value": self.c,
            "confidence": self.c,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "embedding": self.vec.tolist()
            if isinstance(self.vec, np.ndarray)
            else self.vec,
            "episode_type": self.episode_type,
            "selection_count": self.selection_count,
            "creation_time": self.creation_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create Episode from dictionary"""
        return cls(
            text=data["text"],
            vec=np.array(data["embedding"], dtype=np.float32),
            c=data.get("c_value", data.get("confidence", data.get("c", 0.5))),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            episode_type=data.get("episode_type", "experience"),
            selection_count=data.get("selection_count", 0),
            creation_time=data.get("creation_time", time.time()),
            confidence=data.get("confidence"),
        )

    # Aliases for backward compatibility and clarity
    @property
    def c_value(self) -> float:
        return self.c

    @c_value.setter
    def c_value(self, value: float) -> None:
        try:
            v = float(value)
        except Exception:
            v = self.c
        self.c = max(0.1, min(1.0, v))
        self.confidence = self.c


# Backward compatibility - allow old-style initialization
def create_episode(
    vec: np.ndarray, 
    text: str, 
    c: float = 0.5, 
    metadata: Optional[Dict] = None,
    episode_type: str = "experience"
) -> Episode:
    """Legacy function to create episode (for backward compatibility)"""
    return Episode(
        text=text, 
        vec=vec, 
        c=c, 
        metadata=metadata or {},
        episode_type=episode_type
    )
