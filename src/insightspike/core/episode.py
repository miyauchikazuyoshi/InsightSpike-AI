"""
Episode Data Structure
=====================

Core data structure for memory episodes in InsightSpike.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
import time


@dataclass
class Episode:
    """
    Single memory entry with vector embedding, text, and C-value.

    Attributes:
        text: The text content of the episode
        vec: Vector embedding (numpy array)
        c: C-value for importance/reinforcement (0.0 to 1.0)
        timestamp: Creation timestamp
        metadata: Additional metadata dictionary
    """

    text: str
    vec: np.ndarray
    c: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure vec is float32 for FAISS compatibility"""
        if isinstance(self.vec, np.ndarray):
            self.vec = self.vec.astype(np.float32)

        # Ensure c-value is in valid range
        self.c = max(0.0, min(1.0, float(self.c)))

    def __repr__(self):
        """String representation"""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Episode(c={self.c:.3f}, text='{text_preview}')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "c_value": self.c,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "embedding": self.vec.tolist()
            if isinstance(self.vec, np.ndarray)
            else self.vec,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create Episode from dictionary"""
        return cls(
            text=data["text"],
            vec=np.array(data["embedding"], dtype=np.float32),
            c=data.get("c_value", 0.5),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


# Backward compatibility - allow old-style initialization
def create_episode(
    vec: np.ndarray, text: str, c: float = 0.5, metadata: Optional[Dict] = None
) -> Episode:
    """Legacy function to create episode (for backward compatibility)"""
    return Episode(text=text, vec=vec, c=c, metadata=metadata or {})
