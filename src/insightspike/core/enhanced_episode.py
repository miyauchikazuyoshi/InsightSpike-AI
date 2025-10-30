"""
Enhanced Episode Data Structure for geDIG
=========================================

Extension of the basic Episode to support dynamic memory management.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from collections import deque

import numpy as np

from .episode import Episode


@dataclass
class EnhancedEpisode(Episode):
    """
    Extended episode with geDIG memory management features.
    
    Minimal extension that maintains backward compatibility while
    adding essential fields for dynamic memory management.
    """
    
    # Access tracking (for LRU and importance)
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    
    # Size information (for compression decisions)
    token_count: int = 0
    byte_size: int = 0
    
    # Simple IG tracking (no complex history)
    information_gain: float = 0.0
    
    # Compression state
    is_compressed: bool = False
    compression_ratio: float = 1.0
    
    def __post_init__(self):
        """Initialize enhanced fields"""
        super().__post_init__()
        
        # Calculate byte size if not provided
        if self.byte_size == 0:
            self.byte_size = len(self.text.encode('utf-8'))
            
        # Estimate token count if not provided
        if self.token_count == 0:
            # Simple approximation: ~4 chars per token
            self.token_count = len(self.text) // 4
    
    def access(self) -> None:
        """Record an access to this episode"""
        self.access_count += 1
        self.last_access_time = time.time()
        
    def get_priority_score(self, current_time: float = None) -> float:
        """
        Calculate priority score for eviction decisions.
        Higher score = keep in memory
        """
        if current_time is None:
            current_time = time.time()
            
        # Time decay factor (exponential decay over days)
        age_days = (current_time - self.last_access_time) / 86400
        time_factor = np.exp(-age_days * 0.1)  # 10% decay per day
        
        # Access frequency factor (logarithmic)
        access_factor = np.log1p(self.access_count) / 10.0
        
        # C-value factor (importance)
        importance_factor = self.c
        
        # Information gain factor
        ig_factor = min(1.0, self.information_gain)
        
        # Combined score
        score = (
            0.3 * time_factor +
            0.2 * access_factor +
            0.3 * importance_factor +
            0.2 * ig_factor
        )
        
        return float(np.clip(score, 0.0, 1.0))
    
    def should_compress(self, size_threshold: int = 1024) -> bool:
        """Determine if this episode should be compressed"""
        if self.is_compressed:
            return False
            
        # Compress if large and not frequently accessed
        if self.byte_size > size_threshold and self.access_count < 5:
            return True
            
        # Compress if old and low priority
        age_days = (time.time() - self.timestamp) / 86400
        if age_days > 7 and self.get_priority_score() < 0.3:
            return True
            
        return False
    
    def compress(self) -> None:
        """
        Compress the episode (placeholder for actual compression).
        In practice, this might:
        - Quantize the embedding to int8
        - Compress the text
        - Remove redundant metadata
        """
        if not self.is_compressed:
            # Simulate compression by reducing precision
            self.vec = self.vec.astype(np.float16)
            self.is_compressed = True
            self.compression_ratio = 0.5  # Assume 50% size reduction
            
    def to_dict(self) -> Dict[str, Any]:
        """Enhanced serialization with new fields"""
        base_dict = super().to_dict()
        base_dict.update({
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "token_count": self.token_count,
            "byte_size": self.byte_size,
            "information_gain": self.information_gain,
            "is_compressed": self.is_compressed,
            "compression_ratio": self.compression_ratio,
        })
        return base_dict
    
    @classmethod
    def from_episode(cls, episode: Episode, **kwargs) -> "EnhancedEpisode":
        """Create EnhancedEpisode from basic Episode"""
        return cls(
            text=episode.text,
            vec=episode.vec,
            c=episode.c,
            timestamp=episode.timestamp,
            metadata=episode.metadata,
            **kwargs
        )


@dataclass
class EdgeInfo:
    """
    Edge information for graph connections.
    
    Minimal edge attributes for similarity graph.
    """
    source: int
    target: int
    weight: float = 1.0
    edge_type: str = "similarity"  # similarity, temporal, or semantic
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "edge_type": self.edge_type,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeInfo":
        """Create from dictionary"""
        return cls(**data)


class MemoryStatistics:
    """Track memory usage statistics for monitoring"""
    
    def __init__(self):
        self.total_episodes = 0
        self.compressed_episodes = 0
        self.total_bytes = 0
        self.compressed_bytes = 0
        self.eviction_count = 0
        self.access_pattern = deque(maxlen=1000)  # Recent access history
        
    def record_access(self, episode_id: int):
        """Record an episode access"""
        self.access_pattern.append({
            'episode_id': episode_id,
            'timestamp': time.time()
        })
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        compression_ratio = (
            self.compressed_bytes / max(1, self.total_bytes)
            if self.total_bytes > 0 else 0
        )
        
        return {
            'total_episodes': self.total_episodes,
            'compressed_episodes': self.compressed_episodes,
            'compression_ratio': compression_ratio,
            'eviction_count': self.eviction_count,
            'total_mb': self.total_bytes / (1024 * 1024),
            'compressed_mb': self.compressed_bytes / (1024 * 1024),
        }