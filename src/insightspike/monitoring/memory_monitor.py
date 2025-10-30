"""
Memory Monitoring System
========================

Real-time memory usage monitoring to prevent memory explosion.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import warnings

logger = logging.getLogger(__name__)

# Try to import psutil, fallback to basic monitoring if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, using basic memory monitoring")


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    memory_mb: float
    episode_count: int
    cache_size: int
    details: Dict[str, float]


class MemoryMonitor:
    """
    Monitor memory usage and trigger warnings/actions when thresholds are exceeded.
    
    Features:
    - Real-time memory tracking
    - Threshold-based warnings
    - Automatic action triggers
    - Memory usage history
    """
    
    def __init__(
        self, 
        warning_threshold_mb: float = 2048,  # 2GB
        critical_threshold_mb: float = 4096,  # 4GB
        check_interval_seconds: float = 60
    ):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.check_interval = check_interval_seconds
        
        # History tracking
        self.history: List[MemorySnapshot] = []
        self.max_history = 100
        
        # Callbacks
        self.warning_callbacks: List[Callable] = []
        self.critical_callbacks: List[Callable] = []
        
        # State
        self.last_check = 0
        self.warnings_issued = 0
        self.critical_events = 0
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
            
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        else:
            # Fallback: rough estimation
            import sys
            return sys.getsizeof(self) / 1024 / 1024
            
    def check_memory(
        self, 
        episode_count: int = 0, 
        cache_size: int = 0,
        force: bool = False
    ) -> MemorySnapshot:
        """
        Check current memory usage and trigger actions if needed.
        
        Args:
            episode_count: Current number of episodes
            cache_size: Current cache size
            force: Force check regardless of interval
            
        Returns:
            MemorySnapshot with current status
        """
        current_time = time.time()
        
        # Check if we should run (respecting interval unless forced)
        if not force and (current_time - self.last_check) < self.check_interval:
            return None
            
        self.last_check = current_time
        
        # Get memory usage
        memory_mb = self.get_memory_usage_mb()
        
        # Create snapshot
        snapshot = MemorySnapshot(
            timestamp=current_time,
            memory_mb=memory_mb,
            episode_count=episode_count,
            cache_size=cache_size,
            details=self._get_detailed_memory() if PSUTIL_AVAILABLE else {}
        )
        
        # Add to history
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Check thresholds
        if memory_mb > self.critical_threshold_mb:
            self._handle_critical(snapshot)
        elif memory_mb > self.warning_threshold_mb:
            self._handle_warning(snapshot)
            
        return snapshot
        
    def _get_detailed_memory(self) -> Dict[str, float]:
        """Get detailed memory information"""
        if not self.process:
            return {}
            
        try:
            info = self.process.memory_info()
            return {
                'rss_mb': info.rss / 1024 / 1024,
                'vms_mb': info.vms / 1024 / 1024,
                'percent': self.process.memory_percent(),
            }
        except Exception as e:
            logger.error(f"Failed to get detailed memory: {e}")
            return {}
            
    def _handle_warning(self, snapshot: MemorySnapshot):
        """Handle warning threshold exceeded"""
        self.warnings_issued += 1
        
        logger.warning(
            f"Memory usage warning: {snapshot.memory_mb:.1f} MB "
            f"(threshold: {self.warning_threshold_mb} MB)"
        )
        
        # Trigger callbacks
        for callback in self.warning_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Warning callback failed: {e}")
                
    def _handle_critical(self, snapshot: MemorySnapshot):
        """Handle critical threshold exceeded"""
        self.critical_events += 1
        
        logger.critical(
            f"CRITICAL: Memory usage {snapshot.memory_mb:.1f} MB "
            f"exceeds critical threshold {self.critical_threshold_mb} MB!"
        )
        
        # Trigger callbacks
        for callback in self.critical_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Critical callback failed: {e}")
                
    def add_warning_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add callback for warning events"""
        self.warning_callbacks.append(callback)
        
    def add_critical_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add callback for critical events"""
        self.critical_callbacks.append(callback)
        
    def get_memory_trend(self) -> str:
        """Analyze memory usage trend"""
        if len(self.history) < 2:
            return "insufficient_data"
            
        recent = self.history[-10:]
        if len(recent) < 2:
            return "stable"
            
        # Simple linear trend
        start_mb = recent[0].memory_mb
        end_mb = recent[-1].memory_mb
        change = end_mb - start_mb
        
        if change > 50:  # 50MB increase
            return "increasing_rapidly"
        elif change > 10:
            return "increasing"
        elif change < -10:
            return "decreasing"
        else:
            return "stable"
            
    def get_stats(self) -> Dict[str, any]:
        """Get monitoring statistics"""
        if not self.history:
            return {
                'current_mb': 0,
                'warnings': self.warnings_issued,
                'critical_events': self.critical_events,
                'trend': 'no_data'
            }
            
        current = self.history[-1]
        return {
            'current_mb': current.memory_mb,
            'episode_count': current.episode_count,
            'cache_size': current.cache_size,
            'warnings': self.warnings_issued,
            'critical_events': self.critical_events,
            'trend': self.get_memory_trend(),
            'history_size': len(self.history)
        }
        
    def suggest_action(self) -> Optional[str]:
        """Suggest action based on current state"""
        if not self.history:
            return None
            
        current = self.history[-1]
        trend = self.get_memory_trend()
        
        if current.memory_mb > self.critical_threshold_mb:
            return "immediate_cleanup_required"
        elif current.memory_mb > self.warning_threshold_mb:
            if trend == "increasing_rapidly":
                return "urgent_cleanup_recommended"
            else:
                return "cleanup_recommended"
        elif trend == "increasing_rapidly":
            return "monitor_closely"
        else:
            return None


# Global monitor instance (singleton pattern)
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get or create global memory monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def check_memory_usage(episode_count: int = 0, cache_size: int = 0) -> Optional[float]:
    """Quick memory check helper"""
    monitor = get_memory_monitor()
    snapshot = monitor.check_memory(episode_count, cache_size)
    return snapshot.memory_mb if snapshot else None