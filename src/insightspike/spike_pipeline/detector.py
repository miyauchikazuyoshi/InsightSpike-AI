"""Spike decision engine.

Makes the final decision on whether a spike is detected based on analyzed statistics.
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from .analyzer import SpikeStats

logger = logging.getLogger(__name__)


class SpikeDecisionMode(Enum):
    """Different modes for spike decision making."""
    THRESHOLD = "threshold"  # Simple threshold-based
    AND = "and"             # All conditions must be met  
    OR = "or"               # Any condition triggers spike
    WEIGHTED = "weighted"   # Weighted score combination
    ADAPTIVE = "adaptive"   # Adaptive thresholds


@dataclass
class SpikeDecision:
    """Result of spike detection decision."""
    detected: bool
    confidence: float
    score: float
    reasons: List[str]
    mode: str
    thresholds_used: Dict[str, float]
    metadata: Dict[str, Any]


class SpikeDecisionEngine:
    """Makes spike detection decisions based on computed statistics."""
    
    def __init__(self, config=None, mode: SpikeDecisionMode = SpikeDecisionMode.WEIGHTED):
        self.config = config or {}
        self.mode = mode
        self.decision_history: List[SpikeDecision] = []
        self.adaptive_thresholds = self._initialize_adaptive_thresholds()
    
    def decide(self, stats: SpikeStats) -> SpikeDecision:
        """Make spike detection decision based on statistics.
        
        Args:
            stats: Computed spike statistics
            
        Returns:
            SpikeDecision with detection result
        """
        
        try:
            # Route to appropriate decision method
            if self.mode == SpikeDecisionMode.THRESHOLD:
                decision = self._threshold_decision(stats)
            elif self.mode == SpikeDecisionMode.AND:
                decision = self._and_decision(stats)
            elif self.mode == SpikeDecisionMode.OR:
                decision = self._or_decision(stats)
            elif self.mode == SpikeDecisionMode.WEIGHTED:
                decision = self._weighted_decision(stats)
            elif self.mode == SpikeDecisionMode.ADAPTIVE:
                decision = self._adaptive_decision(stats)
            else:
                decision = self._weighted_decision(stats)  # Default fallback
            
            # Update adaptive thresholds if in adaptive mode
            if self.mode == SpikeDecisionMode.ADAPTIVE:
                self._update_adaptive_thresholds(stats, decision)
            
            # Store decision in history
            self.decision_history.append(decision)
            if len(self.decision_history) > 100:  # Keep last 100 decisions
                self.decision_history.pop(0)
            
            return decision
            
        except Exception as e:
            logger.error(f"Spike decision failed: {e}")
            return self._create_fallback_decision(stats, e)
    
    def _threshold_decision(self, stats: SpikeStats) -> SpikeDecision:
        """Simple threshold-based decision."""
        
        thresholds = self._get_thresholds()
        reasons = []
        
        # Check each threshold
        ged_triggered = stats.ged_score >= thresholds['ged']
        ig_triggered = stats.ig_score >= thresholds['ig'] 
        conflict_triggered = stats.conflict_score >= thresholds['conflict']
        
        if ged_triggered:
            reasons.append(f"GED score {stats.ged_score:.3f} >= {thresholds['ged']}")
        if ig_triggered:
            reasons.append(f"IG score {stats.ig_score:.3f} >= {thresholds['ig']}")
        if conflict_triggered:
            reasons.append(f"Conflict score {stats.conflict_score:.3f} >= {thresholds['conflict']}")
        
        # Spike detected if any threshold exceeded
        detected = any([ged_triggered, ig_triggered, conflict_triggered])
        
        # Calculate overall score and confidence
        score = max(stats.ged_score, stats.ig_score, stats.conflict_score)
        confidence = stats.confidence * (1.0 if detected else 0.5)
        
        return SpikeDecision(
            detected=detected,
            confidence=confidence,
            score=score,
            reasons=reasons,
            mode="threshold",
            thresholds_used=thresholds,
            metadata={'stats_composite': stats.composite_score}
        )
    
    def _and_decision(self, stats: SpikeStats) -> SpikeDecision:
        """All conditions must be met for spike detection."""
        
        thresholds = self._get_thresholds()
        reasons = []
        
        # Check each threshold
        ged_ok = stats.ged_score >= thresholds['ged']
        ig_ok = stats.ig_score >= thresholds['ig']
        conflict_ok = stats.conflict_score >= thresholds['conflict']
        
        conditions = [
            (ged_ok, f"GED score {stats.ged_score:.3f} {'✓' if ged_ok else '✗'} {thresholds['ged']}"),
            (ig_ok, f"IG score {stats.ig_score:.3f} {'✓' if ig_ok else '✗'} {thresholds['ig']}"),
            (conflict_ok, f"Conflict score {stats.conflict_score:.3f} {'✓' if conflict_ok else '✗'} {thresholds['conflict']}")
        ]
        
        reasons = [reason for _, reason in conditions]
        
        # All conditions must be true
        detected = all(condition for condition, _ in conditions)
        
        score = stats.composite_score
        confidence = stats.confidence * (1.0 if detected else 0.2)
        
        return SpikeDecision(
            detected=detected,
            confidence=confidence,
            score=score,
            reasons=reasons,
            mode="and",
            thresholds_used=thresholds,
            metadata={'all_conditions_met': detected}
        )
    
    def _or_decision(self, stats: SpikeStats) -> SpikeDecision:
        """Any condition triggers spike detection."""
        
        thresholds = self._get_thresholds()
        reasons = []
        triggered_conditions = []
        
        # Check each threshold
        if stats.ged_score >= thresholds['ged']:
            triggered_conditions.append('ged')
            reasons.append(f"GED score {stats.ged_score:.3f} >= {thresholds['ged']}")
        
        if stats.ig_score >= thresholds['ig']:
            triggered_conditions.append('ig')
            reasons.append(f"IG score {stats.ig_score:.3f} >= {thresholds['ig']}")
        
        if stats.conflict_score >= thresholds['conflict']:
            triggered_conditions.append('conflict')
            reasons.append(f"Conflict score {stats.conflict_score:.3f} >= {thresholds['conflict']}")
        
        detected = len(triggered_conditions) > 0
        score = max(stats.ged_score, stats.ig_score, stats.conflict_score)
        confidence = stats.confidence * (1.0 if detected else 0.3)
        
        return SpikeDecision(
            detected=detected,
            confidence=confidence,
            score=score,
            reasons=reasons,
            mode="or",
            thresholds_used=thresholds,
            metadata={'triggered_conditions': triggered_conditions}
        )
    
    def _weighted_decision(self, stats: SpikeStats) -> SpikeDecision:
        """Weighted combination of scores."""
        
        thresholds = self._get_thresholds()
        
        # Use composite score as primary indicator
        composite_threshold = thresholds.get('composite', 0.6)
        
        detected = stats.composite_score >= composite_threshold
        
        reasons = [
            f"Composite score {stats.composite_score:.3f} {'✓' if detected else '✗'} {composite_threshold}",
            f"  - GED contribution: {stats.ged_score:.3f}",
            f"  - IG contribution: {stats.ig_score:.3f}", 
            f"  - Conflict contribution: {stats.conflict_score:.3f}"
        ]
        
        # Boost confidence if multiple indicators are strong
        confidence_boost = 1.0
        if stats.ged_score > 0.5 and stats.ig_score > 0.5:
            confidence_boost = 1.2
        if stats.conflict_score > 0.7:
            confidence_boost *= 1.1
        
        confidence = min(stats.confidence * confidence_boost, 1.0)
        
        return SpikeDecision(
            detected=detected,
            confidence=confidence,
            score=stats.composite_score,
            reasons=reasons,
            mode="weighted", 
            thresholds_used={'composite': composite_threshold, **thresholds},
            metadata={
                'individual_scores': {
                    'ged': stats.ged_score,
                    'ig': stats.ig_score,
                    'conflict': stats.conflict_score
                }
            }
        )
    
    def _adaptive_decision(self, stats: SpikeStats) -> SpikeDecision:
        """Adaptive thresholds based on history."""
        
        # Use current adaptive thresholds
        thresholds = self.adaptive_thresholds.copy()
        
        # Apply the weighted decision logic with adaptive thresholds
        composite_threshold = thresholds.get('composite', 0.6)
        detected = stats.composite_score >= composite_threshold
        
        reasons = [
            f"Adaptive composite score {stats.composite_score:.3f} {'✓' if detected else '✗'} {composite_threshold}",
            f"  - Adaptive learning from {len(self.decision_history)} decisions"
        ]
        
        return SpikeDecision(
            detected=detected,
            confidence=stats.confidence,
            score=stats.composite_score,
            reasons=reasons,
            mode="adaptive",
            thresholds_used=thresholds,
            metadata={
                'adaptive_adjustments': self._get_adaptive_adjustment_summary(),
                'history_length': len(self.decision_history)
            }
        )
    
    def _get_thresholds(self) -> Dict[str, float]:
        """Get spike detection thresholds from config."""
        
        default_thresholds = {
            'ged': 0.5,
            'ig': 0.4,
            'conflict': 0.6,
            'composite': 0.6
        }
        
        # Try to get from config
        try:
            if isinstance(self.config, dict):
                spike_config = self.config.get('spike', {})
                for key in default_thresholds:
                    if f'{key}_threshold' in spike_config:
                        default_thresholds[key] = spike_config[f'{key}_threshold']
            elif hasattr(self.config, 'spike'):
                spike_config = self.config.spike
                for key in default_thresholds:
                    threshold_attr = f'{key}_threshold'
                    if hasattr(spike_config, threshold_attr):
                        default_thresholds[key] = getattr(spike_config, threshold_attr)
        except Exception as e:
            logger.debug(f"Could not load thresholds from config: {e}")
        
        return default_thresholds
    
    def _initialize_adaptive_thresholds(self) -> Dict[str, float]:
        """Initialize adaptive thresholds."""
        base_thresholds = self._get_thresholds()
        
        # Start with base thresholds
        adaptive = base_thresholds.copy()
        adaptive['learning_rate'] = 0.1
        adaptive['adaptation_count'] = 0
        
        return adaptive
    
    def _update_adaptive_thresholds(self, stats: SpikeStats, decision: SpikeDecision):
        """Update adaptive thresholds based on results."""
        
        learning_rate = self.adaptive_thresholds['learning_rate']
        
        # Simple adaptive logic: adjust thresholds based on confidence
        if decision.detected and decision.confidence > 0.8:
            # High confidence detection - might lower thresholds slightly
            adjustment = -0.01
        elif not decision.detected and stats.composite_score > 0.5:
            # Missed potential spike - might lower thresholds
            adjustment = -0.005
        elif decision.detected and decision.confidence < 0.5:
            # Low confidence detection - might raise thresholds
            adjustment = 0.01
        else:
            adjustment = 0.0
        
        # Apply adjustment
        if adjustment != 0.0:
            for key in ['ged', 'ig', 'conflict', 'composite']:
                if key in self.adaptive_thresholds:
                    old_val = self.adaptive_thresholds[key]
                    new_val = old_val + adjustment * learning_rate
                    # Clamp to reasonable range
                    self.adaptive_thresholds[key] = max(0.1, min(0.9, new_val))
        
        self.adaptive_thresholds['adaptation_count'] += 1
    
    def _get_adaptive_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive adjustments."""
        
        base_thresholds = self._get_thresholds()
        adjustments = {}
        
        for key in ['ged', 'ig', 'conflict', 'composite']:
            if key in base_thresholds and key in self.adaptive_thresholds:
                base_val = base_thresholds[key]
                adaptive_val = self.adaptive_thresholds[key]
                adjustments[key] = {
                    'base': base_val,
                    'adaptive': adaptive_val,
                    'delta': adaptive_val - base_val
                }
        
        return adjustments
    
    def _create_fallback_decision(
        self, 
        stats: SpikeStats, 
        error: Exception
    ) -> SpikeDecision:
        """Create fallback decision when detection fails."""
        
        return SpikeDecision(
            detected=False,
            confidence=0.1,
            score=0.0,
            reasons=[f"Decision failed: {str(error)}"],
            mode=f"fallback_{self.mode.value}",
            thresholds_used={},
            metadata={
                'error': str(error),
                'fallback': True
            }
        )
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of recent decisions."""
        
        if not self.decision_history:
            return {'empty': True}
        
        recent = self.decision_history[-20:]  # Last 20 decisions
        
        detection_rate = sum(1 for d in recent if d.detected) / len(recent)
        avg_confidence = sum(d.confidence for d in recent) / len(recent)
        avg_score = sum(d.score for d in recent) / len(recent)
        
        return {
            'total_decisions': len(self.decision_history),
            'recent_detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'avg_score': avg_score,
            'current_mode': self.mode.value,
            'adaptive_enabled': self.mode == SpikeDecisionMode.ADAPTIVE
        }
    
    def reset_history(self):
        """Reset decision history and adaptive thresholds."""
        self.decision_history.clear()
        self.adaptive_thresholds = self._initialize_adaptive_thresholds()


__all__ = ['SpikeDecisionMode', 'SpikeDecision', 'SpikeDecisionEngine']