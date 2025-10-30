"""Spike statistics analysis stage.

Analyzes collected data to compute statistics needed for spike detection.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import math
from dataclasses import dataclass

from .collector import SpikeDataCollection

logger = logging.getLogger(__name__)


@dataclass 
class SpikeStats:
    """Computed statistics for spike detection."""
    ged_score: float
    ig_score: float  
    conflict_score: float
    composite_score: float
    confidence: float
    anomaly_indicators: Dict[str, float]
    trend_indicators: Dict[str, float]
    metadata: Dict[str, Any]


class SpikeStatsAnalyzer:
    """Analyzes spike data to compute detection statistics."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.history: List[SpikeDataCollection] = []
        self.max_history = 50  # Keep last N collections for trend analysis
    
    def analyze(self, collection: SpikeDataCollection) -> SpikeStats:
        """Analyze collected data to compute spike statistics.
        
        Args:
            collection: Data collected for spike analysis
            
        Returns:
            SpikeStats with computed statistics
        """
        
        try:
            # Add to history for trend analysis
            self.history.append(collection)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Compute core scores
            ged_score = self._compute_ged_score(collection)
            ig_score = self._compute_ig_score(collection)
            conflict_score = self._compute_conflict_score(collection)
            
            # Compute composite score
            composite_score = self._compute_composite_score(
                ged_score, ig_score, conflict_score
            )
            
            # Compute confidence
            confidence = self._compute_confidence(collection)
            
            # Compute anomaly indicators
            anomaly_indicators = self._compute_anomaly_indicators(collection)
            
            # Compute trend indicators
            trend_indicators = self._compute_trend_indicators()
            
            # Collect metadata
            metadata = {
                'analysis_timestamp': collection.timestamp,
                'history_length': len(self.history),
                'thresholds': collection.metadata.get('config_spike_thresholds', {}),
                'data_quality': self._assess_data_quality(collection)
            }
            
            return SpikeStats(
                ged_score=ged_score,
                ig_score=ig_score,
                conflict_score=conflict_score,
                composite_score=composite_score,
                confidence=confidence,
                anomaly_indicators=anomaly_indicators,
                trend_indicators=trend_indicators,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Spike analysis failed: {e}")
            return self._create_fallback_stats(collection, e)
    
    def _compute_ged_score(self, collection: SpikeDataCollection) -> float:
        """Compute normalized GED score for spike detection."""
        
        ged_raw = collection.ged_value
        
        # Normalize GED (negative values indicate structure simplification)
        # More negative = stronger spike signal
        if ged_raw < 0:
            ged_score = abs(ged_raw)  # Convert to positive score
        else:
            ged_score = -ged_raw * 0.5  # Penalize structure complication
        
        # Apply sigmoid normalization to [0, 1] range
        ged_score = self._sigmoid_normalize(ged_score, scale=2.0)
        
        return ged_score
    
    def _compute_ig_score(self, collection: SpikeDataCollection) -> float:
        """Compute normalized IG score for spike detection."""
        
        ig_raw = collection.ig_value
        
        # Normalize IG (positive values indicate information gain)  
        if ig_raw > 0:
            ig_score = ig_raw
        else:
            ig_score = 0.0  # No penalty for zero/negative IG
        
        # Apply logarithmic scaling for IG
        if ig_score > 0:
            ig_score = math.log(1 + ig_score)
        
        # Normalize to [0, 1] range
        ig_score = self._sigmoid_normalize(ig_score, scale=1.0)
        
        return ig_score
    
    def _compute_conflict_score(self, collection: SpikeDataCollection) -> float:
        """Compute conflict resolution score."""
        
        graph_metrics = collection.graph_metrics
        
        # Get conflict metrics with defaults
        total_conflict = graph_metrics.get('conflict_total', 0.0)
        semantic_conflict = graph_metrics.get('conflict_semantic', 0.0)
        structural_conflict = graph_metrics.get('conflict_structural', 0.0)
        
        # Weight different conflict types
        conflict_weighted = (
            0.5 * total_conflict + 
            0.3 * semantic_conflict + 
            0.2 * structural_conflict
        )
        
        # Convert to spike score (higher conflict = higher spike potential)
        conflict_score = self._sigmoid_normalize(conflict_weighted, scale=1.5)
        
        return conflict_score
    
    def _compute_composite_score(
        self, 
        ged_score: float, 
        ig_score: float, 
        conflict_score: float
    ) -> float:
        """Compute composite spike score from individual components."""
        
        # Weighted combination of scores
        weights = {
            'ged': 0.4,      # Structure change is most important
            'ig': 0.35,      # Information gain is second
            'conflict': 0.25  # Conflict resolution is supporting
        }
        
        composite = (
            weights['ged'] * ged_score + 
            weights['ig'] * ig_score +
            weights['conflict'] * conflict_score
        )
        
        return min(max(composite, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _compute_confidence(self, collection: SpikeDataCollection) -> float:
        """Compute confidence in the analysis."""
        
        confidence_factors = []
        
        # Data completeness
        doc_count = len(collection.retrieved_documents)
        if doc_count > 0:
            confidence_factors.append(min(doc_count / 5.0, 1.0))  # Normalize to 5 docs
        else:
            confidence_factors.append(0.1)  # Low confidence with no docs
        
        # Previous state availability
        if collection.previous_state:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
        
        # Graph metrics completeness 
        graph_metric_count = len(collection.graph_metrics)
        if graph_metric_count > 0:
            confidence_factors.append(min(graph_metric_count / 10.0, 1.0))
        else:
            confidence_factors.append(0.2)
        
        # Data quality
        if not collection.metadata.get('fallback', False):
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.1)
        
        # Average confidence factors
        return sum(confidence_factors) / len(confidence_factors)
    
    def _compute_anomaly_indicators(self, collection: SpikeDataCollection) -> Dict[str, float]:
        """Compute indicators of anomalous patterns."""
        
        indicators = {}
        
        # geDIG value anomaly
        gedig_abs = abs(collection.gedig_value)
        indicators['gedig_magnitude'] = self._sigmoid_normalize(gedig_abs, scale=1.0)
        
        # Graph structure anomalies
        metrics = collection.graph_metrics
        if 'density' in metrics:
            density = metrics['density']
            # Very high or very low density can be anomalous
            density_anomaly = abs(density - 0.5) * 2  # Normalize around 0.5
            indicators['density_anomaly'] = min(density_anomaly, 1.0)
        
        # Clustering anomaly
        if 'avg_clustering' in metrics:
            clustering = metrics['avg_clustering']
            indicators['clustering_extreme'] = self._sigmoid_normalize(clustering, scale=0.5)
        
        # Document retrieval anomaly
        doc_count = len(collection.retrieved_documents)
        if doc_count == 0:
            indicators['no_retrieval'] = 1.0
        elif doc_count > 20:
            indicators['excessive_retrieval'] = min((doc_count - 20) / 10.0, 1.0)
        else:
            indicators['retrieval_normal'] = 1.0
        
        return indicators
    
    def _compute_trend_indicators(self) -> Dict[str, float]:
        """Compute trend indicators from history."""
        
        indicators = {}
        
        if len(self.history) < 3:
            indicators['insufficient_history'] = 1.0
            return indicators
        
        # Extract recent geDIG values
        recent_gedigs = [c.gedig_value for c in self.history[-5:]]
        
        # Compute trend
        if len(recent_gedigs) >= 2:
            # Simple linear trend
            trend = (recent_gedigs[-1] - recent_gedigs[0]) / len(recent_gedigs)
            indicators['gedig_trend'] = self._sigmoid_normalize(abs(trend), scale=0.1)
        
        # Volatility indicator
        if len(recent_gedigs) >= 3:
            volatility = self._compute_volatility(recent_gedigs)
            indicators['volatility'] = self._sigmoid_normalize(volatility, scale=0.2)
        
        # Acceleration indicator
        if len(recent_gedigs) >= 3:
            acceleration = self._compute_acceleration(recent_gedigs)
            indicators['acceleration'] = self._sigmoid_normalize(abs(acceleration), scale=0.05)
        
        return indicators
    
    def _compute_volatility(self, values: List[float]) -> float:
        """Compute volatility of a value series."""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return math.sqrt(variance)
    
    def _compute_acceleration(self, values: List[float]) -> float:
        """Compute acceleration (second derivative) of value series."""
        if len(values) < 3:
            return 0.0
        
        # Simple second difference
        return values[-1] - 2 * values[-2] + values[-3]
    
    def _sigmoid_normalize(self, value: float, scale: float = 1.0) -> float:
        """Normalize value using sigmoid function."""
        try:
            return 1.0 / (1.0 + math.exp(-value / scale))
        except OverflowError:
            return 1.0 if value > 0 else 0.0
    
    def _assess_data_quality(self, collection: SpikeDataCollection) -> float:
        """Assess the quality of collected data."""
        
        quality_score = 1.0
        
        # Check for fallback data
        if collection.metadata.get('fallback', False):
            quality_score *= 0.2
        
        # Check for missing values
        if collection.gedig_value == 0.0 and collection.ged_value == 0.0:
            quality_score *= 0.5
        
        # Check graph metrics availability
        if not collection.graph_metrics:
            quality_score *= 0.7
        
        # Check document availability
        if not collection.retrieved_documents:
            quality_score *= 0.8
        
        return quality_score
    
    def _create_fallback_stats(
        self, 
        collection: SpikeDataCollection, 
        error: Exception
    ) -> SpikeStats:
        """Create fallback stats when analysis fails."""
        
        return SpikeStats(
            ged_score=0.0,
            ig_score=0.0,
            conflict_score=0.0,
            composite_score=0.0,
            confidence=0.1,
            anomaly_indicators={'analysis_failed': 1.0},
            trend_indicators={'insufficient_data': 1.0},
            metadata={
                'fallback': True,
                'error': str(error),
                'analysis_timestamp': collection.timestamp
            }
        )
    
    def reset_history(self):
        """Reset analysis history."""
        self.history.clear()
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of analysis history."""
        if not self.history:
            return {'empty': True}
        
        gedigs = [c.gedig_value for c in self.history]
        
        return {
            'count': len(self.history),
            'gedig_mean': sum(gedigs) / len(gedigs),
            'gedig_min': min(gedigs),
            'gedig_max': max(gedigs),
            'latest_timestamp': self.history[-1].timestamp
        }


__all__ = ['SpikeStats', 'SpikeStatsAnalyzer']