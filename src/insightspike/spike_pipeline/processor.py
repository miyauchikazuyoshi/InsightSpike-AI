"""Spike post-processing stage.

Handles post-processing tasks after spike detection, such as logging,
insight registration, and result formatting.
"""

from typing import Dict, Any, List, Optional
import logging
import time
from dataclasses import dataclass

from .detector import SpikeDecision
from .analyzer import SpikeStats
from .collector import SpikeDataCollection

logger = logging.getLogger(__name__)


@dataclass
class SpikeProcessingResult:
    """Final result after spike processing."""
    decision: SpikeDecision
    insights_registered: List[str]
    processing_metadata: Dict[str, Any]
    formatted_result: Dict[str, Any]
    processing_time_ms: float


class SpikePostProcessor:
    """Post-processes spike detection results."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.processing_history: List[SpikeProcessingResult] = []
        
    def process(
        self,
        decision: SpikeDecision,
        stats: SpikeStats,
        collection: SpikeDataCollection,
        context: Optional[Dict[str, Any]] = None
    ) -> SpikeProcessingResult:
        """Process spike detection results.
        
        Args:
            decision: Spike detection decision
            stats: Computed statistics
            collection: Original data collection
            context: Additional processing context
            
        Returns:
            SpikeProcessingResult with all processing outputs
        """
        
        start_time = time.time()
        context = context or {}
        
        try:
            # Initialize results
            insights_registered = []
            processing_metadata = {
                'processing_timestamp': start_time,
                'spike_detected': decision.detected,
                'confidence': decision.confidence,
                'processing_mode': self._get_processing_mode()
            }
            
            # Register insights if spike detected
            if decision.detected:
                insights_registered = self._register_insights(
                    decision, stats, collection, context
                )
            
            # Format result for consumers
            formatted_result = self._format_result(
                decision, stats, collection, insights_registered
            )
            
            # Log spike detection
            self._log_spike_event(decision, stats, collection)
            
            # Update processing metadata
            processing_metadata.update({
                'insights_count': len(insights_registered),
                'formatted_keys': list(formatted_result.keys()),
                'data_quality_score': stats.confidence
            })
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create final result
            result = SpikeProcessingResult(
                decision=decision,
                insights_registered=insights_registered,
                processing_metadata=processing_metadata,
                formatted_result=formatted_result,
                processing_time_ms=processing_time_ms
            )
            
            # Store in history
            self.processing_history.append(result)
            if len(self.processing_history) > 50:
                self.processing_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Spike post-processing failed: {e}")
            return self._create_fallback_result(decision, stats, e, start_time)
    
    def _register_insights(
        self,
        decision: SpikeDecision,
        stats: SpikeStats, 
        collection: SpikeDataCollection,
        context: Dict[str, Any]
    ) -> List[str]:
        """Register insights when spike is detected."""
        
        registered = []
        
        try:
            # Get insight registry if available
            insight_registry = context.get('insight_registry')
            if insight_registry is None:
                logger.debug("No insight registry provided for registration")
                return registered
            
            # Extract insights from retrieved documents
            for doc in collection.retrieved_documents:
                if 'text' in doc and 'c_value' in doc:
                    # Only register high-confidence documents
                    if doc.get('c_value', 0.0) > 0.6:
                        try:
                            # Create insight with spike context
                            insight_id = self._create_insight_from_document(
                                doc, decision, stats, insight_registry
                            )
                            if insight_id:
                                registered.append(insight_id)
                        except Exception as e:
                            logger.warning(f"Failed to register insight from doc: {e}")
            
            # Create synthetic insight from spike detection
            if decision.confidence > 0.8:
                try:
                    synthetic_insight_id = self._create_synthetic_insight(
                        decision, stats, collection, insight_registry
                    )
                    if synthetic_insight_id:
                        registered.append(synthetic_insight_id)
                except Exception as e:
                    logger.warning(f"Failed to create synthetic insight: {e}")
            
        except Exception as e:
            logger.error(f"Insight registration failed: {e}")
        
        return registered
    
    def _create_insight_from_document(
        self,
        document: Dict[str, Any],
        decision: SpikeDecision,
        stats: SpikeStats,
        insight_registry
    ) -> Optional[str]:
        """Create insight from document with spike context."""
        
        # Extract key concepts from the document
        concepts = self._extract_concepts_from_text(document.get('text', ''))
        
        # Create insight metadata
        metadata = {
            'source': 'spike_detection',
            'spike_confidence': decision.confidence,
            'spike_score': decision.score,
            'gedig_value': stats.metadata.get('gedig_value', 0.0),
            'detection_mode': decision.mode,
            'original_c_value': document.get('c_value', 0.0),
            'created_timestamp': time.time()
        }
        
        try:
            # Register insight (this would depend on the actual insight registry interface)
            insight_id = f"spike_{int(time.time())}_{len(concepts)}"
            
            # Mock registration - in real implementation this would call the registry
            logger.info(f"Would register insight {insight_id} with concepts: {concepts}")
            
            return insight_id
            
        except Exception as e:
            logger.error(f"Failed to create insight from document: {e}")
            return None
    
    def _create_synthetic_insight(
        self,
        decision: SpikeDecision, 
        stats: SpikeStats,
        collection: SpikeDataCollection,
        insight_registry
    ) -> Optional[str]:
        """Create synthetic insight from spike detection pattern."""
        
        try:
            # Generate insight description based on spike characteristics
            insight_description = self._generate_insight_description(decision, stats)
            
            # Extract patterns from the spike
            patterns = self._extract_spike_patterns(decision, stats, collection)
            
            # Create synthetic insight
            synthetic_id = f"synthetic_spike_{int(time.time())}"
            
            metadata = {
                'type': 'synthetic',
                'source': 'spike_pipeline',
                'confidence': decision.confidence,
                'patterns': patterns,
                'description': insight_description,
                'created_timestamp': time.time()
            }
            
            logger.info(f"Created synthetic insight {synthetic_id}: {insight_description}")
            
            return synthetic_id
            
        except Exception as e:
            logger.error(f"Failed to create synthetic insight: {e}")
            return None
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract key concepts from text (simplified implementation)."""
        
        # Simple keyword extraction - in real implementation this would be more sophisticated
        words = text.lower().split()
        
        # Filter for meaningful concepts
        concepts = []
        for word in words:
            if len(word) > 4 and word.isalpha():
                concepts.append(word)
        
        # Return top concepts
        return list(set(concepts))[:5]
    
    def _generate_insight_description(
        self, 
        decision: SpikeDecision, 
        stats: SpikeStats
    ) -> str:
        """Generate human-readable insight description."""
        
        if stats.ged_score > stats.ig_score:
            primary_factor = "structural reorganization"
        elif stats.ig_score > stats.conflict_score:
            primary_factor = "information integration"
        else:
            primary_factor = "conflict resolution"
        
        confidence_desc = "high" if decision.confidence > 0.8 else "moderate" if decision.confidence > 0.5 else "low"
        
        return (f"Insight detected through {primary_factor} with {confidence_desc} confidence. "
                f"Composite score: {stats.composite_score:.3f}")
    
    def _extract_spike_patterns(
        self,
        decision: SpikeDecision,
        stats: SpikeStats, 
        collection: SpikeDataCollection
    ) -> Dict[str, Any]:
        """Extract patterns from spike detection."""
        
        patterns = {
            'dominant_signal': self._identify_dominant_signal(stats),
            'anomaly_strength': max(stats.anomaly_indicators.values()) if stats.anomaly_indicators else 0.0,
            'trend_direction': self._identify_trend_direction(stats.trend_indicators),
            'document_influence': len(collection.retrieved_documents),
            'decision_mode': decision.mode
        }
        
        return patterns
    
    def _identify_dominant_signal(self, stats: SpikeStats) -> str:
        """Identify the dominant signal in spike detection."""
        
        signals = {
            'ged': stats.ged_score,
            'ig': stats.ig_score,
            'conflict': stats.conflict_score
        }
        
        return max(signals, key=signals.get)
    
    def _identify_trend_direction(self, trend_indicators: Dict[str, float]) -> str:
        """Identify trend direction from indicators."""
        
        if 'gedig_trend' in trend_indicators:
            if trend_indicators['gedig_trend'] > 0.7:
                return 'increasing'
            elif trend_indicators['gedig_trend'] < 0.3:
                return 'decreasing'
        
        if 'volatility' in trend_indicators:
            if trend_indicators['volatility'] > 0.8:
                return 'volatile'
        
        return 'stable'
    
    def _format_result(
        self,
        decision: SpikeDecision,
        stats: SpikeStats,
        collection: SpikeDataCollection, 
        insights_registered: List[str]
    ) -> Dict[str, Any]:
        """Format final result for consumption."""
        
        # Core spike detection result
        formatted = {
            'spike_detected': decision.detected,
            'spike_confidence': decision.confidence,
            'spike_score': decision.score,
            'detection_mode': decision.mode,
            'reasons': decision.reasons,
        }
        
        # Add detailed statistics if spike detected
        if decision.detected:
            formatted.update({
                'gedig_analysis': {
                    'ged_score': stats.ged_score,
                    'ig_score': stats.ig_score,
                    'conflict_score': stats.conflict_score,
                    'composite_score': stats.composite_score
                },
                'anomaly_indicators': stats.anomaly_indicators,
                'trend_indicators': stats.trend_indicators,
                'insights_registered': insights_registered,
                'data_quality': stats.confidence
            })
        
        # Add processing metadata
        formatted.update({
            'processing_timestamp': time.time(),
            'thresholds_used': decision.thresholds_used,
            'graph_metrics_available': bool(collection.graph_metrics),
            'documents_analyzed': len(collection.retrieved_documents)
        })
        
        return formatted
    
    def _log_spike_event(
        self,
        decision: SpikeDecision,
        stats: SpikeStats,
        collection: SpikeDataCollection
    ):
        """Log spike detection event."""
        
        if decision.detected:
            logger.info(
                f"SPIKE DETECTED - Mode: {decision.mode}, "
                f"Confidence: {decision.confidence:.3f}, "
                f"Score: {decision.score:.3f}, "
                f"Composite: {stats.composite_score:.3f}"
            )
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Spike reasons: {decision.reasons}")
                logger.debug(f"Anomaly indicators: {stats.anomaly_indicators}")
        else:
            logger.debug(
                f"No spike - Composite: {stats.composite_score:.3f}, "
                f"Confidence: {decision.confidence:.3f}"
            )
    
    def _get_processing_mode(self) -> str:
        """Get current processing mode from config."""
        
        try:
            if isinstance(self.config, dict):
                return self.config.get('processing', {}).get('mode', 'standard')
            elif hasattr(self.config, 'processing'):
                return getattr(self.config.processing, 'mode', 'standard')
        except Exception:
            pass
        
        return 'standard'
    
    def _create_fallback_result(
        self,
        decision: SpikeDecision,
        stats: SpikeStats,
        error: Exception,
        start_time: float
    ) -> SpikeProcessingResult:
        """Create fallback result when processing fails."""
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SpikeProcessingResult(
            decision=decision,
            insights_registered=[],
            processing_metadata={
                'error': str(error),
                'fallback': True,
                'processing_timestamp': start_time
            },
            formatted_result={
                'spike_detected': False,
                'error': 'Processing failed',
                'fallback': True
            },
            processing_time_ms=processing_time_ms
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing history."""
        
        if not self.processing_history:
            return {'empty': True}
        
        recent = self.processing_history[-20:]
        
        spike_rate = sum(1 for r in recent if r.decision.detected) / len(recent)
        avg_processing_time = sum(r.processing_time_ms for r in recent) / len(recent)
        total_insights = sum(len(r.insights_registered) for r in recent)
        
        return {
            'total_processed': len(self.processing_history),
            'recent_spike_rate': spike_rate,
            'avg_processing_time_ms': avg_processing_time,
            'total_insights_registered': total_insights,
            'last_processing_time': recent[-1].processing_metadata.get('processing_timestamp')
        }
    
    def reset_history(self):
        """Reset processing history."""
        self.processing_history.clear()


__all__ = ['SpikeProcessingResult', 'SpikePostProcessor']