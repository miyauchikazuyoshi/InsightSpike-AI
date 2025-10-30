"""Unified spike detection pipeline.

Orchestrates the complete spike detection process through all stages:
collect -> analyze -> decide -> process
"""

from typing import Dict, Any, List, Optional
import logging
import time

from .collector import SpikeDataCollector, SpikeDataCollection
from .analyzer import SpikeStatsAnalyzer, SpikeStats  
from .detector import SpikeDecisionEngine, SpikeDecision, SpikeDecisionMode
from .processor import SpikePostProcessor, SpikeProcessingResult

logger = logging.getLogger(__name__)


class SpikePipeline:
    """Complete spike detection pipeline."""
    
    def __init__(
        self,
        config=None,
        decision_mode: SpikeDecisionMode = SpikeDecisionMode.WEIGHTED,
        enable_history: bool = True
    ):
        self.config = config or {}
        self.enable_history = enable_history
        
        # Initialize pipeline components
        self.collector = SpikeDataCollector(config)
        self.analyzer = SpikeStatsAnalyzer(config)
        self.detector = SpikeDecisionEngine(config, decision_mode)
        self.processor = SpikePostProcessor(config)
        
        # Pipeline metrics
        self.pipeline_history: List[Dict[str, Any]] = []
        self.total_executions = 0
        self.total_spikes_detected = 0
        
        logger.info(f"SpikePipeline initialized with mode: {decision_mode.value}")
    
    def execute(
        self,
        gedig_result: Dict[str, Any],
        graph_analysis: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]], 
        previous_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SpikeProcessingResult:
        """Execute complete spike detection pipeline.
        
        Args:
            gedig_result: Result from geDIG calculation
            graph_analysis: Graph analysis results
            retrieved_docs: Retrieved documents from memory
            previous_state: Previous agent state
            context: Additional processing context
            
        Returns:
            SpikeProcessingResult with complete pipeline output
        """
        
        pipeline_start = time.time()
        context = context or {}
        stage_timings = {}
        
        try:
            self.total_executions += 1
            
            # Stage 1: Data Collection
            collect_start = time.time()
            collection = self.collector.collect(
                gedig_result=gedig_result,
                graph_analysis=graph_analysis,
                retrieved_docs=retrieved_docs,
                previous_state=previous_state,
                context=context
            )
            stage_timings['collect_ms'] = (time.time() - collect_start) * 1000
            
            if not self.collector.validate_collection(collection):
                logger.warning("Data collection validation failed")
            
            # Stage 2: Statistical Analysis
            analyze_start = time.time()
            stats = self.analyzer.analyze(collection)
            stage_timings['analyze_ms'] = (time.time() - analyze_start) * 1000
            
            # Stage 3: Spike Decision
            decide_start = time.time()
            decision = self.detector.decide(stats)
            stage_timings['decide_ms'] = (time.time() - decide_start) * 1000
            
            if decision.detected:
                self.total_spikes_detected += 1
            
            # Stage 4: Post-Processing
            process_start = time.time()
            # Add pipeline context for processor
            processor_context = context.copy()
            processor_context.update({
                'collection': collection,
                'pipeline_execution_id': self.total_executions
            })
            
            result = self.processor.process(
                decision=decision,
                stats=stats,
                collection=collection,
                context=processor_context
            )
            stage_timings['process_ms'] = (time.time() - process_start) * 1000
            
            # Record pipeline execution
            pipeline_time_ms = (time.time() - pipeline_start) * 1000
            
            if self.enable_history:
                self._record_pipeline_execution(
                    collection, stats, decision, result, 
                    stage_timings, pipeline_time_ms
                )
            
            # Log execution summary
            self._log_pipeline_execution(decision, stats, stage_timings, pipeline_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Spike pipeline execution failed: {e}")
            return self._create_fallback_pipeline_result(e, pipeline_start)
    
    def execute_lightweight(
        self,
        gedig_result: Dict[str, Any],
        basic_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute lightweight spike detection for simple cases.
        
        This bypasses full graph analysis and document retrieval
        for cases where only basic geDIG-based detection is needed.
        """
        
        try:
            # Create minimal collection
            collection = SpikeDataCollection(
                gedig_value=gedig_result.get('gedig', 0.0),
                ged_value=gedig_result.get('ged', 0.0),
                ig_value=gedig_result.get('ig', 0.0),
                graph_metrics={},
                retrieved_documents=[],
                previous_state={},
                current_context=basic_context or {},
                timestamp=time.time(),
                metadata={'lightweight': True}
            )
            
            # Quick analysis
            stats = self.analyzer.analyze(collection)
            
            # Decision
            decision = self.detector.decide(stats)
            
            # Return simplified result
            return {
                'spike_detected': decision.detected,
                'confidence': decision.confidence,
                'score': decision.score,
                'mode': 'lightweight',
                'gedig_scores': {
                    'composite': stats.composite_score,
                    'ged': stats.ged_score,
                    'ig': stats.ig_score
                }
            }
            
        except Exception as e:
            logger.error(f"Lightweight spike detection failed: {e}")
            return {
                'spike_detected': False,
                'confidence': 0.0,
                'score': 0.0,
                'mode': 'lightweight_fallback',
                'error': str(e)
            }
    
    def _record_pipeline_execution(
        self,
        collection: SpikeDataCollection,
        stats: SpikeStats,
        decision: SpikeDecision, 
        result: SpikeProcessingResult,
        stage_timings: Dict[str, float],
        total_time_ms: float
    ):
        """Record pipeline execution in history."""
        
        execution_record = {
            'execution_id': self.total_executions,
            'timestamp': collection.timestamp,
            'spike_detected': decision.detected,
            'confidence': decision.confidence,
            'composite_score': stats.composite_score,
            'decision_mode': decision.mode,
            'insights_registered': len(result.insights_registered),
            'stage_timings': stage_timings,
            'total_time_ms': total_time_ms,
            'data_quality': stats.confidence,
            'doc_count': len(collection.retrieved_documents)
        }
        
        self.pipeline_history.append(execution_record)
        
        # Keep history bounded
        if len(self.pipeline_history) > 100:
            self.pipeline_history.pop(0)
    
    def _log_pipeline_execution(
        self,
        decision: SpikeDecision,
        stats: SpikeStats,
        stage_timings: Dict[str, float],
        total_time_ms: float
    ):
        """Log pipeline execution summary."""
        
        if decision.detected:
            logger.info(
                f"Pipeline SPIKE #{self.total_spikes_detected}/{self.total_executions} - "
                f"Confidence: {decision.confidence:.3f}, "
                f"Time: {total_time_ms:.1f}ms"
            )
        else:
            logger.debug(
                f"Pipeline execution #{self.total_executions} - "
                f"No spike (score: {stats.composite_score:.3f}), "
                f"Time: {total_time_ms:.1f}ms"
            )
        
        # Debug timing breakdown
        if logger.isEnabledFor(logging.DEBUG):
            timing_str = ", ".join(f"{k}: {v:.1f}ms" for k, v in stage_timings.items())
            logger.debug(f"Stage timings - {timing_str}")
    
    def _create_fallback_pipeline_result(
        self, 
        error: Exception, 
        start_time: float
    ) -> SpikeProcessingResult:
        """Create fallback result when pipeline fails."""
        
        from .detector import SpikeDecision
        from .analyzer import SpikeStats
        
        fallback_decision = SpikeDecision(
            detected=False,
            confidence=0.0,
            score=0.0,
            reasons=[f"Pipeline failed: {str(error)}"],
            mode="pipeline_fallback",
            thresholds_used={},
            metadata={'error': str(error)}
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SpikeProcessingResult(
            decision=fallback_decision,
            insights_registered=[],
            processing_metadata={
                'error': str(error),
                'fallback': True,
                'processing_timestamp': start_time
            },
            formatted_result={
                'spike_detected': False,
                'error': 'Pipeline execution failed',
                'fallback': True
            },
            processing_time_ms=processing_time_ms
        )
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        
        if not self.pipeline_history:
            base_metrics = {
                'total_executions': self.total_executions,
                'total_spikes': self.total_spikes_detected,
                'spike_rate': 0.0,
                'history_empty': True
            }
            return base_metrics
        
        recent = self.pipeline_history[-20:]
        
        # Performance metrics
        avg_total_time = sum(r['total_time_ms'] for r in recent) / len(recent)
        avg_stage_times = {}
        if recent[0].get('stage_timings'):
            stage_keys = recent[0]['stage_timings'].keys()
            for key in stage_keys:
                avg_stage_times[f'avg_{key}'] = sum(
                    r.get('stage_timings', {}).get(key, 0) for r in recent
                ) / len(recent)
        
        # Detection metrics
        recent_spike_rate = sum(1 for r in recent if r['spike_detected']) / len(recent)
        avg_confidence = sum(r['confidence'] for r in recent) / len(recent)
        avg_score = sum(r['composite_score'] for r in recent) / len(recent)
        
        # Quality metrics
        avg_data_quality = sum(r['data_quality'] for r in recent) / len(recent)
        avg_doc_count = sum(r['doc_count'] for r in recent) / len(recent)
        
        return {
            'total_executions': self.total_executions,
            'total_spikes': self.total_spikes_detected,
            'overall_spike_rate': self.total_spikes_detected / max(self.total_executions, 1),
            'recent_spike_rate': recent_spike_rate,
            'avg_confidence': avg_confidence,
            'avg_composite_score': avg_score,
            'avg_total_time_ms': avg_total_time,
            'avg_data_quality': avg_data_quality,
            'avg_doc_count': avg_doc_count,
            **avg_stage_times,
            'history_length': len(self.pipeline_history),
            'decision_mode': self.detector.mode.value
        }
    
    def reset_pipeline(self):
        """Reset pipeline state and history."""
        
        self.collector = SpikeDataCollector(self.config)
        self.analyzer.reset_history()
        self.detector.reset_history()
        self.processor.reset_history()
        
        self.pipeline_history.clear()
        self.total_executions = 0
        self.total_spikes_detected = 0
        
        logger.info("Spike pipeline reset completed")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update pipeline configuration."""
        
        self.config = new_config
        
        # Update component configs
        self.collector.config = new_config
        self.analyzer.config = new_config
        self.detector.config = new_config
        self.processor.config = new_config
        
        logger.info("Pipeline configuration updated")
    
    def change_decision_mode(self, new_mode: SpikeDecisionMode):
        """Change spike decision mode."""
        
        old_mode = self.detector.mode
        self.detector.mode = new_mode
        self.detector.reset_history()  # Reset adaptive thresholds
        
        logger.info(f"Decision mode changed from {old_mode.value} to {new_mode.value}")
    
    def get_component_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries from all pipeline components."""
        
        return {
            'analyzer': self.analyzer.get_history_summary(),
            'detector': self.detector.get_decision_summary(), 
            'processor': self.processor.get_processing_summary()
        }


# Convenience factory functions
def create_standard_pipeline(config=None) -> SpikePipeline:
    """Create pipeline with standard weighted decision mode."""
    return SpikePipeline(config, SpikeDecisionMode.WEIGHTED)


def create_adaptive_pipeline(config=None) -> SpikePipeline:
    """Create pipeline with adaptive decision mode."""
    return SpikePipeline(config, SpikeDecisionMode.ADAPTIVE)


def create_threshold_pipeline(config=None) -> SpikePipeline:
    """Create pipeline with simple threshold decision mode."""
    return SpikePipeline(config, SpikeDecisionMode.THRESHOLD)


__all__ = [
    'SpikePipeline',
    'create_standard_pipeline',
    'create_adaptive_pipeline', 
    'create_threshold_pipeline'
]