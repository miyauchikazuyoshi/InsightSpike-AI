"""Spike data collection stage.

Collects raw data needed for spike detection from various sources.
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpikeDataCollection:
    """Raw data collected for spike detection."""
    gedig_value: float
    ged_value: float
    ig_value: float
    graph_metrics: Dict[str, Any]
    retrieved_documents: List[Dict[str, Any]]
    previous_state: Dict[str, Any] 
    current_context: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any]


class SpikeDataCollector:
    """Collects data needed for spike detection analysis."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def collect(
        self,
        gedig_result: Dict[str, Any],
        graph_analysis: Dict[str, Any], 
        retrieved_docs: List[Dict[str, Any]],
        previous_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SpikeDataCollection:
        """Collect all data needed for spike detection.
        
        Args:
            gedig_result: Result from geDIG calculation
            graph_analysis: Graph analysis results
            retrieved_docs: Retrieved documents from memory search
            previous_state: Previous agent state
            context: Current processing context
            
        Returns:
            SpikeDataCollection with all relevant data
        """
        
        try:
            # Extract core geDIG values with safe defaults
            gedig_value = gedig_result.get('gedig', 0.0)
            ged_value = gedig_result.get('ged', 0.0)  
            ig_value = gedig_result.get('ig', 0.0)
            
            # Extract graph metrics
            graph_metrics = self._extract_graph_metrics(graph_analysis)
            
            # Add timing information
            import time
            timestamp = time.time()
            
            # Collect metadata
            metadata = {
                'collection_timestamp': timestamp,
                'gedig_mode': gedig_result.get('mode', 'unknown'),
                'doc_count': len(retrieved_docs),
                'has_previous_state': bool(previous_state),
                'config_spike_thresholds': self._get_spike_thresholds()
            }
            
            return SpikeDataCollection(
                gedig_value=gedig_value,
                ged_value=ged_value,
                ig_value=ig_value,
                graph_metrics=graph_metrics,
                retrieved_documents=retrieved_docs,
                previous_state=previous_state,
                current_context=context,
                timestamp=timestamp,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            # Return minimal safe collection
            return self._create_fallback_collection(e)
    
    def _extract_graph_metrics(self, graph_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from graph analysis."""
        
        metrics = {}
        
        # Core graph structure metrics
        if 'metrics' in graph_analysis:
            source_metrics = graph_analysis['metrics']
            metrics.update({
                'node_count': source_metrics.get('node_count', 0),
                'edge_count': source_metrics.get('edge_count', 0),
                'density': source_metrics.get('density', 0.0),
                'avg_clustering': source_metrics.get('avg_clustering', 0.0),
                'delta_ged': source_metrics.get('delta_ged', 0.0),
                'delta_ig': source_metrics.get('delta_ig', 0.0)
            })
        
        # Conflict metrics
        if 'conflicts' in graph_analysis:
            conflicts = graph_analysis['conflicts']
            metrics.update({
                'conflict_total': conflicts.get('total', 0.0),
                'conflict_semantic': conflicts.get('semantic', 0.0),
                'conflict_structural': conflicts.get('structural', 0.0)
            })
        
        # Centrality measures if available
        if 'centrality' in graph_analysis:
            centrality = graph_analysis['centrality']
            metrics.update({
                'max_betweenness': max(centrality.get('betweenness', {}).values() or [0.0]),
                'max_closeness': max(centrality.get('closeness', {}).values() or [0.0]),
                'max_degree': max(centrality.get('degree', {}).values() or [0.0])
            })
        
        return metrics
    
    def _get_spike_thresholds(self) -> Dict[str, float]:
        """Get spike detection thresholds from config."""
        
        thresholds = {
            'ged': -0.5,
            'ig': 0.2, 
            'conflict': 0.3
        }
        
        # Override with config values if available
        if isinstance(self.config, dict):
            spike_config = self.config.get('spike', {})
            thresholds.update({
                'ged': spike_config.get('ged_threshold', thresholds['ged']),
                'ig': spike_config.get('ig_threshold', thresholds['ig']), 
                'conflict': spike_config.get('conflict_threshold', thresholds['conflict'])
            })
        elif hasattr(self.config, 'spike'):
            spike_config = self.config.spike
            thresholds.update({
                'ged': getattr(spike_config, 'ged_threshold', thresholds['ged']),
                'ig': getattr(spike_config, 'ig_threshold', thresholds['ig']),
                'conflict': getattr(spike_config, 'conflict_threshold', thresholds['conflict'])
            })
        
        return thresholds
    
    def _create_fallback_collection(self, error: Exception) -> SpikeDataCollection:
        """Create a minimal fallback collection when data collection fails."""
        
        import time
        
        return SpikeDataCollection(
            gedig_value=0.0,
            ged_value=0.0,
            ig_value=0.0,
            graph_metrics={},
            retrieved_documents=[],
            previous_state={},
            current_context={'error': str(error)},
            timestamp=time.time(),
            metadata={
                'fallback': True,
                'error': str(error)
            }
        )
    
    def validate_collection(self, collection: SpikeDataCollection) -> bool:
        """Validate that the collection has minimum required data."""
        
        required_checks = [
            ('gedig_value', lambda x: isinstance(x, (int, float))),
            ('ged_value', lambda x: isinstance(x, (int, float))),
            ('ig_value', lambda x: isinstance(x, (int, float))),
            ('timestamp', lambda x: x > 0),
        ]
        
        for field_name, validator in required_checks:
            field_value = getattr(collection, field_name, None)
            if not validator(field_value):
                self.logger.warning(f"Collection validation failed for {field_name}")
                return False
        
        return True


__all__ = ['SpikeDataCollection', 'SpikeDataCollector']