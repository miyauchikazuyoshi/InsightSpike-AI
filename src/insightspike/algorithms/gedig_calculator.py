"""
Unified geDIG Calculator Interface
=================================

Provides a high-level interface for geDIG calculation.
This is now a thin wrapper around gedig_core for backward compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np

from .gedig_core import GeDIGCore, GeDIGResult

logger = logging.getLogger(__name__)


class GeDIGCalculator:
    """Unified geDIG calculator with configuration-based behavior."""
    
    def __init__(self, config=None):
        """Initialize with configuration.
        
        Args:
            config: Configuration object or dict with geDIG settings
        """
        self.config = config
        
        # Extract configuration for GeDIGCore
        config_dict = self._extract_config()
        
        # Initialize unified calculator
        self._gedig_core = GeDIGCore(**config_dict)
        
        logger.info(f"GeDIGCalculator initialized with config: {config_dict}")
    
    def _extract_config(self) -> Dict[str, Any]:
        """Extract configuration for GeDIGCore."""
        config_dict = {}
        
        if not self.config:
            return config_dict
        
        # Handle different config formats
        if isinstance(self.config, dict):
            graph_config = self.config.get('graph', {})
            metrics_config = self.config.get('metrics', {})
            multihop_config = metrics_config.get('multihop_config', {})
            spectral_config = metrics_config.get('spectral_evaluation', {})
            
            config_dict = {
                'enable_multihop': metrics_config.get('use_multihop_gedig', False),
                'normalization': graph_config.get('ged_normalization', 'sum'),
                'efficiency_weight': graph_config.get('efficiency_weight', 0.3),
                'max_hops': multihop_config.get('max_hops', 3),
                'decay_factor': multihop_config.get('decay_factor', 0.7),
                'adaptive_hops': multihop_config.get('adaptive_hops', True),
                'spike_threshold': graph_config.get('spike_threshold', -0.5),
                # Spectral evaluation settings
                'enable_spectral': spectral_config.get('enabled', False),
                'spectral_weight': spectral_config.get('weight', 0.3),
            }
        elif hasattr(self.config, 'graph') and hasattr(self.config, 'metrics'):
            graph = self.config.graph
            metrics = self.config.metrics
            multihop_config = getattr(metrics, 'multihop_config', None)
            spectral_config = getattr(metrics, 'spectral_evaluation', None)
            
            config_dict = {
                'enable_multihop': getattr(metrics, 'use_multihop_gedig', False),
                'normalization': getattr(graph, 'ged_normalization', 'sum'),
                'efficiency_weight': getattr(graph, 'efficiency_weight', 0.3),
                'spike_threshold': getattr(graph, 'spike_threshold', -0.5),
            }
            
            if multihop_config:
                config_dict.update({
                    'max_hops': getattr(multihop_config, 'max_hops', 3),
                    'decay_factor': getattr(multihop_config, 'decay_factor', 0.7),
                    'adaptive_hops': getattr(multihop_config, 'adaptive_hops', True),
                })
            
            if spectral_config:
                config_dict.update({
                    'enable_spectral': getattr(spectral_config, 'enabled', False),
                    'spectral_weight': getattr(spectral_config, 'weight', 0.3),
                })
        
        return config_dict
    
    def calculate(self, graph_before: Any, graph_after: Any,
                  features_before: Optional[np.ndarray] = None,
                  features_after: Optional[np.ndarray] = None,
                  focal_nodes: Optional[Union[List[int], Set[int]]] = None) -> Dict[str, Any]:
        """Calculate geDIG using unified core implementation.
        
        Args:
            graph_before: Graph before change
            graph_after: Graph after change
            features_before: Node features before (optional)
            features_after: Node features after (optional)
            focal_nodes: Nodes to focus on for multi-hop analysis
            
        Returns:
            Dictionary with results including:
            - gedig: Total geDIG value
            - ged: Graph edit distance component
            - ig: Information gain component
            - has_spike: Whether an insight spike was detected
            - multihop_results: Multi-hop analysis (if enabled)
        """
        # Convert focal nodes to set if provided
        if focal_nodes is not None and len(focal_nodes) > 0:
            if isinstance(focal_nodes, list):
                focal_set = set(focal_nodes)  # Keep original type
            else:
                focal_set = set(focal_nodes)
        else:
            focal_set = None
        
        # Use unified calculator
        result = self._gedig_core.calculate(
            graph_before, graph_after,
            features_before, features_after,
            focal_nodes=focal_set
        )
        
        # Convert to backward-compatible format
        output = {
            'gedig': result.gedig_value,
            'ged': result.ged_value,
            'ig': result.ig_value,
            'has_spike': result.has_spike,
            'structural_improvement': result.structural_improvement,
            'information_integration': result.information_integration
        }
        
        # Add multi-hop results if available
        if result.hop_results:
            output['multihop_results'] = {
                'total_gedig': result.gedig_value,
                'hop_details': result.hop_results,
                'optimal_hop': self._find_optimal_hop(result.hop_results)
            }
        
        return output
    
    def _find_optimal_hop(self, hop_results: Dict[int, Any]) -> int:
        """Find the hop with the strongest geDIG signal."""
        if not hop_results:
            return 0
        
        optimal_hop = 0
        min_gedig = float('inf')
        
        for hop, result in hop_results.items():
            if result.gedig < min_gedig:
                min_gedig = result.gedig
                optimal_hop = hop
        
        return optimal_hop
    
    @property
    def use_multihop(self) -> bool:
        """Check if multi-hop is enabled."""
        return self._gedig_core.enable_multihop
    
    @property
    def _multihop_calculator(self):
        """Backward compatibility property for tests."""
        # Return a mock object with expected properties
        class MockMultiHop:
            def __init__(self, core):
                self._core = core
            
            @property
            def max_hops(self):
                return self._core.max_hops
            
            @property
            def decay_factor(self):
                return self._core.decay_factor
            
            @property
            def adaptive_hops(self):
                return self._core.adaptive_hops
        
        return MockMultiHop(self._gedig_core)


# Backward compatibility functions
def calculate_gedig(graph_before: Any, graph_after: Any, 
                   config: Optional[Dict[str, Any]] = None,
                   **kwargs) -> float:
    """Simple interface for geDIG calculation.
    
    Returns just the geDIG value for backward compatibility.
    """
    calculator = GeDIGCalculator(config)
    result = calculator.calculate(graph_before, graph_after, **kwargs)
    return result['gedig']