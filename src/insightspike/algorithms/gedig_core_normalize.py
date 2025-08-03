"""
Normalized geDIG Calculator
==========================

Implements conservation-based normalization for geDIG calculation.
Uses ΔGED + ΔIG as a conserved quantity to minimize arbitrariness.

Key differences from gedig_core.py:
- Two-stage normalization (size + ratio)
- Z-score transformation for IG
- Unified reward calculation (not separate GED/IG evaluation)
- Sign-based spike detection (R < 0)
"""

import logging
from typing import Dict, Any, Optional, Set, Tuple, Union
from collections import deque
import numpy as np
import networkx as nx

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    Data = None

from .gedig_core import GeDIGCore

logger = logging.getLogger(__name__)


class GeDIGNormalizedCalculator:
    """
    Normalized version of geDIG calculator with conservation-based approach.
    
    The reward function:
    R = λ * Z(ΔIG) - μ * ΔGED_normalized
    
    where:
    - ΔGED_normalized = pure_GED / (|E| + β|N|) / (ΔGED + ΔIG)
    - Z(ΔIG) = (ΔIG - μ_IG) / σ_IG
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        
        # Extract normalization config
        self.norm_config = config.get("normalization", {})
        self.enabled = self.norm_config.get("enabled", True)
        self.mode = self.norm_config.get("mode", "conservation")
        
        # Size normalization parameters
        size_norm = self.norm_config.get("size_normalization", {})
        self.beta = size_norm.get("beta", 0.5)
        
        # Z-transform parameters
        z_config = self.norm_config.get("z_transform", {})
        self.use_running_stats = z_config.get("use_running_stats", True)
        self.window_size = z_config.get("window_size", 100)
        
        # Reward function parameters
        reward_config = self.norm_config.get("reward", {})
        self.lambda_ig = reward_config.get("lambda_ig", 1.0)
        self.mu_ged = reward_config.get("mu_ged", 0.5)
        
        # Spike detection
        spike_config = self.norm_config.get("spike_detection", {})
        self.spike_mode = spike_config.get("mode", "threshold")
        self.spike_threshold = spike_config.get("threshold", 0.0)
        
        # Initialize base calculator for GED/IG computation
        # Extract parameters for GeDIGCore
        # Get metrics config for multihop settings
        graph_config = config.get('graph', {})
        metrics_config = graph_config.get('metrics', {})
        
        core_params = {
            'node_cost': config.get('node_cost', 1.0),
            'edge_cost': config.get('edge_cost', 0.5),
            'spike_threshold': self.spike_threshold,  # Use our threshold
            # Add multihop parameters
            'enable_multihop': metrics_config.get('use_multihop_gedig', False),
            'max_hops': metrics_config.get('max_hops', 2),
            'decay_factor': metrics_config.get('decay_factor', 0.5)
        }
        self._base_calculator = GeDIGCore(**core_params)
        
        # Statistics tracking for Z-transform
        self.ig_history = deque(maxlen=self.window_size)
        self.ig_mean = 0.0
        self.ig_std = 1.0
        
        logger.info(f"Initialized GeDIGNormalizedCalculator with mode: {self.mode}")
        
    def _convert_to_networkx(self, graph: Union[nx.Graph, 'Data']) -> nx.Graph:
        """Convert PyTorch Geometric Data to NetworkX if needed."""
        if isinstance(graph, nx.Graph):
            return graph
            
        if HAS_TORCH_GEOMETRIC and isinstance(graph, Data):
            # Convert PyG Data to NetworkX
            nx_graph = nx.Graph()
            
            # Add nodes
            num_nodes = graph.num_nodes
            nx_graph.add_nodes_from(range(num_nodes))
            
            # Add node features if available
            if hasattr(graph, 'x') and graph.x is not None:
                for i in range(num_nodes):
                    nx_graph.nodes[i]['features'] = graph.x[i].numpy() if hasattr(graph.x[i], 'numpy') else graph.x[i]
            
            # Add edges
            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                edge_list = graph.edge_index.t().tolist()
                nx_graph.add_edges_from(edge_list)
                
            return nx_graph
        else:
            raise TypeError(f"Unsupported graph type: {type(graph)}")
            
    def calculate(
        self,
        graph_before: Union[nx.Graph, 'Data'],
        graph_after: Union[nx.Graph, 'Data'],
        focal_nodes: Optional[Set[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate normalized geDIG metrics.
        
        Args:
            graph_before: Graph state before changes
            graph_after: Graph state after changes
            focal_nodes: Optional set of focal nodes
            context: Optional context information
            
        Returns:
            Dictionary with metrics and spike detection
        """
        
        # Convert to NetworkX if needed
        nx_before = self._convert_to_networkx(graph_before) if graph_before else None
        nx_after = self._convert_to_networkx(graph_after) if graph_after else None
        
        # If normalization is disabled, fall back to base calculator
        if not self.enabled:
            return self._base_calculator.calculate(
                nx_before, nx_after, focal_nodes, context
            )
        
        # Step 1: Calculate raw GED and IG using base calculator
        base_result = self._base_calculator.calculate(
            nx_before, nx_after, focal_nodes, context
        )
        
        # Extract raw values
        # Handle both dict and GeDIGResult object
        if hasattr(base_result, 'ged_value'):
            # GeDIGResult object
            raw_ged = base_result.ged_value
            raw_ig = base_result.ig_value
        else:
            # Dictionary
            raw_ged = base_result["ged"]
            raw_ig = base_result["ig"]
        
        # Step 2: Calculate graph sizes and determine simplification
        size_before = self._calculate_graph_size(nx_before)
        size_after = self._calculate_graph_size(nx_after)
        
        # Determine if graph simplified (insight)
        total_before = size_before["nodes"] + size_before["edges"]
        total_after = size_after["nodes"] + size_after["edges"]
        graph_simplified = total_after < total_before
        
        # Step 3: Calculate before/after metrics for conservation
        # Get before state metrics
        if nx_before is not None:
            before_result = self._base_calculator.calculate(None, nx_before, focal_nodes, context)
            ged_before = before_result.ged_value if hasattr(before_result, 'ged_value') else before_result.get("ged", 0)
            ig_before = before_result.ig_value if hasattr(before_result, 'ig_value') else before_result.get("ig", 0)
        else:
            ged_before = 0
            ig_before = 0
            
        # Normalize GED by graph size
        ged_size_normalized = self._normalize_by_size(abs(raw_ged), size_after)
        ged_before_normalized = self._normalize_by_size(abs(ged_before), size_before) if nx_before else 0
        
        # Apply conservation law with all four components
        conservation_sum = ged_before_normalized + ged_size_normalized + abs(ig_before) + abs(raw_ig)
        
        if conservation_sum > 1e-6:  # Avoid division by zero
            ged_normalized = ged_size_normalized / conservation_sum
            ig_normalized = abs(raw_ig) / conservation_sum
            # Also calculate before state normalized values for completeness
            ged_before_norm = ged_before_normalized / conservation_sum
            ig_before_norm = abs(ig_before) / conservation_sum
        else:
            ged_normalized = 0.0
            ig_normalized = 0.0
            ged_before_norm = 0.0
            ig_before_norm = 0.0
            
        # Step 4: Z-transform for IG
        ig_z_score = self._z_transform_ig(raw_ig)
        
        # Step 5: Calculate unified reward
        # Base reward calculation
        base_reward = self.lambda_ig * ig_z_score - self.mu_ged * ged_normalized
        
        # Use base reward directly without sign adjustment
        reward = base_reward
        
        # Step 6: Spike detection based on sign
        has_spike = self._detect_spike(reward)
        
        # Prepare comprehensive result
        result = {
            # Core values (backward compatibility)
            "gedig": reward,  # The normalized unified metric
            "ged": raw_ged,   # Raw GED value
            "ig": raw_ig,     # Raw IG value
            "has_spike": has_spike,
            "spike_detected": has_spike,  # Alias for L3GraphReasoner compatibility
            
            # Normalized metrics
            "normalized_metrics": {
                "ged_normalized": ged_normalized,
                "ig_z_score": ig_z_score,
                "conservation_sum": conservation_sum,
                "reward": reward,
                "ged_size_normalized": ged_size_normalized,
                "ged_before_norm": ged_before_norm if 'ged_before_norm' in locals() else 0,
                "ig_before_norm": ig_before_norm if 'ig_before_norm' in locals() else 0,
            },
            
            # Statistics
            "statistics": {
                "ig_mean": self.ig_mean,
                "ig_std": self.ig_std,
                "ig_history_size": len(self.ig_history),
                "graph_size_before": size_before,
                "graph_size_after": size_after,
            },
            
            # Include multihop results if available
            "multihop_results": getattr(base_result, "hop_results", None) if hasattr(base_result, 'hop_results') else base_result.get("multihop_results", None)
        }
        
        # Log key metrics
        logger.debug(
            f"Normalized geDIG: R={reward:.3f}, "
            f"GED_norm={ged_normalized:.3f}, IG_z={ig_z_score:.3f}, "
            f"Conservation={conservation_sum:.3f}, Spike={has_spike}"
        )
        
        return result
        
    def _calculate_graph_size(self, graph: nx.Graph) -> Dict[str, int]:
        """Calculate graph size metrics."""
        if graph is None:
            return {"nodes": 0, "edges": 0}
            
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges()
        }
        
    def _normalize_by_size(self, ged: float, graph_size: Dict[str, int]) -> float:
        """
        Normalize GED by graph size.
        
        Formula: pure_GED / (|E| + β|N|)
        """
        num_edges = graph_size["edges"]
        num_nodes = graph_size["nodes"]
        
        # Size factor with beta weighting
        size_factor = num_edges + self.beta * num_nodes
        
        if size_factor > 0:
            return abs(ged) / size_factor
        else:
            return 0.0
            
    def _z_transform_ig(self, ig: float) -> float:
        """
        Apply Z-score transformation to IG.
        
        Formula: (IG - μ) / σ
        """
        # Update history
        self.ig_history.append(ig)
        
        # Calculate running statistics if enabled
        if self.use_running_stats and len(self.ig_history) >= 2:
            self.ig_mean = np.mean(self.ig_history)
            self.ig_std = np.std(self.ig_history)
            
            # Prevent division by zero
            if self.ig_std < 1e-6:
                self.ig_std = 1.0
        
        # Apply Z-transform
        z_score = (ig - self.ig_mean) / self.ig_std
        
        return z_score
        
    def _detect_spike(self, reward: float) -> bool:
        """
        Detect spike based on reward sign.
        
        For conservation mode: spike when R < threshold (default 0)
        """
        if self.spike_mode == "threshold":
            return reward < self.spike_threshold
        else:
            # Future: implement gradient-based detection
            return reward < self.spike_threshold
            
    def reset_statistics(self):
        """Reset IG statistics (useful for new experiments)."""
        self.ig_history.clear()
        self.ig_mean = 0.0
        self.ig_std = 1.0
        logger.info("Reset IG statistics for Z-transform")
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "parameters": {
                "beta": self.beta,
                "lambda_ig": self.lambda_ig,
                "mu_ged": self.mu_ged,
                "spike_threshold": self.spike_threshold
            },
            "statistics": {
                "ig_mean": self.ig_mean,
                "ig_std": self.ig_std,
                "history_size": len(self.ig_history)
            }
        }


# Factory function for backward compatibility
def create_gedig_calculator(config: Dict[str, Any]) -> GeDIGNormalizedCalculator:
    """
    Factory function to create appropriate calculator based on config.
    
    This allows L3GraphReasoner to use either normalized or legacy version.
    """
    norm_config = config.get("normalization", {})
    
    if norm_config.get("enabled", False) and norm_config.get("mode") == "conservation":
        logger.info("Creating normalized geDIG calculator")
        return GeDIGNormalizedCalculator(config)
    else:
        logger.info("Creating legacy geDIG calculator")
        from .gedig_calculator import GeDIGCalculator
        return GeDIGCalculator(config)