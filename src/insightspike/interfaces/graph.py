"""
Graph Interfaces
===============

Protocols for graph operations using PyTorch Geometric.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
import numpy as np
from torch_geometric.data import Data

from ..core.episode import Episode


@runtime_checkable
class IGraphBuilder(Protocol):
    """
    Interface for building PyTorch Geometric graphs.
    """
    
    def build_graph(self, episodes: List[Episode]) -> Data:
        """
        Build a PyG graph from episodes.
        
        Args:
            episodes: List of episodes
            
        Returns:
            PyTorch Geometric Data object
        """
        ...
    
    def update_graph(self, graph: Data, new_episodes: List[Episode]) -> Data:
        """
        Update existing graph with new episodes.
        
        Args:
            graph: Existing PyG graph
            new_episodes: New episodes to add
            
        Returns:
            Updated PyG graph
        """
        ...
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set similarity threshold for edge creation.
        
        Args:
            threshold: Similarity threshold
        """
        ...


@runtime_checkable
class IGraphAnalyzer(Protocol):
    """
    Interface for graph analysis and metrics.
    """
    
    def calculate_metrics(
        self,
        current_graph: Data,
        previous_graph: Optional[Data],
        delta_ged_func: Any,
        delta_ig_func: Any
    ) -> Dict[str, float]:
        """
        Calculate graph metrics including ΔGED and ΔIG.
        
        Args:
            current_graph: Current PyG graph
            previous_graph: Previous PyG graph
            delta_ged_func: GED calculation function
            delta_ig_func: IG calculation function
            
        Returns:
            Dictionary of metrics
        """
        ...
    
    def detect_spike(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> bool:
        """
        Detect if metrics indicate an insight spike.
        
        Args:
            metrics: Calculated metrics
            conflicts: Conflict scores
            thresholds: Detection thresholds
            
        Returns:
            True if spike detected
        """
        ...
    
    def assess_quality(
        self,
        metrics: Dict[str, float],
        conflicts: Dict[str, float]
    ) -> float:
        """
        Assess overall quality of reasoning.
        
        Args:
            metrics: Graph metrics
            conflicts: Conflict scores
            
        Returns:
            Quality score [0, 1]
        """
        ...