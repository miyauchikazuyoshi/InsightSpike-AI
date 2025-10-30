"""Core geDIG evaluation system for paper-ready experiments."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """Types of graph updates."""
    ADD = "add"
    PRUNE = "prune"
    MERGE = "merge"
    SUMMARIZE = "summarize"


@dataclass
class GraphUpdate:
    """Represents a potential graph update."""
    update_type: UpdateType
    target_nodes: List[str]
    new_node_data: Optional[Dict] = None
    new_edges: Optional[List[Tuple[str, str, Dict]]] = None
    remove_nodes: Optional[List[str]] = None
    remove_edges: Optional[List[Tuple[str, str]]] = None
    metadata: Optional[Dict] = None
    
    def get_affected_nodes(self) -> List[str]:
        """Get all nodes affected by this update."""
        affected = set(self.target_nodes)
        
        if self.new_edges:
            for source, target, _ in self.new_edges:
                affected.add(source)
                affected.add(target)
        
        if self.remove_nodes:
            affected.update(self.remove_nodes)
        
        if self.remove_edges:
            for source, target in self.remove_edges:
                affected.add(source)
                affected.add(target)
        
        return list(affected)


@dataclass
class GeDIGResult:
    """Result of geDIG evaluation."""
    delta_ged: float
    delta_ig: float
    delta_gedig: float
    confidence: float
    computation_time: float
    local_subgraph_size: int
    affected_nodes: List[str]
    detailed_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Ensure geDIG is calculated correctly."""
        # geDIG = ΔGED - k * ΔIG (k is handled by evaluator)
        pass  # delta_gedig should already be calculated by evaluator


class LocalGraphAnalyzer:
    """Analyzes local graph properties for efficient delta calculations."""
    
    def __init__(self, radius: int = 2):
        """Initialize analyzer.
        
        Args:
            radius: Radius for local subgraph extraction
        """
        self.radius = radius
        self._cache = {}  # Cache for expensive computations
    
    def extract_local_subgraph(self, graph: nx.Graph, center_nodes: List[str]) -> nx.Graph:
        """Extract local subgraph around center nodes.
        
        Args:
            graph: Full graph
            center_nodes: Nodes to center the subgraph around
            
        Returns:
            Local subgraph
        """
        if not center_nodes:
            return nx.Graph()
        
        # Get all nodes within radius using BFS
        local_nodes = set(center_nodes)
        
        for center in center_nodes:
            if center not in graph:
                continue
                
            # BFS from center node
            visited = {center}
            current_level = {center}
            
            for _ in range(self.radius):
                next_level = set()
                for node in current_level:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_level.add(neighbor)
                
                current_level = next_level
                if not current_level:
                    break
            
            local_nodes.update(visited)
        
        # Create subgraph
        subgraph = graph.subgraph(local_nodes).copy()
        return subgraph
    
    def calculate_local_metrics(self, subgraph: nx.Graph) -> Dict[str, float]:
        """Calculate structural metrics for local subgraph.
        
        Args:
            subgraph: Local subgraph to analyze
            
        Returns:
            Dictionary of structural metrics
        """
        if subgraph.number_of_nodes() == 0:
            return {
                'node_count': 0,
                'edge_count': 0,
                'avg_degree': 0,
                'clustering_coeff': 0,
                'density': 0,
                'degree_variance': 0
            }
        
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        
        # Basic metrics
        metrics = {
            'node_count': n_nodes,
            'edge_count': n_edges,
        }
        
        # Degree-based metrics
        degrees = [d for n, d in subgraph.degree()]
        if degrees:
            metrics['avg_degree'] = np.mean(degrees)
            metrics['degree_variance'] = np.var(degrees)
        else:
            metrics['avg_degree'] = 0
            metrics['degree_variance'] = 0
        
        # Clustering coefficient
        try:
            if n_nodes > 2:
                clustering = nx.average_clustering(subgraph)
                metrics['clustering_coeff'] = clustering
            else:
                metrics['clustering_coeff'] = 0
        except:
            metrics['clustering_coeff'] = 0
        
        # Graph density
        if n_nodes > 1:
            max_edges = n_nodes * (n_nodes - 1) / 2
            metrics['density'] = n_edges / max_edges if max_edges > 0 else 0
        else:
            metrics['density'] = 0
        
        return metrics


class DeltaGEDCalculator:
    """Efficient ΔGED calculator using local graph analysis."""
    
    def __init__(self):
        """Initialize calculator with structural feature weights."""
        self.structural_weights = {
            'node_count': 0.3,
            'edge_count': 0.2,
            'avg_degree': 0.15,
            'clustering_coeff': 0.15,
            'density': 0.1,
            'degree_variance': 0.1
        }
    
    def calculate(self, 
                 graph_before: nx.Graph,
                 graph_after: nx.Graph,
                 affected_nodes: List[str],
                 analyzer: LocalGraphAnalyzer) -> float:
        """Calculate ΔGED using local graph metrics.
        
        Args:
            graph_before: Graph before update
            graph_after: Graph after update
            affected_nodes: Nodes affected by the update
            analyzer: Local graph analyzer instance
            
        Returns:
            ΔGED approximation
        """
        try:
            # Extract local subgraphs
            subgraph_before = analyzer.extract_local_subgraph(graph_before, affected_nodes)
            subgraph_after = analyzer.extract_local_subgraph(graph_after, affected_nodes)
            
            # Calculate metrics
            metrics_before = analyzer.calculate_local_metrics(subgraph_before)
            metrics_after = analyzer.calculate_local_metrics(subgraph_after)
            
            # Compute weighted difference
            delta_ged = 0.0
            for metric, weight in self.structural_weights.items():
                before_val = metrics_before.get(metric, 0)
                after_val = metrics_after.get(metric, 0)
                delta_ged += weight * abs(after_val - before_val)
            
            return delta_ged
            
        except Exception as e:
            logger.warning(f"ΔGED calculation failed: {e}, using fallback")
            # Simple fallback: node/edge count difference
            node_diff = abs(graph_after.number_of_nodes() - graph_before.number_of_nodes())
            edge_diff = abs(graph_after.number_of_edges() - graph_before.number_of_edges())
            return (node_diff + edge_diff) * 0.1


class DeltaIGCalculator:
    """Efficient ΔIG calculator using heuristic approaches."""
    
    def calculate(self,
                 graph_before: nx.Graph,
                 graph_after: nx.Graph,
                 update: GraphUpdate,
                 analyzer: LocalGraphAnalyzer) -> float:
        """Calculate ΔIG using heuristic approximation.
        
        Args:
            graph_before: Graph before update
            graph_after: Graph after update
            update: Update operation being evaluated
            analyzer: Local graph analyzer instance
            
        Returns:
            ΔIG heuristic estimate
        """
        try:
            if update.update_type == UpdateType.ADD:
                return self._calculate_addition_ig(graph_before, graph_after, update)
            elif update.update_type == UpdateType.PRUNE:
                return self._calculate_removal_ig(graph_before, graph_after, update)
            elif update.update_type == UpdateType.MERGE:
                return self._calculate_merge_ig(graph_before, graph_after, update)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"ΔIG calculation failed: {e}")
            return 0.0
    
    def _calculate_addition_ig(self, graph_before: nx.Graph, graph_after: nx.Graph, update: GraphUpdate) -> float:
        """Calculate ΔIG for node addition."""
        # Heuristic: New connections indicate potential information gain
        if not update.new_edges:
            return 0.1  # Small gain for isolated new node
        
        # Information gain based on connectivity
        connection_gain = len(update.new_edges) * 0.2
        
        # Bonus for connecting previously unconnected components
        if graph_before.number_of_nodes() > 0:
            components_before = nx.number_connected_components(graph_before)
            components_after = nx.number_connected_components(graph_after)
            
            if components_after < components_before:
                connection_gain += 0.5  # Bonus for connecting components
        
        return min(connection_gain, 1.0)  # Cap at 1.0
    
    def _calculate_removal_ig(self, graph_before: nx.Graph, graph_after: nx.Graph, update: GraphUpdate) -> float:
        """Calculate ΔIG for node removal (usually negative)."""
        if not update.remove_nodes:
            return 0.0
        
        # Information loss based on removed connections
        removed_connections = 0
        for node_id in update.remove_nodes:
            if node_id in graph_before:
                removed_connections += graph_before.degree(node_id)
        
        # Negative information gain (loss)
        return -removed_connections * 0.1
    
    def _calculate_merge_ig(self, graph_before: nx.Graph, graph_after: nx.Graph, update: GraphUpdate) -> float:
        """Calculate ΔIG for node merging."""
        # Merging can reduce redundancy (positive) but lose specificity (negative)
        if len(update.target_nodes) < 2:
            return 0.0
        
        # Simple heuristic: small positive gain from reducing redundancy
        return 0.3


class GeDIGEvaluator:
    """Core geDIG evaluation system for paper experiments."""
    
    def __init__(self, k_coefficient: float = 0.5, radius: int = 2):
        """Initialize geDIG evaluator.
        
        Args:
            k_coefficient: Weight for ΔIG in geDIG formula (geDIG = ΔGED - k*ΔIG)
            radius: Radius for local subgraph analysis
        """
        self.k = k_coefficient
        self.analyzer = LocalGraphAnalyzer(radius=radius)
        self.ged_calculator = DeltaGEDCalculator()
        self.ig_calculator = DeltaIGCalculator()
        
        # Logging for paper analysis
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_update(self,
                       graph_before: nx.Graph,
                       update: GraphUpdate,
                       metadata: Optional[Dict] = None) -> GeDIGResult:
        """Evaluate a graph update using geDIG.
        
        Args:
            graph_before: Current graph state
            update: Proposed update
            metadata: Optional metadata for logging
            
        Returns:
            geDIG evaluation result
        """
        start_time = time.time()
        
        try:
            # Simulate the update to create graph_after
            graph_after = self._simulate_update(graph_before, update)
            affected_nodes = update.get_affected_nodes()
            
            # Calculate ΔGED (structural change)
            delta_ged = self.ged_calculator.calculate(
                graph_before, graph_after, affected_nodes, self.analyzer
            )
            
            # Calculate ΔIG (information gain)
            delta_ig = self.ig_calculator.calculate(
                graph_before, graph_after, update, self.analyzer
            )
            
            # Calculate geDIG
            delta_gedig = delta_ged - self.k * delta_ig
            
            # Confidence based on local subgraph size and metrics consistency
            local_subgraph = self.analyzer.extract_local_subgraph(graph_before, affected_nodes)
            confidence = min(1.0, local_subgraph.number_of_nodes() / 10.0)
            
            computation_time = time.time() - start_time
            
            result = GeDIGResult(
                delta_ged=delta_ged,
                delta_ig=delta_ig,
                delta_gedig=delta_gedig,
                confidence=confidence,
                computation_time=computation_time,
                local_subgraph_size=local_subgraph.number_of_nodes(),
                affected_nodes=affected_nodes,
                detailed_metrics={
                    'k_coefficient': self.k,
                    'update_type': update.update_type.value,
                    'n_target_nodes': len(update.target_nodes)
                }
            )
            
            # Log for paper analysis
            self._log_evaluation(result, update, metadata)
            
            return result
            
        except Exception as e:
            logger.error(f"geDIG evaluation failed: {e}")
            # Return fallback result
            return GeDIGResult(
                delta_ged=0.0,
                delta_ig=0.0,
                delta_gedig=0.0,
                confidence=0.0,
                computation_time=time.time() - start_time,
                local_subgraph_size=0,
                affected_nodes=[]
            )
    
    def _simulate_update(self, graph_before: nx.Graph, update: GraphUpdate) -> nx.Graph:
        """Simulate applying an update to create the after-state graph.
        
        Args:
            graph_before: Original graph
            update: Update to simulate
            
        Returns:
            Graph after applying the update
        """
        graph_after = graph_before.copy()
        
        # Handle removals first
        if update.remove_nodes:
            for node_id in update.remove_nodes:
                if node_id in graph_after:
                    graph_after.remove_node(node_id)
        
        if update.remove_edges:
            for source, target in update.remove_edges:
                if graph_after.has_edge(source, target):
                    graph_after.remove_edge(source, target)
        
        # Handle additions
        if update.new_node_data:
            node_id = update.new_node_data['id']
            graph_after.add_node(node_id, **update.new_node_data)
        
        if update.new_edges:
            for source, target, edge_data in update.new_edges:
                graph_after.add_edge(source, target, **edge_data)
        
        return graph_after
    
    def _log_evaluation(self, result: GeDIGResult, update: GraphUpdate, metadata: Optional[Dict]):
        """Log evaluation for paper analysis."""
        log_entry = {
            'timestamp': time.time(),
            'delta_ged': result.delta_ged,
            'delta_ig': result.delta_ig,
            'delta_gedig': result.delta_gedig,
            'confidence': result.confidence,
            'computation_time': result.computation_time,
            'update_type': update.update_type.value,
            'n_affected_nodes': len(result.affected_nodes),
            'local_subgraph_size': result.local_subgraph_size,
            'metadata': metadata
        }
        
        self.evaluation_history.append(log_entry)
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics from evaluation history for paper analysis."""
        if not self.evaluation_history:
            return {}
        
        # Aggregate statistics
        delta_geds = [entry['delta_ged'] for entry in self.evaluation_history]
        delta_igs = [entry['delta_ig'] for entry in self.evaluation_history]
        delta_gedigs = [entry['delta_gedig'] for entry in self.evaluation_history]
        computation_times = [entry['computation_time'] for entry in self.evaluation_history]
        
        return {
            'n_evaluations': len(self.evaluation_history),
            'delta_ged_stats': {
                'mean': np.mean(delta_geds),
                'std': np.std(delta_geds),
                'min': np.min(delta_geds),
                'max': np.max(delta_geds)
            },
            'delta_ig_stats': {
                'mean': np.mean(delta_igs),
                'std': np.std(delta_igs),
                'min': np.min(delta_igs),
                'max': np.max(delta_igs)
            },
            'delta_gedig_stats': {
                'mean': np.mean(delta_gedigs),
                'std': np.std(delta_gedigs),
                'min': np.min(delta_gedigs),
                'max': np.max(delta_gedigs)
            },
            'performance_stats': {
                'avg_computation_time': np.mean(computation_times),
                'total_computation_time': np.sum(computation_times)
            },
            'update_type_distribution': self._get_update_type_distribution()
        }
    
    def _get_update_type_distribution(self) -> Dict[str, int]:
        """Get distribution of update types evaluated."""
        distribution = {}
        for entry in self.evaluation_history:
            update_type = entry['update_type']
            distribution[update_type] = distribution.get(update_type, 0) + 1
        return distribution