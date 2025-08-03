"""
Core geDIG Implementation
========================

This is the main implementation of geDIG (Graph Edit Distance with Information Gain)
that combines all the latest improvements:
- Scale-invariant normalized GED
- Entropy variance-based information gain  
- Multi-hop analysis for different abstraction levels

This module unifies and replaces:
- normalized_ged.py
- entropy_variance_ig.py
- multihop_gedig.py
- gedig_calculator.py
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeDIGResult:
    """Complete result of geDIG calculation."""
    
    # Core metrics
    gedig_value: float
    ged_value: float
    ig_value: float
    
    # Detailed breakdowns
    structural_improvement: float
    information_integration: float
    
    # Multi-hop results (if enabled)
    hop_results: Optional[Dict[int, 'HopResult']] = None
    
    # Metadata
    computation_time: float = 0.0
    focal_nodes: Optional[Set[str]] = None
    
    @property
    def has_spike(self) -> bool:
        """Check if this represents an insight spike."""
        return self.gedig_value < -0.5  # Configurable threshold


@dataclass 
class HopResult:
    """Result for a specific hop distance."""
    hop: int
    ged: float
    ig: float
    gedig: float
    node_count: int
    edge_count: int


class GeDIGCore:
    """
    Unified geDIG calculator with all improvements.
    
    Features:
    - Normalized, scale-invariant GED calculation
    - Entropy variance-based information gain
    - Multi-hop analysis (0-hop to N-hop)
    - Configurable thresholds and weights
    """
    
    def __init__(self,
                 # GED parameters
                 node_cost: float = 1.0,
                 edge_cost: float = 1.0,
                 normalization: str = 'sum',
                 efficiency_weight: float = 0.3,
                 
                 # IG parameters
                 min_nodes: int = 2,
                 smoothing: float = 1e-10,
                 
                 # Multi-hop parameters
                 enable_multihop: bool = False,
                 max_hops: int = 3,
                 decay_factor: float = 0.7,
                 adaptive_hops: bool = True,
                 
                 # Spike detection
                 spike_threshold: float = -0.5,
                 
                 # Spectral evaluation parameters
                 enable_spectral: bool = False,
                 spectral_weight: float = 0.3):
        """
        Initialize geDIG calculator.
        
        Args:
            node_cost: Cost of adding/removing nodes
            edge_cost: Cost of adding/removing edges
            normalization: GED normalization method ('sum', 'max', 'average')
            efficiency_weight: Weight for structural efficiency in GED
            min_nodes: Minimum nodes for IG calculation
            smoothing: Smoothing factor for entropy
            enable_multihop: Enable multi-hop analysis
            max_hops: Maximum hop distance
            decay_factor: Weight decay per hop
            adaptive_hops: Stop early if no improvement
            spike_threshold: Threshold for insight spike detection
        """
        # GED parameters
        self.node_cost = node_cost
        self.edge_cost = edge_cost
        self.normalization = normalization
        self.efficiency_weight = efficiency_weight
        
        # IG parameters
        self.min_nodes = min_nodes
        self.smoothing = smoothing
        
        # Multi-hop parameters
        self.enable_multihop = enable_multihop
        self.max_hops = max_hops
        self.decay_factor = decay_factor
        self.adaptive_hops = adaptive_hops
        
        # Spike detection
        self.spike_threshold = spike_threshold
        
        # Spectral evaluation
        self.enable_spectral = enable_spectral
        self.spectral_weight = spectral_weight
        
        logger.info(f"GeDIGCore initialized: multihop={enable_multihop}, max_hops={max_hops}, spectral={enable_spectral}")
    
    def calculate(self,
                  graph_before: Any,
                  graph_after: Any,
                  features_before: Optional[np.ndarray] = None,
                  features_after: Optional[np.ndarray] = None,
                  focal_nodes: Optional[Set[str]] = None) -> GeDIGResult:
        """
        Calculate geDIG between two graph states.
        
        Args:
            graph_before: Graph state before change
            graph_after: Graph state after change
            features_before: Node features before (optional)
            features_after: Node features after (optional)
            focal_nodes: Specific nodes to focus on (for multi-hop)
            
        Returns:
            GeDIGResult with all metrics
        """
        start_time = time.time()
        
        # Convert to NetworkX
        g1 = self._ensure_networkx(graph_before)
        g2 = self._ensure_networkx(graph_after)
        
        # Extract or generate features
        if features_before is None:
            features_before = self._extract_features(g1)
        if features_after is None:
            features_after = self._extract_features(g2)
        
        # Multi-hop or single calculation
        if self.enable_multihop:
            # If no focal nodes, use all nodes that changed
            if not focal_nodes:
                # Find nodes that differ between graphs
                nodes1 = set(g1.nodes())
                nodes2 = set(g2.nodes())
                focal_nodes = (nodes1 - nodes2) | (nodes2 - nodes1) | {n for n in nodes1 & nodes2 if g1.degree(n) != g2.degree(n)}
                if not focal_nodes:
                    # If no structural changes, use random sample
                    focal_nodes = set(list(g2.nodes())[:min(5, g2.number_of_nodes())])
            
            result = self._calculate_multihop(
                g1, g2, features_before, features_after, focal_nodes
            )
        else:
            # Single calculation
            ged_result = self._calculate_normalized_ged(g1, g2)
            ig_result = self._calculate_entropy_variance_ig(
                g2, features_before, features_after
            )
            
            gedig = ged_result['structural_improvement'] + ig_result['ig_value']
            
            result = GeDIGResult(
                gedig_value=gedig,
                ged_value=ged_result['normalized_ged'],
                ig_value=ig_result['ig_value'],
                structural_improvement=ged_result['structural_improvement'],
                information_integration=ig_result['ig_value'],
                computation_time=time.time() - start_time
            )
        
        return result
    
    def _calculate_multihop(self,
                           g1: nx.Graph,
                           g2: nx.Graph, 
                           features_before: np.ndarray,
                           features_after: np.ndarray,
                           focal_nodes: Set[str]) -> GeDIGResult:
        """Calculate multi-hop geDIG."""
        hop_results = {}
        total_gedig = 0.0
        total_weight = 0.0
        
        # 各ホップで計算
        for hop in range(self.max_hops + 1):
            # Extract k-hop subgraphs
            sub_g1, nodes1 = self._extract_k_hop_subgraph(g1, focal_nodes, hop)
            sub_g2, nodes2 = self._extract_k_hop_subgraph(g2, focal_nodes, hop)
            
            # If both graphs are empty, skip
            if len(sub_g1) == 0 and len(sub_g2) == 0:
                continue
            
            # If one graph is empty, create an empty graph with same structure
            if len(sub_g1) == 0:
                sub_g1 = nx.Graph()
            if len(sub_g2) == 0:
                sub_g2 = nx.Graph()
            
            # Calculate GED and IG for subgraph
            ged_result = self._calculate_normalized_ged(sub_g1, sub_g2)
            
            # Get features for subgraph nodes
            sub_features_before = self._filter_features(features_before, nodes1, g1)
            sub_features_after = self._filter_features(features_after, nodes2, g2)
            
            ig_result = self._calculate_entropy_variance_ig(
                sub_g2, sub_features_before, sub_features_after
            )
            
            # Combine with decay
            weight = self.decay_factor ** hop
            hop_gedig = ged_result['structural_improvement'] + ig_result['ig_value']
            weighted_gedig = weight * hop_gedig
            
            hop_results[hop] = HopResult(
                hop=hop,
                ged=ged_result['normalized_ged'],
                ig=ig_result['ig_value'],
                gedig=hop_gedig,
                node_count=len(sub_g2),
                edge_count=sub_g2.number_of_edges()
            )
            
            total_gedig += weighted_gedig
            total_weight += weight
            
            # Adaptive stopping
            if self.adaptive_hops and hop > 0:
                if abs(hop_gedig) < 0.01:  # No significant change
                    logger.debug(f"Adaptive stop at hop {hop}")
                    break
        
        # Normalize by total weight
        if total_weight > 0:
            total_gedig = total_gedig / total_weight
        else:
            # If no hops produced results, fall back to full graph comparison
            ged_result = self._calculate_normalized_ged(g1, g2)
            ig_result = self._calculate_entropy_variance_ig(
                g2, features_before, features_after
            )
            total_gedig = ged_result['structural_improvement'] + ig_result['ig_value']
            
            # Add as 0-hop result
            hop_results[0] = HopResult(
                hop=0,
                ged=ged_result['normalized_ged'],
                ig=ig_result['ig_value'],
                gedig=total_gedig,
                node_count=g2.number_of_nodes(),
                edge_count=g2.number_of_edges()
            )
        
        # Get 0-hop results for main metrics
        hop0 = hop_results.get(0, None)
        
        return GeDIGResult(
            gedig_value=total_gedig,
            ged_value=hop0.ged if hop0 else 0.0,
            ig_value=hop0.ig if hop0 else 0.0,
            structural_improvement=hop0.ged if hop0 else 0.0,
            information_integration=hop0.ig if hop0 else 0.0,
            hop_results=hop_results,
            focal_nodes=focal_nodes,
            computation_time=time.time()
        )
    
    def _calculate_normalized_ged(self, g1: nx.Graph, g2: nx.Graph) -> Dict[str, float]:
        """Calculate normalized GED with structural improvement."""
        # Node and edge differences
        n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
        e1, e2 = g1.number_of_edges(), g2.number_of_edges()
        
        # Raw GED calculation
        node_ops = abs(n2 - n1) * self.node_cost
        
        # Find common edges
        common_nodes = set(g1.nodes()) & set(g2.nodes())
        common_edges = 0
        for u, v in g1.edges():
            if u in common_nodes and v in common_nodes and g2.has_edge(u, v):
                common_edges += 1
        
        edge_ops = ((e1 - common_edges) + (e2 - common_edges)) * self.edge_cost
        raw_ged = node_ops + edge_ops
        
        # Normalize
        if self.normalization == 'sum':
            norm_factor = (n1 + n2 + e1 + e2) * max(self.node_cost, self.edge_cost)
        elif self.normalization == 'max':
            norm_factor = max(n1, n2) * self.node_cost + max(e1, e2) * self.edge_cost
        else:  # average
            norm_factor = ((n1 + n2) / 2 * self.node_cost + 
                          (e1 + e2) / 2 * self.edge_cost)
        
        normalized_ged = raw_ged / norm_factor if norm_factor > 0 else 0.0
        
        # Calculate efficiency change
        eff1 = self._graph_efficiency(g1)
        eff2 = self._graph_efficiency(g2)
        efficiency_change = eff2 - eff1
        
        # Structural improvement
        if efficiency_change > 0.1:
            structural_improvement = -normalized_ged
        else:
            structural_improvement = normalized_ged
        
        structural_improvement = (
            structural_improvement * (1 - self.efficiency_weight) +
            efficiency_change * self.efficiency_weight
        )
        
        # Spectral evaluation (if enabled)
        if self.enable_spectral:
            spectral_before = self._calculate_spectral_score(g1)
            spectral_after = self._calculate_spectral_score(g2)
            
            # Improvement when structure becomes more regular (lower std dev)
            spectral_improvement = (spectral_before - spectral_after) / (spectral_before + 1e-10)
            
            # Combine with existing structural improvement
            structural_improvement = (
                structural_improvement * (1 - self.spectral_weight) +
                np.tanh(spectral_improvement) * self.spectral_weight
            )
        
        return {
            'raw_ged': raw_ged,
            'normalized_ged': normalized_ged,
            'structural_improvement': np.clip(structural_improvement, -1.0, 1.0),
            'efficiency_change': efficiency_change
        }
    
    def _calculate_entropy_variance_ig(self,
                                      graph: nx.Graph,
                                      features_before: np.ndarray,
                                      features_after: np.ndarray) -> Dict[str, float]:
        """Calculate information gain using entropy variance."""
        if graph.number_of_nodes() < self.min_nodes:
            return {'ig_value': 0.0, 'variance_reduction': 0.0}
        
        # Calculate local entropies
        entropies_before = self._calculate_local_entropies(graph, features_before)
        entropies_after = self._calculate_local_entropies(graph, features_after)
        
        # Variance as measure of information spread
        var_before = np.var(entropies_before) if len(entropies_before) > 1 else 0.0
        var_after = np.var(entropies_after) if len(entropies_after) > 1 else 0.0
        
        # IG is reduction in variance (more uniform = more integrated)
        ig_value = var_before - var_after
        
        return {
            'ig_value': ig_value,
            'variance_reduction': ig_value,
            'entropy_before': np.mean(entropies_before),
            'entropy_after': np.mean(entropies_after)
        }
    
    def _calculate_local_entropies(self,
                                  graph: nx.Graph,
                                  features: np.ndarray) -> np.ndarray:
        """Calculate Shannon entropy for each node's neighborhood."""
        entropies = []
        
        for node in graph.nodes():
            # Get node and neighbors
            neighbors = list(graph.neighbors(node))
            local_nodes = [node] + neighbors
            
            # Get features for local neighborhood
            local_features = []
            for n in local_nodes:
                # Handle both int and str node IDs
                try:
                    node_idx = int(n) if isinstance(n, str) else n
                    if node_idx < len(features):
                        local_features.append(features[node_idx])
                except (ValueError, TypeError):
                    # Skip non-numeric node IDs
                    continue
            
            if not local_features:
                continue
            
            # Calculate local entropy
            local_features = np.array(local_features)
            
            # Use feature variance as proxy for entropy
            if len(local_features) > 1:
                # Normalize features
                normalized = local_features / (np.linalg.norm(local_features, axis=1, keepdims=True) + self.smoothing)
                # Calculate pairwise similarities
                similarities = np.dot(normalized, normalized.T)
                # Convert to probabilities
                probs = (similarities + 1) / 2  # Map from [-1,1] to [0,1]
                # Flatten and normalize
                probs = probs.flatten()
                probs = probs / (probs.sum() + self.smoothing)
                # Shannon entropy
                entropy = -np.sum(probs * np.log(probs + self.smoothing))
            else:
                entropy = 0.0
            
            entropies.append(entropy)
        
        return np.array(entropies)
    
    def _graph_efficiency(self, g: nx.Graph) -> float:
        """Calculate combined graph efficiency metric."""
        if g.number_of_nodes() == 0:
            return 0.0
        
        try:
            global_eff = nx.global_efficiency(g)
        except:
            global_eff = 0.0
        
        try:
            clustering = nx.average_clustering(g)
        except:
            clustering = 0.0
        
        return 0.7 * global_eff + 0.3 * clustering
    
    def _extract_k_hop_subgraph(self,
                               graph: nx.Graph,
                               focal_nodes: Set[Any],
                               k: int) -> Tuple[nx.Graph, Set[Any]]:
        """Extract k-hop subgraph around focal nodes."""
        # Ensure focal nodes exist in graph
        valid_focal = {n for n in focal_nodes if n in graph}
        if not valid_focal:
            logger.warning(f"No focal nodes found in graph. Focal: {focal_nodes}, Graph nodes: {list(graph.nodes())[:5]}")
            return nx.Graph(), set()
        
        if k == 0:
            # Only focal nodes
            subgraph = graph.subgraph(valid_focal).copy()
            return subgraph, valid_focal
        
        # BFS to find k-hop neighbors
        all_nodes = set(valid_focal)
        current_layer = valid_focal
        
        for _ in range(k):
            next_layer = set()
            for node in current_layer:
                if node in graph:
                    next_layer.update(graph.neighbors(node))
            all_nodes.update(next_layer)
            current_layer = next_layer
        
        subgraph = graph.subgraph(all_nodes).copy()
        return subgraph, all_nodes
    
    def _ensure_networkx(self, graph: Any) -> nx.Graph:
        """Convert various graph types to NetworkX."""
        if isinstance(graph, nx.Graph):
            return graph
        
        # Handle PyG Data
        if hasattr(graph, 'edge_index') or hasattr(graph, 'x'):
            return self._pyg_to_networkx(graph)
        
        # Handle adjacency matrix
        if isinstance(graph, np.ndarray) and graph.ndim == 2:
            # CRITICAL FIX: Ensure it's a square matrix before treating as adjacency
            if graph.shape[0] == graph.shape[1]:
                return nx.from_numpy_array(graph)
            else:
                logger.warning(
                    f"Received non-square numpy array {graph.shape} where a graph "
                    f"was expected. This might be a feature matrix. Returning empty graph."
                )
                return nx.Graph()
        
        logger.warning(f"Unknown graph type: {type(graph)}")
        return nx.Graph()
    
    def _pyg_to_networkx(self, data: Any) -> nx.Graph:
        """Convert PyTorch Geometric Data to NetworkX."""
        G = nx.Graph()
        
        # Add nodes
        if hasattr(data, 'num_nodes'):
            num_nodes = data.num_nodes
        elif hasattr(data, 'x') and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            logger.warning("Cannot determine number of nodes from PyG Data")
            return G
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_array = data.edge_index
            if hasattr(edge_array, 'cpu'):
                edge_array = edge_array.cpu().numpy()
            
            # Ensure it's a numpy array
            edge_array = np.array(edge_array)

            if edge_array.ndim == 2 and edge_array.shape[0] == 2:
                edges = edge_array.T.tolist()
                G.add_edges_from(edges)
            else:
                logger.warning(f"edge_index has unexpected shape: {edge_array.shape}")

        # Add node features as attributes
        if hasattr(data, 'x') and data.x is not None:
            features = data.x
            if hasattr(features, 'cpu'):
                features = features.cpu().numpy()
            
            for i in range(num_nodes):
                if i < len(features):
                    G.nodes[i]['feature'] = features[i]
        
        return G
    
    def _extract_features(self, graph: nx.Graph) -> np.ndarray:
        """Extract or generate node features."""
        n = graph.number_of_nodes()
        
        # Try to get existing features
        features = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if 'feature' in node_data:
                features.append(node_data['feature'])
            elif 'vec' in node_data:
                features.append(node_data['vec'])
            else:
                # Generate random features as fallback
                features.append(np.random.randn(64))
        
        return np.array(features)
    
    def _filter_features(self,
                        features: np.ndarray,
                        node_set: Set[str],
                        original_graph: nx.Graph) -> np.ndarray:
        """Filter features to match node subset."""
        # Map node IDs to indices
        node_to_idx = {node: i for i, node in enumerate(original_graph.nodes())}
        
        filtered = []
        for node in sorted(node_set):
            if node in node_to_idx and node_to_idx[node] < len(features):
                filtered.append(features[node_to_idx[node]])
        
        return np.array(filtered) if filtered else np.empty((0, features.shape[1]))
    
    def _calculate_spectral_score(self, g: nx.Graph) -> float:
        """Calculate structural score using Laplacian eigenvalues.
        
        Returns:
            Standard deviation of eigenvalues (higher = more irregular structure)
        """
        if g.number_of_nodes() < 2:
            return 0.0
        
        try:
            # Calculate Laplacian matrix
            L = nx.laplacian_matrix(g).toarray()
            
            # Calculate eigenvalues (real only)
            eigvals = np.linalg.eigvalsh(L)
            
            # Use standard deviation as irregularity metric
            return np.std(eigvals)
            
        except Exception as e:
            logger.warning(f"Spectral score calculation failed: {e}")
            return 0.0


# Convenience functions for backward compatibility
def calculate_gedig(graph_before: Any,
                   graph_after: Any,
                   config: Optional[Dict[str, Any]] = None,
                   **kwargs) -> float:
    """
    Simple interface to calculate geDIG value.
    
    Returns just the geDIG score for backward compatibility.
    """
    if config:
        # Handle both old and new config formats
        metrics_config = config.get('metrics', config)
        spectral_config = metrics_config.get('spectral_evaluation', {})
        
        calculator = GeDIGCore(
            enable_multihop=metrics_config.get('use_multihop', False),
            max_hops=metrics_config.get('max_hops', 3),
            enable_spectral=spectral_config.get('enabled', False),
            spectral_weight=spectral_config.get('weight', 0.3),
            **kwargs
        )
    else:
        calculator = GeDIGCore(**kwargs)
    
    result = calculator.calculate(graph_before, graph_after)
    return result.gedig_value


def detect_insight_spike(graph_before: Any,
                        graph_after: Any,
                        threshold: float = -0.5,
                        **kwargs) -> bool:
    """
    Check if the graph change represents an insight spike.
    """
    calculator = GeDIGCore(spike_threshold=threshold, **kwargs)
    result = calculator.calculate(graph_before, graph_after)
    return result.has_spike


# Wrapper functions for backward compatibility with metrics_selector.py
def delta_ged(graph_before: Any, graph_after: Any, **kwargs) -> float:
    """
    Calculate ΔGED for backward compatibility.
    Returns negative value when graph simplifies (insight formation).
    """
    # Check if config is passed via kwargs
    config = kwargs.get('config', {})
    if config and 'metrics' in config:
        metrics_config = config['metrics']
        calculator = GeDIGCore(
            enable_multihop=metrics_config.get('use_multihop_gedig', False),
            max_hops=metrics_config.get('max_hops', 2),
            decay_factor=metrics_config.get('decay_factor', 0.5)
        )
    else:
        calculator = GeDIGCore()
    result = calculator.calculate(graph_before, graph_after)
    return result.ged_value


def delta_ig(graph_before: Any, graph_after: Any, **kwargs) -> float:
    """
    Calculate ΔIG for backward compatibility.
    Returns positive value when information gain occurs.
    """
    # Check if config is passed via kwargs
    config = kwargs.get('config', {})
    if config and 'metrics' in config:
        metrics_config = config['metrics']
        calculator = GeDIGCore(
            enable_multihop=metrics_config.get('use_multihop_gedig', False),
            max_hops=metrics_config.get('max_hops', 2),
            decay_factor=metrics_config.get('decay_factor', 0.5)
        )
    else:
        calculator = GeDIGCore()
    result = calculator.calculate(graph_before, graph_after)
    return result.ig_value
