"""
Graph-Based Importance Calculator
=================================

Replaces static C-values with dynamic graph-based importance metrics.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GraphImportanceCalculator:
    """
    Calculate node importance using various graph centrality measures.

    Features:
    - Degree centrality (number of connections)
    - Betweenness centrality (how often node appears in shortest paths)
    - PageRank-style importance propagation
    - Access frequency tracking
    - Time-decay for recency
    """

    def __init__(
        self,
        decay_factor: float = 0.1,
        alpha: float = 0.85,  # PageRank damping factor
        epsilon: float = 1e-6,  # Convergence threshold
        max_iterations: int = 100,
    ):
        self.decay_factor = decay_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        # Access tracking
        self.access_count = defaultdict(int)
        self.last_access = defaultdict(float)

        # Cache for expensive computations
        self._pagerank_cache = None
        self._betweenness_cache = None
        self._cache_timestamp = 0
        self._cache_validity = 300  # 5 minutes

    def calculate_importance(
        self, graph: Data, node_idx: int, include_access: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive importance score for a node.

        Returns:
            Dictionary with individual scores and combined importance
        """
        try:
            if graph.num_nodes == 0:
                return self._empty_importance()

            # Update access if tracking
            if include_access:
                self._update_access(node_idx)

            # Calculate individual metrics
            degree_score = self._degree_centrality(graph, node_idx)
            betweenness_score = self._betweenness_centrality(graph, node_idx)
            pagerank_score = self._pagerank_score(graph, node_idx)
            access_score = self._access_score(node_idx) if include_access else 0.0

            # Combine scores with weights
            combined_score = (
                0.3 * degree_score
                + 0.2 * betweenness_score  # 30% from connections
                + 0.3 * pagerank_score  # 20% from path importance
                + 0.2 * access_score  # 30% from PageRank  # 20% from usage patterns
            )

            return {
                "degree": degree_score,
                "betweenness": betweenness_score,
                "pagerank": pagerank_score,
                "access": access_score,
                "combined": combined_score,
            }

        except Exception as e:
            logger.error(f"Failed to calculate importance for node {node_idx}: {e}")
            return self._empty_importance()

    def _degree_centrality(self, graph: Data, node_idx: int) -> float:
        """Calculate normalized degree centrality."""
        if graph.edge_index.numel() == 0:
            return 0.0

        # Count edges for this node
        edge_mask = (graph.edge_index[0] == node_idx) | (
            graph.edge_index[1] == node_idx
        )
        degree = edge_mask.sum().item()

        # Normalize by maximum possible degree
        max_degree = graph.num_nodes - 1
        return degree / max_degree if max_degree > 0 else 0.0

    def _betweenness_centrality(self, graph: Data, node_idx: int) -> float:
        """
        Approximate betweenness centrality using sampling.
        Full calculation is expensive, so we sample paths.
        """
        # Check cache
        if self._is_cache_valid() and self._betweenness_cache is not None:
            return self._betweenness_cache.get(node_idx, 0.0)

        try:
            # Convert to adjacency list for path finding
            adj_list = self._to_adjacency_list(graph)

            if len(adj_list) < 3:  # Need at least 3 nodes for betweenness
                return 0.0

            # Sample pairs of nodes
            num_samples = min(50, graph.num_nodes * (graph.num_nodes - 1) // 2)
            sampled_paths = 0
            paths_through_node = 0

            nodes = list(range(graph.num_nodes))
            for _ in range(num_samples):
                # Random source and target
                source, target = np.random.choice(nodes, 2, replace=False)
                if source == node_idx or target == node_idx:
                    continue

                # Find shortest path (BFS)
                path = self._find_shortest_path(adj_list, source, target)
                if path and node_idx in path[1:-1]:  # Exclude endpoints
                    paths_through_node += 1
                if path:
                    sampled_paths += 1

            score = paths_through_node / sampled_paths if sampled_paths > 0 else 0.0

            # Update cache
            if self._betweenness_cache is None:
                self._betweenness_cache = {}
            self._betweenness_cache[node_idx] = score

            return score

        except Exception as e:
            logger.warning(f"Betweenness calculation failed: {e}")
            return 0.0

    def _pagerank_score(self, graph: Data, node_idx: int) -> float:
        """Calculate PageRank score for a specific node."""
        # Check cache
        if self._is_cache_valid() and self._pagerank_cache is not None:
            return self._pagerank_cache.get(node_idx, 0.0)

        try:
            # Calculate PageRank for entire graph
            pagerank = self._calculate_pagerank(graph)

            # Cache results
            self._pagerank_cache = {i: float(pagerank[i]) for i in range(len(pagerank))}
            self._cache_timestamp = time.time()

            return float(pagerank[node_idx])

        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
            return 1.0 / graph.num_nodes if graph.num_nodes > 0 else 0.0

    def _calculate_pagerank(self, graph: Data) -> np.ndarray:
        """Power iteration method for PageRank calculation."""
        n = graph.num_nodes
        if n == 0:
            return np.array([])

        # Initialize scores uniformly
        scores = np.ones(n) / n

        # Build transition matrix
        adj_list = self._to_adjacency_list(graph)

        for iteration in range(self.max_iterations):
            new_scores = np.ones(n) * (1 - self.alpha) / n

            for node in range(n):
                neighbors = adj_list.get(node, [])
                if neighbors:
                    contribution = self.alpha * scores[node] / len(neighbors)
                    for neighbor in neighbors:
                        new_scores[neighbor] += contribution

            # Check convergence
            if np.abs(new_scores - scores).max() < self.epsilon:
                break

            scores = new_scores

        return scores

    def _access_score(self, node_idx: int) -> float:
        """Calculate access-based importance with time decay."""
        if node_idx not in self.access_count:
            return 0.0

        frequency = self.access_count[node_idx]
        time_since_access = time.time() - self.last_access[node_idx]

        # Apply exponential decay
        recency_factor = np.exp(
            -self.decay_factor * time_since_access / 3600
        )  # Decay per hour

        # Normalize frequency (assume 100 accesses is very high)
        normalized_frequency = min(1.0, frequency / 100)

        return normalized_frequency * recency_factor

    def _update_access(self, node_idx: int):
        """Update access statistics for a node."""
        self.access_count[node_idx] += 1
        self.last_access[node_idx] = time.time()

    def _to_adjacency_list(self, graph: Data) -> Dict[int, List[int]]:
        """Convert edge index to adjacency list."""
        adj_list = defaultdict(list)

        if graph.edge_index.numel() > 0:
            edges = graph.edge_index.t().cpu().numpy()
            for src, dst in edges:
                adj_list[int(src)].append(int(dst))

        return adj_list

    def _find_shortest_path(
        self, adj_list: Dict[int, List[int]], source: int, target: int
    ) -> Optional[List[int]]:
        """BFS to find shortest path between nodes."""
        if source == target:
            return [source]

        visited = {source}
        queue = [(source, [source])]

        while queue:
            node, path = queue.pop(0)

            for neighbor in adj_list.get(node, []):
                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _is_cache_valid(self) -> bool:
        """Check if cached values are still valid."""
        return (time.time() - self._cache_timestamp) < self._cache_validity

    def _empty_importance(self) -> Dict[str, float]:
        """Return empty importance scores."""
        return {
            "degree": 0.0,
            "betweenness": 0.0,
            "pagerank": 0.0,
            "access": 0.0,
            "combined": 0.0,
        }

    def invalidate_cache(self):
        """Invalidate all cached computations."""
        self._pagerank_cache = None
        self._betweenness_cache = None
        self._cache_timestamp = 0

    def get_top_k_important(self, graph: Data, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k most important nodes."""
        importance_scores = []

        for node_idx in range(graph.num_nodes):
            scores = self.calculate_importance(graph, node_idx, include_access=False)
            importance_scores.append((node_idx, scores["combined"]))

        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        return importance_scores[:k]

    def update_graph_importance(self, graph: Data) -> Dict[int, float]:
        """
        Update importance for all nodes in the graph.

        Returns:
            Dictionary mapping node indices to importance scores
        """
        self.invalidate_cache()  # Force recalculation

        importance_map = {}
        for node_idx in range(graph.num_nodes):
            scores = self.calculate_importance(graph, node_idx, include_access=True)
            importance_map[node_idx] = scores["combined"]

        return importance_map


class DynamicImportanceTracker:
    """
    Track and update importance scores dynamically as the graph evolves.
    """

    def __init__(self, calculator: Optional[GraphImportanceCalculator] = None):
        self.calculator = calculator or GraphImportanceCalculator()
        self.importance_history: Dict[int, List[Tuple[float, float]]] = defaultdict(
            list
        )

    def track_importance(
        self, graph: Data, node_idx: int, timestamp: Optional[float] = None
    ):
        """Track importance score over time."""
        timestamp = timestamp or time.time()

        scores = self.calculator.calculate_importance(graph, node_idx)
        self.importance_history[node_idx].append((timestamp, scores["combined"]))

        # Keep only recent history (last 100 entries)
        if len(self.importance_history[node_idx]) > 100:
            self.importance_history[node_idx] = self.importance_history[node_idx][-100:]

    def get_importance_trend(self, node_idx: int, window: int = 10) -> float:
        """
        Calculate importance trend (positive = increasing, negative = decreasing).
        """
        history = self.importance_history.get(node_idx, [])

        if len(history) < 2:
            return 0.0

        # Get recent values
        recent = history[-window:] if len(history) >= window else history

        # Calculate linear trend
        times = np.array([t for t, _ in recent])
        values = np.array([v for _, v in recent])

        if len(times) < 2:
            return 0.0

        # Normalize time
        times = times - times[0]

        # Linear regression
        A = np.vstack([times, np.ones(len(times))]).T
        try:
            slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
            return float(slope)
        except:
            return 0.0

    def should_promote_node(self, node_idx: int, threshold: float = 0.1) -> bool:
        """Determine if a node should be promoted based on importance trend."""
        trend = self.get_importance_trend(node_idx)
        return trend > threshold

    def should_demote_node(self, node_idx: int, threshold: float = -0.1) -> bool:
        """Determine if a node should be demoted based on importance trend."""
        trend = self.get_importance_trend(node_idx)
        return trend < threshold
