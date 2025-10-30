"""
Optimized Message Passing Module with Hop Limitation
===================================================

This module implements an optimized version of message passing with:
- Configurable hop limits (default 2 hops)
- Batch similarity computation
- Sparse graph maintenance
- Early stopping for convergence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import torch
import torch_geometric
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class OptimizedMessagePassing:
    """
    Optimized question-aware message passing with hop limitation.
    
    Key optimizations:
    1. Limits propagation to k-hop neighbors (default 2)
    2. Batch computes similarities
    3. Caches computation results
    4. Early stopping when converged
    """
    
    def __init__(self, 
                 alpha: float = 0.3,
                 iterations: int = 3,
                 max_hops: int = 1,
                 aggregation: str = "weighted_mean",
                 self_loop_weight: float = 0.5,
                 decay_factor: float = 0.8,
                 convergence_threshold: float = 1e-4,
                 similarity_threshold: float = 0.3):
        """
        Initialize optimized message passing module.
        
        Args:
            alpha: Weight for question influence (0-1)
            iterations: Maximum number of message passing iterations
            max_hops: Maximum hops from query-relevant nodes (default 2)
            aggregation: Aggregation method ('weighted_mean', 'max')
            self_loop_weight: Weight for self-loop in propagation
            decay_factor: Decay factor for question relevance over distance
            convergence_threshold: Threshold for early stopping
            similarity_threshold: Minimum similarity to maintain edges
        """
        self.alpha = alpha
        self.iterations = iterations
        self.max_hops = max_hops
        self.aggregation = aggregation
        self.self_loop_weight = self_loop_weight
        self.decay_factor = decay_factor
        self.convergence_threshold = convergence_threshold
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Initialized OptimizedMessagePassing with alpha={alpha}, "
                   f"iterations={iterations}, max_hops={max_hops}")
    
    def _get_k_hop_neighbors(self, 
                            adjacency: Dict[int, List[int]], 
                            start_nodes: Set[int], 
                            k: int) -> Set[int]:
        """
        Get all nodes within k hops from start nodes.
        
        Args:
            adjacency: Adjacency list representation
            start_nodes: Starting nodes
            k: Maximum number of hops
            
        Returns:
            Set of nodes within k hops
        """
        if k == 0:
            return start_nodes
        
        current_nodes = start_nodes.copy()
        all_nodes = start_nodes.copy()
        
        for hop in range(k):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(adjacency.get(node, []))
            
            # Add new nodes
            all_nodes.update(next_nodes)
            current_nodes = next_nodes - all_nodes
            
            if not current_nodes:  # No new nodes found
                break
        
        return all_nodes
    
    def _compute_batch_similarities(self, 
                                   embeddings: np.ndarray, 
                                   indices: List[int]) -> np.ndarray:
        """
        Compute similarities for a batch of nodes.
        
        Args:
            embeddings: All node embeddings
            indices: Indices to compute similarities for
            
        Returns:
            Similarity matrix for specified indices
        """
        if not indices:
            return np.array([])
        
        # Extract relevant embeddings
        batch_embeddings = embeddings[indices]
        
        # Compute all pairwise similarities at once
        similarities = cosine_similarity(batch_embeddings)
        
        return similarities
    
    def forward(self, 
                graph: torch_geometric.data.Data,
                query_vector: np.ndarray,
                node_features: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """
        Perform optimized message passing on the graph.
        
        Args:
            graph: PyTorch Geometric graph
            query_vector: Query embedding vector
            node_features: Optional custom node features
            
        Returns:
            Updated node representations after message passing
        """
        # Extract node features
        if node_features is None:
            if hasattr(graph, 'x') and graph.x is not None:
                node_embeddings = graph.x.numpy() if torch.is_tensor(graph.x) else graph.x
            else:
                raise ValueError("No node features found in graph")
        else:
            # Convert dict to array
            num_nodes = len(node_features)
            dim = len(next(iter(node_features.values())))
            node_embeddings = np.zeros((num_nodes, dim))
            for idx, vec in node_features.items():
                node_embeddings[idx] = vec
        
        num_nodes = len(node_embeddings)
        
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Step 1: Calculate initial question relevance (batch computation)
        relevance_scores = cosine_similarity(node_embeddings, query_vector).flatten()
        
        # Step 2: Find highly relevant nodes (starting points)
        relevant_threshold = np.percentile(relevance_scores, 75)  # Top 25%
        start_nodes = set(np.where(relevance_scores >= relevant_threshold)[0])
        
        logger.debug(f"Starting with {len(start_nodes)} highly relevant nodes")
        
        # Step 3: Build sparse adjacency for k-hop neighborhood
        edge_index = graph.edge_index.numpy() if torch.is_tensor(graph.edge_index) else graph.edge_index
        
        # Build full adjacency first
        full_adjacency = {i: [] for i in range(num_nodes)}
        for src, dst in edge_index.T:
            if src != dst:  # Skip self-loops for now
                full_adjacency[src].append(dst)
                full_adjacency[dst].append(src)
        
        # Get k-hop neighbors from relevant nodes
        active_nodes = self._get_k_hop_neighbors(full_adjacency, start_nodes, self.max_hops)
        active_indices = sorted(list(active_nodes))
        
        logger.debug(f"Processing {len(active_nodes)} nodes within {self.max_hops} hops")
        
        # Step 4: Build sparse adjacency for active nodes only
        sparse_adjacency = {i: [] for i in active_indices}
        for node in active_indices:
            neighbors = [n for n in full_adjacency[node] if n in active_nodes]
            sparse_adjacency[node] = neighbors
        
        # Step 5: Initialize node representations with question influence
        h = np.copy(node_embeddings)
        for i in active_indices:
            # Blend original representation with query based on relevance
            h[i] = (1 - self.alpha * relevance_scores[i]) * node_embeddings[i] + \
                   (self.alpha * relevance_scores[i]) * query_vector.flatten()
        
        # Step 6: Precompute all similarities for active nodes (batch)
        if active_indices:
            active_embeddings = h[active_indices]
            all_similarities = cosine_similarity(active_embeddings)
            
            # Create similarity lookup
            sim_lookup = {}
            for i, idx_i in enumerate(active_indices):
                for j, idx_j in enumerate(active_indices):
                    if idx_i != idx_j:
                        sim_lookup[(idx_i, idx_j)] = all_similarities[i, j]
        
        # Step 7: Message passing with early stopping
        prev_h = None
        for t in range(self.iterations):
            h_new = np.copy(h)
            
            # Only update active nodes
            for node_idx in active_indices:
                neighbors = sparse_adjacency[node_idx]
                
                if neighbors:
                    neighbor_messages = []
                    weights = []
                    
                    for neighbor_idx in neighbors:
                        # Use precomputed similarity
                        similarity = sim_lookup.get((node_idx, neighbor_idx), 0)
                        
                        # Skip weak connections
                        if similarity < self.similarity_threshold:
                            continue
                        
                        # Boost weight by neighbor's question relevance
                        weight = similarity * (1 + relevance_scores[neighbor_idx] * self.decay_factor)
                        
                        neighbor_messages.append(h[neighbor_idx])
                        weights.append(weight)
                    
                    if neighbor_messages:
                        # Aggregate messages
                        if self.aggregation == "weighted_mean":
                            weights = np.array(weights)
                            weights = weights / (weights.sum() + 1e-8)
                            aggregated = np.average(neighbor_messages, axis=0, weights=weights)
                        elif self.aggregation == "max":
                            aggregated = np.max(neighbor_messages, axis=0)
                        else:
                            aggregated = np.mean(neighbor_messages, axis=0)
                        
                        # Update with self-loop
                        h_new[node_idx] = self.self_loop_weight * h[node_idx] + \
                                         (1 - self.self_loop_weight) * aggregated
            
            # Check convergence
            if prev_h is not None:
                diff = np.linalg.norm(h_new - prev_h) / (np.linalg.norm(prev_h) + 1e-8)
                if diff < self.convergence_threshold:
                    logger.debug(f"Converged after {t+1} iterations (diff={diff:.6f})")
                    break
            
            prev_h = h.copy()
            h = h_new
            
            logger.debug(f"Message passing iteration {t+1}/{self.iterations} completed")
        
        # Convert back to dictionary format
        updated_representations = {i: h[i] for i in range(num_nodes)}
        
        return updated_representations
    
    def compute_attention_weights(self, 
                                  query_vector: np.ndarray,
                                  key_vectors: np.ndarray) -> np.ndarray:
        """
        Compute attention weights for aggregation.
        
        Args:
            query_vector: Query embedding
            key_vectors: Key embeddings
            
        Returns:
            Attention weights
        """
        # Simple dot-product attention
        scores = np.dot(key_vectors, query_vector.T).flatten()
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / exp_scores.sum()
        
        return weights