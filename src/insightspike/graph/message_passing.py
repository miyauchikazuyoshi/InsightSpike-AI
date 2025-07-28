"""
Message Passing Module for Question-Aware Graph Enhancement
===========================================================

This module implements message passing algorithms that propagate question relevance
through the knowledge graph, updating node representations based on their neighbors
and the query context.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch_geometric
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class MessagePassing:
    """
    Question-aware message passing for knowledge graphs.
    
    This implementation propagates question relevance through the graph,
    allowing nodes to update their representations based on neighbors
    and the query context.
    """
    
    def __init__(self, 
                 alpha: float = 0.3,
                 iterations: int = 3,
                 aggregation: str = "weighted_mean",
                 self_loop_weight: float = 0.5,
                 decay_factor: float = 0.8):
        """
        Initialize message passing module.
        
        Args:
            alpha: Weight for question influence (0-1)
            iterations: Number of message passing iterations
            aggregation: Aggregation method ('weighted_mean', 'max', 'attention')
            self_loop_weight: Weight for self-loop in propagation
            decay_factor: Decay factor for question relevance over distance
        """
        self.alpha = alpha
        self.iterations = iterations
        self.aggregation = aggregation
        self.self_loop_weight = self_loop_weight
        self.decay_factor = decay_factor
        
        logger.info(f"Initialized MessagePassing with alpha={alpha}, iterations={iterations}")
    
    def forward(self, 
                graph: torch_geometric.data.Data,
                query_vector: np.ndarray,
                node_features: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """
        Perform message passing on the graph.
        
        Args:
            graph: PyTorch Geometric graph
            query_vector: Query embedding vector
            node_features: Optional custom node features (if None, uses graph.x)
            
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
        
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Calculate initial question relevance for each node
        relevance_scores = cosine_similarity(node_embeddings, query_vector).flatten()
        
        # Initialize node representations with question influence
        h = np.zeros_like(node_embeddings)
        for i in range(len(node_embeddings)):
            # Blend original representation with query based on relevance
            h[i] = (1 - self.alpha * relevance_scores[i]) * node_embeddings[i] + \
                   (self.alpha * relevance_scores[i]) * query_vector.flatten()
        
        # Convert edge_index to adjacency information
        edge_index = graph.edge_index.numpy() if torch.is_tensor(graph.edge_index) else graph.edge_index
        
        # Build adjacency lists
        adjacency = {i: [] for i in range(len(node_embeddings))}
        for src, dst in edge_index.T:
            adjacency[src].append(dst)
            adjacency[dst].append(src)  # Assuming undirected graph
        
        # Perform message passing iterations
        for t in range(self.iterations):
            h_new = np.zeros_like(h)
            
            for node_idx in range(len(h)):
                # Collect messages from neighbors
                if adjacency[node_idx]:
                    neighbor_messages = []
                    weights = []
                    
                    for neighbor_idx in adjacency[node_idx]:
                        # Calculate weight based on similarity and question relevance
                        similarity = cosine_similarity(
                            h[node_idx].reshape(1, -1),
                            h[neighbor_idx].reshape(1, -1)
                        )[0, 0]
                        
                        # Boost weight by neighbor's question relevance
                        weight = similarity * (1 + relevance_scores[neighbor_idx] * self.decay_factor)
                        
                        neighbor_messages.append(h[neighbor_idx])
                        weights.append(weight)
                    
                    # Aggregate messages
                    if self.aggregation == "weighted_mean":
                        weights = np.array(weights)
                        weights = weights / (weights.sum() + 1e-8)
                        aggregated = np.average(neighbor_messages, axis=0, weights=weights)
                    elif self.aggregation == "max":
                        # Max pooling
                        aggregated = np.max(neighbor_messages, axis=0)
                    else:  # Default to mean
                        aggregated = np.mean(neighbor_messages, axis=0)
                    
                    # Update with self-loop
                    h_new[node_idx] = self.self_loop_weight * h[node_idx] + \
                                      (1 - self.self_loop_weight) * aggregated
                else:
                    # No neighbors, keep current representation
                    h_new[node_idx] = h[node_idx]
            
            h = h_new
            
            logger.debug(f"Message passing iteration {t+1}/{self.iterations} completed")
        
        # Convert back to dictionary format
        updated_representations = {i: h[i] for i in range(len(h))}
        
        return updated_representations
    
    def compute_attention_weights(self, 
                                  query_vector: np.ndarray,
                                  key_vectors: np.ndarray) -> np.ndarray:
        """
        Compute attention weights for aggregation.
        
        Args:
            query_vector: Query representation
            key_vectors: Neighbor representations
            
        Returns:
            Attention weights
        """
        # Simple dot-product attention
        scores = np.dot(key_vectors, query_vector)
        weights = np.exp(scores) / np.sum(np.exp(scores))
        return weights