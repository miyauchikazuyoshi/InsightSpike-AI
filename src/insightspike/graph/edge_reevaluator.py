"""
Edge Re-evaluation Module
=========================

This module re-evaluates graph edges after message passing, discovering new
connections and adjusting edge weights based on updated node representations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import torch
import torch_geometric
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class EdgeReevaluator:
    """
    Re-evaluates edges after message passing to discover new connections
    and adjust existing edge weights.
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.7,
                 new_edge_threshold: float = 0.8,
                 max_new_edges_per_node: int = 5,
                 edge_decay_factor: float = 0.9):
        """
        Initialize edge re-evaluator.
        
        Args:
            similarity_threshold: Minimum similarity for edge retention
            new_edge_threshold: Minimum similarity for new edge creation
            max_new_edges_per_node: Maximum new edges to add per node
            edge_decay_factor: Factor to decay original edge weights
        """
        self.similarity_threshold = similarity_threshold
        self.new_edge_threshold = new_edge_threshold
        self.max_new_edges_per_node = max_new_edges_per_node
        self.edge_decay_factor = edge_decay_factor
        
        logger.info(f"Initialized EdgeReevaluator with thresholds: {similarity_threshold}, {new_edge_threshold}")
    
    def reevaluate(self,
                   graph: torch_geometric.data.Data,
                   updated_representations: Dict[int, np.ndarray],
                   query_vector: np.ndarray,
                   return_edge_scores: bool = False) -> torch_geometric.data.Data:
        """
        Re-evaluate edges based on updated node representations.
        
        Args:
            graph: Original graph
            updated_representations: Node representations after message passing
            query_vector: Query embedding for relevance scoring
            return_edge_scores: Whether to return detailed edge scores
            
        Returns:
            New graph with re-evaluated edges
        """
        num_nodes = len(updated_representations)
        
        # Convert representations to matrix
        node_matrix = np.array([updated_representations[i] for i in range(num_nodes)])
        
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Calculate query relevance for each node
        query_relevance = cosine_similarity(node_matrix, query_vector).flatten()
        
        # Get existing edges
        edge_index = graph.edge_index.numpy() if torch.is_tensor(graph.edge_index) else graph.edge_index
        existing_edges = set()
        for src, dst in edge_index.T:
            existing_edges.add((min(src, dst), max(src, dst)))  # Undirected
        
        # Re-evaluate existing edges
        new_edges = []
        edge_weights = []
        edge_info = []
        
        # First, re-evaluate existing edges
        for src, dst in edge_index.T:
            # Calculate new similarity based on updated representations
            sim = cosine_similarity(
                node_matrix[src].reshape(1, -1),
                node_matrix[dst].reshape(1, -1)
            )[0, 0]
            
            # Boost similarity based on query relevance
            relevance_boost = (query_relevance[src] + query_relevance[dst]) / 2
            adjusted_sim = sim * (1 + 0.2 * relevance_boost)
            
            # Keep edge if above threshold
            if adjusted_sim >= self.similarity_threshold:
                new_edges.append([src, dst])
                # Blend old weight (if exists) with new
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    old_weight = graph.edge_attr[len(new_edges)-1].item()
                    weight = self.edge_decay_factor * old_weight + (1 - self.edge_decay_factor) * adjusted_sim
                else:
                    weight = adjusted_sim
                edge_weights.append(weight)
                
                edge_info.append({
                    'src': src,
                    'dst': dst,
                    'type': 'existing',
                    'similarity': sim,
                    'adjusted_similarity': adjusted_sim,
                    'weight': weight
                })
        
        logger.info(f"Retained {len(new_edges)} existing edges after re-evaluation")
        
        # Discover new edges
        similarity_matrix = cosine_similarity(node_matrix)
        
        # Find potential new edges
        new_edge_candidates = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (i, j) not in existing_edges:
                    sim = similarity_matrix[i, j]
                    
                    # Check if similarity after message passing is high enough
                    if sim >= self.new_edge_threshold:
                        # Additional boost for query-relevant pairs
                        relevance_boost = (query_relevance[i] + query_relevance[j]) / 2
                        adjusted_sim = sim * (1 + 0.3 * relevance_boost)
                        
                        new_edge_candidates.append({
                            'src': i,
                            'dst': j,
                            'similarity': sim,
                            'adjusted_similarity': adjusted_sim,
                            'query_relevance': relevance_boost
                        })
        
        # Sort candidates by adjusted similarity
        new_edge_candidates.sort(key=lambda x: x['adjusted_similarity'], reverse=True)
        
        # Add top new edges (with per-node limit)
        node_new_edge_count = {i: 0 for i in range(num_nodes)}
        added_new_edges = 0
        
        for candidate in new_edge_candidates:
            src, dst = candidate['src'], candidate['dst']
            
            # Check per-node limits
            if (node_new_edge_count[src] < self.max_new_edges_per_node and
                node_new_edge_count[dst] < self.max_new_edges_per_node):
                
                # Add bidirectional edges
                new_edges.append([src, dst])
                new_edges.append([dst, src])
                edge_weights.extend([candidate['adjusted_similarity']] * 2)
                
                node_new_edge_count[src] += 1
                node_new_edge_count[dst] += 1
                added_new_edges += 1
                
                edge_info.append({
                    'src': src,
                    'dst': dst,
                    'type': 'new',
                    'similarity': candidate['similarity'],
                    'adjusted_similarity': candidate['adjusted_similarity'],
                    'weight': candidate['adjusted_similarity']
                })
        
        logger.info(f"Added {added_new_edges} new edges after message passing")
        
        # Create new graph
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
        new_edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # Copy node features
        new_graph = torch_geometric.data.Data(
            x=torch.tensor(node_matrix, dtype=torch.float),
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            num_nodes=num_nodes
        )
        
        # Add metadata
        new_graph.edge_info = edge_info if return_edge_scores else None
        new_graph.query_relevance = query_relevance
        
        return new_graph
    
    def get_edge_statistics(self, original_graph: torch_geometric.data.Data,
                            reevaluated_graph: torch_geometric.data.Data) -> Dict:
        """
        Calculate statistics about edge changes.
        
        Args:
            original_graph: Graph before re-evaluation
            reevaluated_graph: Graph after re-evaluation
            
        Returns:
            Dictionary with edge statistics
        """
        orig_edges = original_graph.edge_index.shape[1]
        new_edges = reevaluated_graph.edge_index.shape[1]
        
        stats = {
            'original_edges': orig_edges,
            'reevaluated_edges': new_edges,
            'edges_added': max(0, new_edges - orig_edges),
            'edges_removed': max(0, orig_edges - new_edges),
            'edge_change_ratio': abs(new_edges - orig_edges) / orig_edges if orig_edges > 0 else 0
        }
        
        if hasattr(reevaluated_graph, 'edge_info') and reevaluated_graph.edge_info:
            new_edge_count = sum(1 for e in reevaluated_graph.edge_info if e['type'] == 'new')
            stats['discovered_edges'] = new_edge_count
        
        return stats