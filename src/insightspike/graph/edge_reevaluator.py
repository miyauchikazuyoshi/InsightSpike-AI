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
        """Re-evaluate edges based on updated node representations with safety guards."""
        # Safety: missing graph or representations
        if (graph is None or not hasattr(graph, 'edge_index') or
                updated_representations is None or len(updated_representations) == 0):
            empty = torch_geometric.data.Data(
                x=torch.empty(0, 0),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                num_nodes=0
            )
            empty.edge_info = [{'skipped_empty': True}]
            return empty

        num_nodes = len(updated_representations)
        if num_nodes == 0:
            empty = torch_geometric.data.Data(
                x=torch.empty(0, 0),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                num_nodes=0
            )
            empty.edge_info = [{'skipped_empty': True}]
            return empty

        # Build node matrix
        try:
            node_matrix = np.vstack([updated_representations[i] for i in range(num_nodes)])
        except Exception as e:
            logger.warning(f"EdgeReevaluator: failed to stack representations ({e}); returning empty graph")
            empty = torch_geometric.data.Data(
                x=torch.empty(0, 0),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                num_nodes=0
            )
            empty.edge_info = [{'stack_failure': True}]
            return empty

        # Ensure float32
        if node_matrix.dtype != np.float32:
            node_matrix = node_matrix.astype(np.float32, copy=False)

        # Query vector shape
        if query_vector is None:
            query_vector = np.zeros((1, node_matrix.shape[1]), dtype=np.float32)
        elif query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Handle potential dim mismatch (truncate/pad query)
        if query_vector.shape[1] != node_matrix.shape[1]:
            min_d = min(query_vector.shape[1], node_matrix.shape[1])
            query_vector = query_vector[:, :min_d]
            node_matrix = node_matrix[:, :min_d]

        # Query relevance
        try:
            query_relevance = cosine_similarity(node_matrix, query_vector).flatten()
        except Exception:
            # Fallback: zeros
            query_relevance = np.zeros(num_nodes, dtype=np.float32)

        # Existing edges (tensor -> numpy)
        edge_index_np = graph.edge_index.cpu().numpy() if torch.is_tensor(graph.edge_index) else graph.edge_index
        if edge_index_np.size == 0:
            edge_index_np = np.empty((2, 0), dtype=np.int64)
        existing_pairs: Set[Tuple[int, int]] = set()
        for s, d in edge_index_np.T:
            existing_pairs.add((min(s, d), max(s, d)))

        new_edges: List[List[int]] = []
        edge_weights: List[float] = []
        edge_info: List[Dict] = []

        # Re-evaluate existing edges
        for idx_e, (src, dst) in enumerate(edge_index_np.T):
            if src >= num_nodes or dst >= num_nodes:
                continue  # Skip out-of-range (defensive)
            try:
                sim = float(cosine_similarity(
                    node_matrix[src].reshape(1, -1),
                    node_matrix[dst].reshape(1, -1)
                )[0, 0])
            except Exception:
                continue
            relevance_boost = (query_relevance[src] + query_relevance[dst]) / 2.0
            adjusted_sim = sim * (1.0 + 0.2 * relevance_boost)
            # Additional retention rule: if one endpoint is highly query-relevant, keep existing edge
            high_relevance_keep = False
            if adjusted_sim < self.similarity_threshold:
                # If either endpoint strongly matches the query (>=0.8) we retain the edge even if node-node sim is low.
                if max(query_relevance[src], query_relevance[dst]) >= 0.8:
                    high_relevance_keep = True
                    # Provide a minimum adjusted similarity just below threshold to preserve ordering but mark retention.
                    adjusted_sim = max(adjusted_sim, self.similarity_threshold * 0.85)
            if adjusted_sim >= self.similarity_threshold or high_relevance_keep:
                new_edges.append([src, dst])
                if hasattr(graph, 'edge_attr') and getattr(graph, 'edge_attr') is not None:
                    try:
                        if idx_e < graph.edge_attr.size(0):
                            old_weight = float(graph.edge_attr[idx_e].item())
                        else:
                            old_weight = adjusted_sim
                        weight = self.edge_decay_factor * old_weight + (1 - self.edge_decay_factor) * adjusted_sim
                    except Exception:
                        weight = adjusted_sim
                else:
                    weight = adjusted_sim
                edge_weights.append(weight)
                edge_info.append({
                    'src': int(src), 'dst': int(dst), 'type': 'existing',
                    'similarity': sim, 'adjusted_similarity': adjusted_sim, 'weight': weight
                })

        logger.debug(f"EdgeReevaluator: retained {len(new_edges)} edges >= threshold")

        # Discover new edges if we have at least 2 nodes
        added_new_edges = 0
        if num_nodes >= 2:
            try:
                similarity_matrix = cosine_similarity(node_matrix)
            except Exception:
                similarity_matrix = np.eye(num_nodes, dtype=np.float32)
            candidates: List[Dict] = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if (i, j) in existing_pairs:
                        continue
                    sim = float(similarity_matrix[i, j])
                    if sim >= self.new_edge_threshold:
                        relevance_boost = (query_relevance[i] + query_relevance[j]) / 2.0
                        adjusted_sim = sim * (1.0 + 0.3 * relevance_boost)
                        candidates.append({
                            'src': i, 'dst': j, 'similarity': sim,
                            'adjusted_similarity': adjusted_sim, 'relevance': relevance_boost
                        })
            candidates.sort(key=lambda c: c['adjusted_similarity'], reverse=True)
            per_node_count = [0] * num_nodes
            for c in candidates:
                s, d = c['src'], c['dst']
                if (per_node_count[s] < self.max_new_edges_per_node and
                        per_node_count[d] < self.max_new_edges_per_node):
                    # Add bidirectional
                    new_edges.append([s, d])
                    new_edges.append([d, s])
                    edge_weights.extend([c['adjusted_similarity']] * 2)
                    per_node_count[s] += 1
                    per_node_count[d] += 1
                    added_new_edges += 1
                    edge_info.append({
                        'src': s, 'dst': d, 'type': 'new',
                        'similarity': c['similarity'],
                        'adjusted_similarity': c['adjusted_similarity'],
                        'weight': c['adjusted_similarity']
                    })
        logger.debug(f"EdgeReevaluator: added {added_new_edges} new edges")

        if len(new_edges) == 0:
            edge_index_tensor = torch.empty(2, 0, dtype=torch.long)
            edge_attr_tensor = torch.empty(0, 1, dtype=torch.float)
        else:
            edge_index_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        new_graph = torch_geometric.data.Data(
            x=torch.tensor(node_matrix, dtype=torch.float),
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            num_nodes=num_nodes
        )
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
        try:
            orig_edge_index = getattr(original_graph, 'edge_index', None)
            new_edge_index = getattr(reevaluated_graph, 'edge_index', None)
            if orig_edge_index is None or new_edge_index is None:
                return {
                    'original_edges': 0,
                    'reevaluated_edges': 0,
                    'edges_added': 0,
                    'edges_removed': 0,
                    'edge_change_ratio': 0.0
                }
            orig_edges_count = orig_edge_index.shape[1]
            new_edges_count = new_edge_index.shape[1]

            # Build directed edge sets for precise diff accounting
            orig_set = set((int(orig_edge_index[0, i]), int(orig_edge_index[1, i])) for i in range(orig_edges_count))
            new_set = set((int(new_edge_index[0, i]), int(new_edge_index[1, i])) for i in range(new_edges_count))
            removed = orig_set - new_set
            added = new_set - orig_set
            stats = {
                'original_edges': orig_edges_count,
                'reevaluated_edges': new_edges_count,
                'edges_added': len(added),
                'edges_removed': len(removed),
                'edge_change_ratio': (len(added) + len(removed)) / orig_edges_count if orig_edges_count > 0 else 0.0
            }
            if hasattr(reevaluated_graph, 'edge_info') and reevaluated_graph.edge_info:
                new_edge_count = sum(1 for e in reevaluated_graph.edge_info if e.get('type') == 'new')
                stats['discovered_edges'] = new_edge_count
            return stats
        except Exception as e:
            logger.debug(f"EdgeReevaluator.get_edge_statistics failed: {e}")
            return {
                'original_edges': 0,
                'reevaluated_edges': 0,
                'edges_added': 0,
                'edges_removed': 0,
                'edge_change_ratio': 0.0
            }