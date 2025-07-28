"""
PyG Graph Builder
=================

Graph builder using PyTorch Geometric with multi-dimensional edge attributes.
Supports future enhancements for semantic, structural, and temporal similarities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

from ...core.episode import Episode
from ...interfaces.graph import IGraphBuilder
from ...config.legacy_adapter import LegacyConfigAdapter

logger = logging.getLogger(__name__)


class PyGGraphBuilder(IGraphBuilder):
    """
    Graph builder using PyTorch Geometric for advanced graph operations.
    
    This implementation:
    - Uses PyTorch Geometric Data objects
    - Supports multi-dimensional edge attributes
    - Creates edges based on similarity with edge_attr tensors
    - Enables future GNN and multi-modal edge features
    """
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        if config is None:
            from ...config import InsightSpikeConfig
            config = InsightSpikeConfig()
        
        self.config = LegacyConfigAdapter.ensure_pydantic(config)
        self.similarity_threshold = self.config.graph.similarity_threshold
        
        logger.info(f"PyGGraphBuilder initialized with threshold: {self.similarity_threshold}")
    
    def build_graph(self, episodes: List[Episode]) -> Data:
        """
        Build a PyG graph from episodes.
        
        Args:
            episodes: List of Episode objects
            
        Returns:
            PyTorch Geometric Data object with node features and edge attributes
        """
        if not episodes:
            return Data(x=torch.empty(0, 384), edge_index=torch.empty(2, 0, dtype=torch.long))
        
        # Extract embeddings and create node features
        embeddings = np.array([ep.vec for ep in episodes])
        x = torch.tensor(embeddings, dtype=torch.float)
        
        # Build edges based on similarity
        edge_list = []
        edge_attrs = []  # For future multi-dimensional edge attributes
        
        if len(episodes) > 1:
            similarities = cosine_similarity(embeddings)
            
            # Create edges for similar nodes
            for i in range(len(episodes)):
                for j in range(i + 1, len(episodes)):
                    sim = similarities[i, j]
                    if sim >= self.similarity_threshold:
                        # Add bidirectional edges
                        edge_list.extend([[i, j], [j, i]])
                        # Store similarity as edge attribute (can be extended later)
                        edge_attrs.extend([sim, sim])
        
        # Convert to PyG format
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 1, dtype=torch.float)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(episodes)
        
        # Store episode metadata
        data.episodes = episodes
        
        logger.info(f"Built PyG graph with {data.num_nodes} nodes and {edge_index.size(1)} edges")
        return data
    
    def update_graph(self, graph: Data, new_episodes: List[Episode]) -> Data:
        """
        Update existing graph with new episodes.
        
        Args:
            graph: Existing PyG Data object
            new_episodes: New episodes to add
            
        Returns:
            Updated PyG Data object
        """
        if not new_episodes:
            return graph
        
        # For PyG, it's more efficient to rebuild with all episodes
        all_episodes = graph.episodes if hasattr(graph, 'episodes') else []
        all_episodes.extend(new_episodes)
        
        return self.build_graph(all_episodes)
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """Set the similarity threshold for edge creation"""
        self.similarity_threshold = threshold
        logger.info(f"Similarity threshold set to: {threshold}")
    
    def build_subgraph(self, graph: Data, node_ids: List[int], hop_limit: int = 1) -> Data:
        """
        Build a subgraph around specified nodes.
        
        Args:
            graph: Full PyG graph
            node_ids: Starting node IDs
            hop_limit: Number of hops to include
            
        Returns:
            Subgraph as PyG Data object
        """
        if not node_ids or hop_limit < 0:
            return Data(x=torch.empty(0, 384), edge_index=torch.empty(2, 0, dtype=torch.long))
        
        # Implementation for PyG subgraph extraction
        # This is a simplified version - could be enhanced with torch_geometric.utils.k_hop_subgraph
        
        # For now, return the full graph (can be optimized later)
        logger.warning("Subgraph extraction for PyG not fully implemented, returning full graph")
        return graph
    
    def merge_graphs(self, graph1: Data, graph2: Data) -> Data:
        """
        Merge two PyG graphs.
        
        Args:
            graph1: First PyG graph
            graph2: Second PyG graph
            
        Returns:
            Merged PyG graph
        """
        # Merge episodes from both graphs
        episodes1 = graph1.episodes if hasattr(graph1, 'episodes') else []
        episodes2 = graph2.episodes if hasattr(graph2, 'episodes') else []
        
        all_episodes = episodes1 + episodes2
        
        # Rebuild merged graph
        merged = self.build_graph(all_episodes)
        
        logger.info(f"Merged graphs: {graph1.num_nodes} + {graph2.num_nodes} → {merged.num_nodes} nodes")
        return merged