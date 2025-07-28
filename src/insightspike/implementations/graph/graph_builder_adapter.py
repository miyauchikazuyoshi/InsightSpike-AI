"""
Graph Builder Adapter
====================

Adapts between different graph builder implementations and ensures
PyG output for consistency with future multi-dimensional edge features.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import Data

from ...core.episode import Episode
from ..layers.scalable_graph_builder import ScalableGraphBuilder
from .pyg_graph_builder import PyGGraphBuilder

logger = logging.getLogger(__name__)


class GraphBuilderAdapter:
    """
    Adapter that provides a unified interface for graph building.
    
    Always returns PyG Data objects for consistency and future enhancements.
    """
    
    def __init__(self, config=None, use_scalable: bool = True):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration object
            use_scalable: If True, use ScalableGraphBuilder; otherwise use PyGGraphBuilder
        """
        self.config = config
        self.use_scalable = use_scalable
        
        if use_scalable:
            self.builder = ScalableGraphBuilder(config)
            logger.info("Using ScalableGraphBuilder")
        else:
            self.builder = PyGGraphBuilder(config)
            logger.info("Using PyGGraphBuilder")
    
    def build_graph(self, episodes: List[Episode]) -> Data:
        """
        Build a graph from episodes.
        
        Args:
            episodes: List of Episode objects
            
        Returns:
            PyG Data object
        """
        if not episodes:
            return Data(x=torch.empty(0, 384), edge_index=torch.empty(2, 0, dtype=torch.long))
        
        if self.use_scalable:
            # Convert episodes to document format for ScalableGraphBuilder
            documents = []
            embeddings = []
            
            for idx, episode in enumerate(episodes):
                documents.append({
                    "text": episode.text,
                    "index": idx,
                    "c_value": episode.c,
                    "timestamp": episode.timestamp,
                    "metadata": episode.metadata
                })
                embeddings.append(episode.vec)
            
            embeddings_array = np.array(embeddings)
            
            # Build PyG graph
            pyg_graph = self.builder.build_graph(documents, embeddings_array)
            
            # Store episodes in graph
            pyg_graph.episodes = episodes
            
            return pyg_graph
        else:
            # Direct PyG building
            return self.builder.build_graph(episodes)
    
    def update_graph(self, graph: Data, new_episodes: List[Episode]) -> Data:
        """
        Update existing graph with new episodes.
        
        Args:
            graph: Existing PyG Data object
            new_episodes: New episodes to add
            
        Returns:
            Updated PyG Data object
        """
        # For both builders, update is handled by the builder itself
        return self.builder.update_graph(graph, new_episodes)
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """Set the similarity threshold for edge creation"""
        if hasattr(self.builder, 'set_similarity_threshold'):
            self.builder.set_similarity_threshold(threshold)
        else:
            # For ScalableGraphBuilder
            self.builder.similarity_threshold = threshold