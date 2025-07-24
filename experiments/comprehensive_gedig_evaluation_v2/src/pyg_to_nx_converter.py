"""
Convert PyTorch Geometric graphs to NetworkX for metric calculation.
"""

import networkx as nx
import numpy as np
from typing import Any, Optional


def convert_pyg_to_networkx(pyg_graph: Any) -> nx.Graph:
    """
    Convert PyTorch Geometric Data object to NetworkX graph.
    
    Args:
        pyg_graph: PyTorch Geometric Data object
        
    Returns:
        NetworkX graph with node embeddings
    """
    G = nx.Graph()
    
    # Handle None or empty graph
    if pyg_graph is None:
        return G
    
    # Check if it's already a NetworkX graph
    if isinstance(pyg_graph, nx.Graph):
        return pyg_graph
    
    # PyTorch Geometric Data object
    if hasattr(pyg_graph, 'edge_index') and hasattr(pyg_graph, 'x'):
        # Add nodes with embeddings
        num_nodes = pyg_graph.x.shape[0] if pyg_graph.x is not None else 0
        
        for i in range(num_nodes):
            # Get embedding for node
            embedding = pyg_graph.x[i].numpy() if hasattr(pyg_graph.x[i], 'numpy') else pyg_graph.x[i]
            G.add_node(i, embedding=embedding)
        
        # Add edges
        if pyg_graph.edge_index is not None and pyg_graph.edge_index.shape[1] > 0:
            edge_index = pyg_graph.edge_index
            # Convert to numpy if it's a tensor
            if hasattr(edge_index, 'numpy'):
                edge_index = edge_index.numpy()
            
            # Add edges (PyG uses COO format)
            for j in range(edge_index.shape[1]):
                src = int(edge_index[0, j])
                dst = int(edge_index[1, j])
                # Avoid duplicate edges in undirected graph
                if src < dst:
                    G.add_edge(src, dst)
    
    return G


def extract_embeddings_from_pyg(pyg_graph: Any) -> np.ndarray:
    """
    Extract embeddings from PyTorch Geometric graph.
    
    Args:
        pyg_graph: PyTorch Geometric Data object
        
    Returns:
        Numpy array of embeddings
    """
    if pyg_graph is None:
        return np.array([]).reshape(0, 768)
    
    if hasattr(pyg_graph, 'x') and pyg_graph.x is not None:
        # Convert to numpy if tensor
        if hasattr(pyg_graph.x, 'numpy'):
            return pyg_graph.x.numpy()
        else:
            return np.array(pyg_graph.x)
    
    return np.array([]).reshape(0, 768)