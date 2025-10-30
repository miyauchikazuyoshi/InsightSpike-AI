"""
PyTorch Geometric Compatible Metrics
====================================

Provides GED and IG calculations for PyTorch Geometric Data objects.
"""

import logging
import networkx as nx
import numpy as np
from typing import Any, Optional

logger = logging.getLogger(__name__)


def pyg_to_networkx(pyg_graph: Any) -> nx.Graph:
    """Convert PyTorch Geometric Data to NetworkX graph."""
    try:
        import torch
        
        G = nx.Graph()
        
        # Add nodes
        if hasattr(pyg_graph, 'num_nodes'):
            num_nodes = pyg_graph.num_nodes
        elif hasattr(pyg_graph, 'x') and pyg_graph.x is not None:
            num_nodes = pyg_graph.x.size(0)
        else:
            return G  # Empty graph
            
        for i in range(num_nodes):
            node_attrs = {'idx': i}
            # Add node features if available
            if hasattr(pyg_graph, 'x') and pyg_graph.x is not None:
                node_attrs['features'] = pyg_graph.x[i].cpu().numpy() if hasattr(pyg_graph.x, 'cpu') else pyg_graph.x[i].numpy()
            G.add_node(i, **node_attrs)
        
        # Add edges
        if hasattr(pyg_graph, 'edge_index') and pyg_graph.edge_index is not None:
            edge_index = pyg_graph.edge_index
            if hasattr(edge_index, 'cpu'):
                edge_index = edge_index.cpu().numpy()
            else:
                edge_index = edge_index.numpy()
                
            # Convert to list of edges
            edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]
            
            # Add edge weights if available
            if hasattr(pyg_graph, 'edge_attr') and pyg_graph.edge_attr is not None:
                edge_attr = pyg_graph.edge_attr
                if hasattr(edge_attr, 'cpu'):
                    edge_attr = edge_attr.cpu().numpy()
                else:
                    edge_attr = edge_attr.numpy()
                    
                for i, (u, v) in enumerate(edges):
                    G.add_edge(u, v, weight=float(edge_attr[i]) if edge_attr.ndim == 1 else float(edge_attr[i, 0]))
            else:
                G.add_edges_from(edges)
                
        return G
        
    except Exception as e:
        logger.error(f"Failed to convert PyG to NetworkX: {e}")
        return nx.Graph()


def delta_ged_pyg(g_old: Any, g_new: Any) -> float:
    """
    Calculate ΔGED for PyTorch Geometric graphs.
    
    Returns negative value when graph simplifies (insight formation).
    """
    try:
        # Convert to NetworkX
        nx_old = pyg_to_networkx(g_old)
        nx_new = pyg_to_networkx(g_new)
        
        # Basic size comparison for quick check
        old_nodes = nx_old.number_of_nodes()
        new_nodes = nx_new.number_of_nodes()
        old_edges = nx_old.number_of_edges()
        new_edges = nx_new.number_of_edges()
        
        logger.debug(f"Graph comparison: ({old_nodes},{old_edges}) -> ({new_nodes},{new_edges})")
        
        # Simple heuristic GED calculation
        # More sophisticated calculation would use actual graph edit distance
        node_diff = abs(new_nodes - old_nodes)
        edge_diff = abs(new_edges - old_edges)
        
        # Weight node changes more than edge changes
        ged = node_diff * 1.0 + edge_diff * 0.5
        
        # Make negative if graph simplified
        if new_nodes < old_nodes or (new_nodes == old_nodes and new_edges < old_edges):
            ged = -ged
            
        logger.debug(f"Calculated GED: {ged}")
        return float(ged)
        
    except Exception as e:
        logger.error(f"GED calculation failed: {e}")
        return 0.0


def delta_ig_pyg(g_old: Any, g_new: Any) -> float:
    """
    Calculate ΔIG for PyTorch Geometric graphs.
    
    Returns positive value when information increases (insight formation).
    """
    try:
        # Extract node features
        old_features = None
        new_features = None
        
        if hasattr(g_old, 'x') and g_old.x is not None:
            old_features = g_old.x.cpu().numpy() if hasattr(g_old.x, 'cpu') else g_old.x.numpy()
        if hasattr(g_new, 'x') and g_new.x is not None:
            new_features = g_new.x.cpu().numpy() if hasattr(g_new.x, 'cpu') else g_new.x.numpy()
            
        if old_features is None or new_features is None:
            return 0.0
            
        # Simple entropy-based IG calculation
        # Calculate variance as proxy for entropy
        old_var = np.var(old_features)
        new_var = np.var(new_features)
        
        # Lower variance = more organized = information gain
        ig = float(old_var - new_var)
        
        # Normalize to reasonable range
        if old_var > 0:
            ig = ig / old_var
            
        logger.debug(f"Calculated IG: {ig} (old_var={old_var:.4f}, new_var={new_var:.4f})")
        return ig
        
    except Exception as e:
        logger.error(f"IG calculation failed: {e}")
        return 0.0