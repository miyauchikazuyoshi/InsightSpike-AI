"""
Graph Type Adapter - Unified interface for NetworkX and PyTorch Geometric graphs
"""

import logging
from typing import Union, Optional, Dict, Any, Tuple
import numpy as np
import networkx as nx

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Data = None

logger = logging.getLogger(__name__)


class GraphTypeAdapter:
    """
    Adapter for converting between NetworkX and PyTorch Geometric graph formats.
    
    This ensures consistent graph handling throughout the system.
    NetworkX is the primary format, PyG is used only for GNN processing.
    """
    
    @staticmethod
    def to_networkx(graph: Union[nx.Graph, 'Data', Dict]) -> nx.Graph:
        """
        Convert any graph format to NetworkX.
        
        Args:
            graph: Graph in any supported format
            
        Returns:
            nx.Graph: NetworkX graph
        """
        if isinstance(graph, nx.Graph):
            return graph
        
        if isinstance(graph, dict):
            # Handle dict representation
            if 'nodes' in graph and 'edges' in graph:
                G = nx.Graph()
                G.add_nodes_from(graph['nodes'])
                G.add_edges_from(graph['edges'])
                return G
            else:
                raise ValueError("Dict must have 'nodes' and 'edges' keys")
        
        if TORCH_AVAILABLE and isinstance(graph, Data):
            return GraphTypeAdapter._pyg_to_networkx(graph)
        
        raise TypeError(f"Unsupported graph type: {type(graph)}")
    
    @staticmethod
    def to_pyg(graph: Union[nx.Graph, 'Data'], node_features: Optional[np.ndarray] = None) -> 'Data':
        """
        Convert any graph format to PyTorch Geometric Data.
        
        Args:
            graph: Graph in any supported format
            node_features: Optional node features of shape (n_nodes, n_features)
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch Geometric not available")
        
        if isinstance(graph, Data):
            return graph
        
        if isinstance(graph, nx.Graph):
            return GraphTypeAdapter._networkx_to_pyg(graph, node_features)
        
        # Convert to NetworkX first, then to PyG
        nx_graph = GraphTypeAdapter.to_networkx(graph)
        return GraphTypeAdapter._networkx_to_pyg(nx_graph, node_features)
    
    @staticmethod
    def _pyg_to_networkx(data: 'Data') -> nx.Graph:
        """Convert PyTorch Geometric Data to NetworkX."""
        G = nx.Graph()
        
        # Add nodes
        num_nodes = data.num_nodes
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        if data.edge_index is not None:
            edge_index = data.edge_index.cpu().numpy()
            edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
            G.add_edges_from(edges)
        
        # Add node features as attributes
        if data.x is not None:
            features = data.x.cpu().numpy()
            for i in range(num_nodes):
                G.nodes[i]['features'] = features[i]
        
        logger.debug(f"Converted PyG Data to NetworkX: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    @staticmethod
    def _networkx_to_pyg(G: nx.Graph, node_features: Optional[np.ndarray] = None) -> 'Data':
        """Convert NetworkX to PyTorch Geometric Data."""
        # Get node mapping
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        # Convert edges
        edge_list = []
        for u, v in G.edges():
            edge_list.append([node_mapping[u], node_mapping[v]])
            edge_list.append([node_mapping[v], node_mapping[u]])  # Undirected
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # Handle node features
        if node_features is not None:
            x = torch.tensor(node_features, dtype=torch.float)
        else:
            # Try to extract from node attributes
            features_list = []
            for node in G.nodes():
                if 'features' in G.nodes[node]:
                    features_list.append(G.nodes[node]['features'])
                elif 'embedding' in G.nodes[node]:
                    features_list.append(G.nodes[node]['embedding'])
            
            if features_list:
                x = torch.tensor(np.array(features_list), dtype=torch.float)
            else:
                # No features, create dummy ones
                x = torch.zeros((G.number_of_nodes(), 1), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index)
        logger.debug(f"Converted NetworkX to PyG Data: {data.num_nodes} nodes, {data.edge_index.shape[1]//2} edges")
        return data
    
    @staticmethod
    def ensure_networkx(graph: Any) -> Tuple[nx.Graph, bool]:
        """
        Ensure graph is in NetworkX format.
        
        Returns:
            Tuple of (networkx_graph, was_converted)
        """
        if isinstance(graph, nx.Graph):
            return graph, False
        
        converted = GraphTypeAdapter.to_networkx(graph)
        return converted, True
    
    @staticmethod
    def get_graph_info(graph: Any) -> Dict[str, Any]:
        """
        Get information about a graph regardless of its type.
        
        Returns:
            Dict with keys: type, num_nodes, num_edges, has_features
        """
        graph_type = type(graph).__name__
        
        if isinstance(graph, nx.Graph):
            return {
                'type': 'NetworkX',
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'has_features': any('features' in data for _, data in graph.nodes(data=True))
            }
        
        if TORCH_AVAILABLE and isinstance(graph, Data):
            return {
                'type': 'PyTorch Geometric',
                'num_nodes': graph.num_nodes,
                'num_edges': graph.edge_index.shape[1] // 2 if graph.edge_index is not None else 0,
                'has_features': graph.x is not None
            }
        
        # Try to convert and get info
        try:
            nx_graph = GraphTypeAdapter.to_networkx(graph)
            info = GraphTypeAdapter.get_graph_info(nx_graph)
            info['type'] = f'{graph_type} (converted)'
            return info
        except Exception as e:
            return {
                'type': graph_type,
                'num_nodes': 'unknown',
                'num_edges': 'unknown',
                'has_features': 'unknown',
                'error': str(e)
            }