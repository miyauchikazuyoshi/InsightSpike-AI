#!/usr/bin/env python3
"""Verify graph nodes in saved file"""

import torch
from pathlib import Path

# Load graph
graph_path = Path("data/graph_pyg.pt")
graph = torch.load(graph_path)

print(f"Graph type: {type(graph)}")
print(f"Number of nodes: {graph.num_nodes}")
print(f"Number of edges: {graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0}")
print(f"Has features: {hasattr(graph, 'x') and graph.x is not None}")

if hasattr(graph, 'x') and graph.x is not None:
    print(f"Feature shape: {graph.x.shape}")
    
if hasattr(graph, 'documents'):
    print(f"Number of documents: {len(graph.documents)}")