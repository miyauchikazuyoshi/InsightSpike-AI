#!/usr/bin/env python3
"""Check experiment 4 results"""

import os
import sys
import json
import torch
from pathlib import Path

# Check current directory
if os.path.exists('data'):
    data_dir = 'data'
else:
    data_dir = 'experiment_4/data'

# Check episodes
episodes_path = Path(data_dir) / 'episodes.json'
if episodes_path.exists():
    with open(episodes_path, 'r') as f:
        episodes = json.load(f)
    print(f"Episodes: {len(episodes)}")
    print(f"Episodes file size: {episodes_path.stat().st_size / 1024 / 1024:.2f} MB")

# Check graph
graph_path = Path(data_dir) / 'graph_pyg.pt'
if graph_path.exists():
    graph = torch.load(graph_path)
    print(f"\nGraph nodes: {graph.num_nodes}")
    print(f"Graph edges: {graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0}")
    print(f"Graph file size: {graph_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    if hasattr(graph, 'documents'):
        print(f"Documents in graph: {len(graph.documents)}")

# Check index
index_path = Path(data_dir) / 'index.faiss'
if index_path.exists():
    print(f"\nFAISS index size: {index_path.stat().st_size / 1024:.2f} KB")

# Calculate compression
if episodes_path.exists():
    total_text_size = sum(len(ep['text'].encode('utf-8')) for ep in episodes)
    total_storage = episodes_path.stat().st_size + graph_path.stat().st_size + index_path.stat().st_size
    
    print(f"\nCompression Analysis:")
    print(f"Raw text size: {total_text_size / 1024:.2f} KB")
    print(f"Total storage: {total_storage / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {total_text_size / total_storage:.2f}x")