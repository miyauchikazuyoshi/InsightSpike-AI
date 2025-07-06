#!/usr/bin/env python3
"""Quick verification of graph functionality"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
import torch
import json

print("=== Graph Functionality Verification ===\n")

# Check if graph file exists
graph_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/graph_pyg.pt")
episodes_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/episodes.json")

print("1. File Status:")
print(f"   graph_pyg.pt: {'✓ Exists' if graph_path.exists() else '✗ Not found'}")
print(f"   episodes.json: {'✓ Exists' if episodes_path.exists() else '✗ Not found'}")

if graph_path.exists():
    # Load and analyze graph
    print("\n2. Graph Analysis:")
    data = torch.load(graph_path)
    nodes = data.x.shape[0] if data.x is not None else 0
    edges = data.edge_index.shape[1] if data.edge_index is not None else 0
    density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
    
    print(f"   Nodes: {nodes}")
    print(f"   Edges: {edges}")
    print(f"   Density: {density:.3f}")
    print(f"   File size: {graph_path.stat().st_size} bytes ({graph_path.stat().st_size/1024:.1f} KB)")
    
    # Show edge statistics
    if edges > 0:
        edge_index = data.edge_index
        degrees = torch.zeros(nodes)
        for i in range(edges):
            src = edge_index[0, i].item()
            degrees[src] += 1
        
        print(f"\n3. Degree Statistics:")
        print(f"   Average degree: {degrees.mean().item():.2f}")
        print(f"   Max degree: {degrees.max().item():.0f}")
        print(f"   Min degree: {degrees.min().item():.0f}")
        
        # Show sample edges
        print(f"\n4. Sample Edges (first 5):")
        for i in range(min(5, edges)):
            src, dst = edge_index[:, i]
            print(f"   {src.item()} → {dst.item()}")
    
if episodes_path.exists():
    print("\n5. Episodes Analysis:")
    with open(episodes_path, 'r') as f:
        episodes = json.load(f)
    print(f"   Total episodes: {len(episodes)}")
    print(f"   File size: {episodes_path.stat().st_size} bytes ({episodes_path.stat().st_size/1024/1024:.1f} MB)")

print("\n=== Verification Complete ===")