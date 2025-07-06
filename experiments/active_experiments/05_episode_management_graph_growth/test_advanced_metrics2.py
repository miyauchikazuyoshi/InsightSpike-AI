#!/usr/bin/env python3
"""Test advanced metrics with PyG graphs"""

import os
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("Testing advanced metrics...")

try:
    from insightspike.utils.advanced_graph_metrics import AdvancedGraphMetrics
    from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
    
    # Create test documents
    docs = []
    for i in range(10):
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        docs.append({
            'text': f'Document {i}',
            'embedding': embedding,
            'id': i
        })
    
    # Build graph
    print("Building graph...")
    builder = ScalableGraphBuilder()
    graph1 = builder.build_graph(docs[:5])
    graph2 = builder.build_graph(docs[:8])
    
    print(f"Graph1: {graph1.num_nodes} nodes, {graph1.edge_index.size(1)} edges")
    print(f"Graph2: {graph2.num_nodes} nodes, {graph2.edge_index.size(1)} edges")
    
    # Test advanced metrics
    print("\nTesting AdvancedGraphMetrics...")
    metrics = AdvancedGraphMetrics(use_exact_ged=False)
    
    # Test ΔGED
    print("\nCalculating ΔGED...")
    try:
        delta_ged = metrics.delta_ged(graph1, graph2)
        print(f"ΔGED: {delta_ged}")
    except Exception as e:
        print(f"ΔGED error: {e}")
        traceback.print_exc()
    
    # Test ΔIG
    print("\nCalculating ΔIG...")
    try:
        delta_ig = metrics.delta_ig(graph1, graph2)
        print(f"ΔIG: {delta_ig}")
    except Exception as e:
        print(f"ΔIG error: {e}")
        traceback.print_exc()
    
    # Test combined insight score
    print("\nCalculating combined insight score...")
    try:
        score, is_insight, components = metrics.combined_insight_score(graph1, graph2)
        print(f"Score: {score}, Is insight: {is_insight}")
        print(f"Components: {components}")
    except Exception as e:
        print(f"Combined score error: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"\nMain error: {e}")
    traceback.print_exc()

print("\nTest complete.")