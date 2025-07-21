#!/usr/bin/env python3
"""Test graph storage in SQLite"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore

# Create test graph data
test_graph = {
    'nodes': {
        'concept_1': {
            'type': 'concept',
            'attributes': {
                'text': 'Machine Learning',
                'importance': 0.9
            }
        },
        'concept_2': {
            'type': 'concept', 
            'attributes': {
                'text': 'Neural Networks',
                'importance': 0.8
            }
        },
        'concept_3': {
            'type': 'concept',
            'attributes': {
                'text': 'Deep Learning',
                'importance': 0.85
            }
        }
    },
    'edges': [
        {
            'source': 'concept_1',
            'target': 'concept_2',
            'type': 'related',
            'attributes': {'weight': 0.7}
        },
        {
            'source': 'concept_2', 
            'target': 'concept_3',
            'type': 'subset',
            'attributes': {'weight': 0.9}
        }
    ]
}

# Save to datastore
datastore = SQLiteDataStore('./data/sqlite/insightspike.db')
success = datastore.save_graph(test_graph, 'knowledge_graph_v1', namespace='main')

if success:
    print("✓ Graph saved successfully!")
    
    # Load it back
    loaded_graph = datastore.load_graph('knowledge_graph_v1', namespace='main')
    
    if loaded_graph:
        print(f"✓ Graph loaded successfully!")
        print(f"  Nodes: {len(loaded_graph.get('nodes', {}))}")
        print(f"  Edges: {len(loaded_graph.get('edges', []))}")
        
        # Display nodes
        print("\nNodes:")
        for node_id, node_data in loaded_graph.get('nodes', {}).items():
            print(f"  - {node_id}: {node_data}")
            
        print("\nEdges:")
        for edge in loaded_graph.get('edges', []):
            print(f"  - {edge['source']} -> {edge['target']} ({edge['type']})")
    else:
        print("✗ Failed to load graph")
else:
    print("✗ Failed to save graph")