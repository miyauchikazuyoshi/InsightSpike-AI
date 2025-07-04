#!/usr/bin/env python3
"""Test imports to debug the issue"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from insightspike.core.layers.layer3_graph_reasoner import GraphBuilder
    print("✓ GraphBuilder imported successfully")
except Exception as e:
    print(f"✗ Failed to import GraphBuilder: {e}")

try:
    from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
    print("✓ ScalableGraphBuilder imported successfully")
except Exception as e:
    print(f"✗ Failed to import ScalableGraphBuilder: {e}")

try:
    import faiss
    print("✓ FAISS imported successfully")
except Exception as e:
    print(f"✗ Failed to import FAISS: {e}")
    print("  Installing faiss-cpu may be required: pip install faiss-cpu")