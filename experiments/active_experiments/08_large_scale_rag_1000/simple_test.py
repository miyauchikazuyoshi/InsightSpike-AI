#!/usr/bin/env python3
"""
Simple Performance Test
=======================
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config


def simple_test():
    """Run a simple test."""
    print("=== Simple Performance Test ===")
    
    # Setup
    config = get_config()
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "data"
    
    config.paths.data_dir = str(data_dir)
    config.memory.index_file = str(data_dir / "index.faiss")
    config.reasoning.graph_file = str(data_dir / "graph_pyg.pt")
    
    # Check files
    print("\nChecking files:")
    for fname in ["episodes.json", "index.faiss", "graph_pyg.pt", "qa_pairs.json"]:
        fpath = data_dir / fname
        if fpath.exists():
            print(f"✓ {fname}: {fpath.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            print(f"✗ {fname}: NOT FOUND")
    
    # Load Q&A pairs
    qa_file = data_dir / "qa_pairs.json"
    with open(qa_file, 'r') as f:
        qa_pairs = json.load(f)
    print(f"\nLoaded {len(qa_pairs)} Q&A pairs")
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = MainAgent(config)
    agent.l2_memory = L2EnhancedScalableMemory(
        dim=config.embedding.dimension,
        config=config,
        use_scalable_graph=True
    )
    
    # Load state
    if agent.load_state():
        print(f"✓ Loaded {len(agent.l2_memory.episodes)} episodes")
    else:
        print("✗ Failed to load state")
        return
    
    # Test a few questions
    print("\n=== Testing Q&A ===")
    test_qa = qa_pairs[:5]  # Test first 5
    
    for i, qa in enumerate(test_qa):
        print(f"\nQ{i+1}: {qa['question']}")
        print(f"Expected: {qa['answer']}")
        
        try:
            # Simple search
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            query_vec = embedder.encode(qa['question'])
            
            # Search in FAISS
            if hasattr(agent.l2_memory, 'index') and agent.l2_memory.index:
                D, I = agent.l2_memory.index.search(
                    query_vec.reshape(1, -1).astype(np.float32), 
                    k=3
                )
                
                print("Retrieved episodes:")
                for idx, dist in zip(I[0], D[0]):
                    if idx >= 0 and idx < len(agent.l2_memory.episodes):
                        ep = agent.l2_memory.episodes[idx]
                        ep_text = ep.text if hasattr(ep, 'text') else str(ep)
                        print(f"  [{idx}] (dist={dist:.3f}): {ep_text[:100]}...")
                        
                        if qa['answer'].lower() in ep_text.lower():
                            print(f"  ✓ Contains answer!")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    simple_test()