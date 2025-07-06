#!/usr/bin/env python3
"""
Fix Save Issue - Manual Save Episodes
=====================================
"""

import json
from pathlib import Path

def fix_save_issue():
    """Manually save episodes from agent memory."""
    
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from insightspike.core.agents.main_agent import MainAgent
    from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
    from insightspike.core.config import get_config
    
    print("=== Fixing Save Issue ===")
    
    # Setup
    config = get_config()
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "data"
    
    config.paths.data_dir = str(data_dir)
    config.memory.index_file = str(data_dir / "index.faiss")
    config.reasoning.graph_file = str(data_dir / "graph_pyg.pt")
    
    # Initialize agent
    agent = MainAgent(config)
    agent.l2_memory = L2EnhancedScalableMemory(
        dim=config.embedding.dimension,
        config=config,
        use_scalable_graph=True
    )
    
    # Check if graph exists (which means we have data in memory)
    graph_file = data_dir / "graph_pyg.pt"
    if graph_file.exists():
        print("✓ Graph file exists, attempting to load state...")
        
        # Try loading
        if agent.load_state():
            print(f"✓ Loaded state successfully")
            print(f"  Episodes in memory: {len(agent.l2_memory.episodes)}")
            
            # Manually save episodes
            if hasattr(agent.l2_memory, 'episodes') and agent.l2_memory.episodes:
                # Save episodes
                episodes_file = data_dir / "episodes.json"
                episodes_data = []
                
                for i, ep in enumerate(agent.l2_memory.episodes):
                    ep_dict = {
                        'id': i,
                        'text': ep.text if hasattr(ep, 'text') else str(ep),
                        'c': ep.c_value if hasattr(ep, 'c_value') else 0.5,
                        'metadata': ep.metadata if hasattr(ep, 'metadata') else {},
                        'vec': ep.vector.tolist() if hasattr(ep, 'vector') else []
                    }
                    episodes_data.append(ep_dict)
                
                with open(episodes_file, 'w') as f:
                    json.dump(episodes_data, f, indent=2)
                
                print(f"✓ Saved {len(episodes_data)} episodes to {episodes_file}")
                
                # Try to save FAISS index
                if hasattr(agent.l2_memory, 'index') and agent.l2_memory.index:
                    import faiss
                    index_file = data_dir / "index.faiss"
                    faiss.write_index(agent.l2_memory.index, str(index_file))
                    print(f"✓ Saved FAISS index to {index_file}")
            else:
                print("✗ No episodes found in memory")
        else:
            print("✗ Failed to load state")
    else:
        print("✗ No graph file found")
    
    # Check what we have now
    print("\n=== Current State ===")
    files = [
        (data_dir / "episodes.json", "Episodes"),
        (data_dir / "index.faiss", "FAISS index"),
        (data_dir / "graph_pyg.pt", "Graph"),
        (data_dir / "qa_pairs.json", "Q&A pairs")
    ]
    
    for file_path, name in files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {name}: {size_mb:.2f} MB")
        else:
            print(f"✗ {name}: NOT FOUND")


if __name__ == "__main__":
    fix_save_issue()