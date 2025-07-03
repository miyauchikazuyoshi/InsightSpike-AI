#!/usr/bin/env python3
"""
Dynamic Growth Experiment using MainAgent API
Demonstrates proper data and graph growth in InsightSpike-AI
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent

def load_huggingface_data(limit=50):
    """Load HuggingFace dataset samples"""
    data_file = Path(__file__).parent.parent / "experiment_1" / "data" / "huggingface_samples.json"
    
    if not data_file.exists():
        print(f"Error: HuggingFace data not found at {data_file}")
        return []
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data[:limit]

def measure_file_sizes():
    """Measure current data file sizes"""
    files = {
        "episodes.json": Path("data/episodes.json"),
        "graph_pyg.pt": Path("data/graph_pyg.pt"),
        "index.faiss": Path("data/index.faiss")
    }
    
    sizes = {}
    for name, path in files.items():
        if path.exists():
            sizes[name] = path.stat().st_size
        else:
            sizes[name] = 0
    
    return sizes

def run_dynamic_growth_experiment():
    """Run the dynamic growth experiment"""
    print("=== InsightSpike-AI Dynamic Growth Experiment ===\n")
    
    # Initialize agent
    print("1. Initializing MainAgent...")
    agent = MainAgent()
    if not agent.initialize():
        print("Failed to initialize agent")
        return
    
    # Load initial state
    print("2. Loading initial state...")
    agent.load_state()
    
    # Get initial statistics
    initial_stats = agent.get_stats()
    initial_sizes = measure_file_sizes()
    initial_graph_state = agent.get_memory_graph_state()
    
    print(f"\nInitial State:")
    memory_stats = initial_stats.get('memory_stats', {})
    print(f"  Episodes: {memory_stats.get('total_episodes', 0)}")
    print(f"  Graph nodes: {initial_graph_state.get('graph_nodes', 0)}")
    print(f"  File sizes:")
    for name, size in initial_sizes.items():
        print(f"    {name}: {size:,} bytes")
    
    # Load HuggingFace data
    print("\n3. Loading HuggingFace dataset samples...")
    hf_data = load_huggingface_data(limit=50)
    
    if not hf_data:
        print("No HuggingFace data available, using synthetic data")
        hf_data = [
            {"text": "Machine learning is a subset of artificial intelligence.", "metadata": {"source": "synthetic"}},
            {"text": "Deep learning uses neural networks with multiple layers.", "metadata": {"source": "synthetic"}},
            {"text": "Natural language processing enables computers to understand human language.", "metadata": {"source": "synthetic"}},
            {"text": "Computer vision allows machines to interpret visual information.", "metadata": {"source": "synthetic"}},
            {"text": "Reinforcement learning trains agents through reward and punishment.", "metadata": {"source": "synthetic"}}
        ]
    
    print(f"  Loaded {len(hf_data)} samples")
    
    # Add data with graph updates
    print("\n4. Adding episodes with graph updates...")
    successful_adds = 0
    failed_adds = 0
    
    for i, item in enumerate(hf_data):
        text = item.get('text', '')
        if not text:
            continue
        
        # Add episode with graph update
        result = agent.add_episode_with_graph_update(
            text=text,
            c_value=0.5
        )
        
        if result.get('success', False):
            successful_adds += 1
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(hf_data)} episodes added")
        else:
            failed_adds += 1
    
    print(f"\n  Results: {successful_adds} successful, {failed_adds} failed")
    
    # Save state after all additions
    print("\n5. Saving state to disk...")
    if agent.save_state():
        print("  State saved successfully")
    else:
        print("  Failed to save state")
    
    # Get final statistics
    final_stats = agent.get_stats()
    final_sizes = measure_file_sizes()
    
    # Calculate growth metrics
    print("\n=== Growth Metrics ===")
    print(f"\nEpisode Growth:")
    initial_memory = initial_stats.get('memory_stats', {})
    final_memory = final_stats.get('memory_stats', {})
    initial_episodes = initial_memory.get('total_episodes', 0)
    final_episodes = final_memory.get('total_episodes', 0)
    print(f"  Initial: {initial_episodes}")
    print(f"  Final: {final_episodes}")
    print(f"  Added: {final_episodes - initial_episodes}")
    if initial_episodes > 0:
        print(f"  Growth rate: {(final_episodes / initial_episodes - 1) * 100:.1f}%")
    
    print(f"\nGraph Growth:")
    # Get final graph state
    final_graph_state = agent.get_memory_graph_state()
    initial_nodes = initial_graph_state.get('graph_nodes', 0)
    final_nodes = final_graph_state.get('graph_nodes', 0)
    print(f"  Initial nodes: {initial_nodes}")
    print(f"  Final nodes: {final_nodes}")
    print(f"  Added nodes: {final_nodes - initial_nodes}")
    
    print(f"\nFile Size Growth:")
    for name in initial_sizes:
        initial = initial_sizes[name]
        final = final_sizes[name]
        growth = final - initial
        growth_pct = (final / initial - 1) * 100 if initial > 0 else 0
        print(f"  {name}:")
        print(f"    Initial: {initial:,} bytes")
        print(f"    Final: {final:,} bytes")
        print(f"    Growth: {growth:,} bytes ({growth_pct:.1f}%)")
    
    # Compression ratio calculation
    print("\n=== Compression Analysis ===")
    
    # Estimate raw text size
    total_text_size = sum(len(item.get('text', '').encode('utf-8')) for item in hf_data[:successful_adds])
    total_stored_size = sum(final_sizes.values()) - sum(initial_sizes.values())
    
    if total_stored_size > 0:
        compression_ratio = total_text_size / total_stored_size
        print(f"  Raw text size: {total_text_size:,} bytes")
        print(f"  Stored size growth: {total_stored_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # Verify data persistence
    print("\n6. Verifying data persistence...")
    
    # Create new agent and load state
    verify_agent = MainAgent()
    verify_agent.initialize()
    verify_agent.load_state()
    
    verify_stats = verify_agent.get_stats()
    verify_memory = verify_stats.get('memory_stats', {})
    verify_graph = verify_agent.get_memory_graph_state()
    
    print(f"  Episodes after reload: {verify_memory.get('total_episodes', 0)}")
    print(f"  Graph nodes after reload: {verify_graph.get('graph_nodes', 0)}")
    
    if verify_memory.get('total_episodes', 0) == final_memory.get('total_episodes', 0):
        print("  ✅ Data persistence verified!")
    else:
        print("  ❌ Data persistence issue detected")
    
    print("\n=== Experiment Complete ===")

if __name__ == "__main__":
    run_dynamic_growth_experiment()