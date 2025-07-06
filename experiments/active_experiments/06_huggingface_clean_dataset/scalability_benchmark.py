#!/usr/bin/env python3
"""
Scalability Benchmark for Enhanced Graph Implementation
======================================================

Compare performance with and without scalable features.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory
from insightspike.core.config import get_config


def generate_test_documents(n: int) -> List[str]:
    """Generate test documents with some semantic relationships."""
    topics = [
        "machine learning", "artificial intelligence", "deep learning",
        "natural language processing", "computer vision", "data science",
        "neural networks", "reinforcement learning", "transformers",
        "generative AI", "large language models", "embeddings"
    ]
    
    actions = [
        "revolutionizes", "transforms", "enhances", "improves",
        "accelerates", "enables", "facilitates", "advances"
    ]
    
    domains = [
        "healthcare", "finance", "education", "manufacturing",
        "retail", "transportation", "energy", "agriculture"
    ]
    
    docs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        action = actions[(i // len(topics)) % len(actions)]
        domain = domains[(i // (len(topics) * len(actions))) % len(domains)]
        
        doc = f"{topic} {action} {domain}"
        
        # Add variations
        if i % 5 == 0:
            doc += f" with {np.random.randint(80, 99)}% accuracy"
        if i % 7 == 0:
            doc += f" saving ${np.random.randint(1, 100)} million annually"
        
        docs.append(doc)
    
    return docs


def benchmark_original_implementation(docs: List[str]) -> Dict[str, Any]:
    """Benchmark the original implementation without scalable features."""
    print("\n=== Benchmarking Original Implementation ===")
    
    config = get_config()
    config.reasoning.use_scalable_graph = False
    
    # Create agent with original memory
    agent = MainAgent()
    agent.initialize()
    
    # Replace knowledge graph with original version
    if hasattr(agent.l2_memory, 'knowledge_graph'):
        agent.l2_memory.knowledge_graph = KnowledgeGraphMemory(
            embedding_dim=config.embedding.dimension,
            similarity_threshold=config.reasoning.similarity_threshold
        )
    
    start_time = time.time()
    build_times = []
    
    # Process documents
    for i, doc in enumerate(docs):
        doc_start = time.time()
        result = agent.add_episode_with_graph_update(doc)
        doc_time = time.time() - doc_start
        build_times.append(doc_time)
        
        if (i + 1) % 100 == 0:
            avg_time = sum(build_times[-100:]) / 100
            print(f"  Processed {i + 1} docs, avg time: {avg_time:.3f}s")
    
    total_time = time.time() - start_time
    
    # Get final stats
    stats = agent.get_stats()
    memory_stats = stats.get('memory_stats', {})
    graph_state = agent.get_memory_graph_state()
    
    return {
        "implementation": "original",
        "total_time": total_time,
        "avg_time_per_doc": total_time / len(docs),
        "episodes": memory_stats.get('total_episodes', 0),
        "graph_nodes": graph_state['graph'].get('num_nodes', 0),
        "graph_edges": graph_state['graph'].get('edge_index_shape', [0, 0])[1],
        "build_times": build_times
    }


def benchmark_scalable_implementation(docs: List[str]) -> Dict[str, Any]:
    """Benchmark the enhanced scalable implementation."""
    print("\n=== Benchmarking Scalable Implementation ===")
    
    config = get_config()
    config.reasoning.use_scalable_graph = True
    
    # Create agent with enhanced memory
    agent = MainAgent()
    
    # Replace with enhanced memory
    old_memory = agent.l2_memory
    agent.l2_memory = L2EnhancedScalableMemory(
        dim=config.embedding.dimension,
        config=config,
        use_scalable_graph=True
    )
    
    agent.initialize()
    
    start_time = time.time()
    build_times = []
    conflicts_detected = 0
    
    # Process documents
    for i, doc in enumerate(docs):
        doc_start = time.time()
        result = agent.add_episode_with_graph_update(doc)
        doc_time = time.time() - doc_start
        build_times.append(doc_time)
        
        # Track conflicts
        if hasattr(agent.l2_memory, 'recent_conflicts'):
            conflicts_detected = len(agent.l2_memory.recent_conflicts)
        
        if (i + 1) % 100 == 0:
            avg_time = sum(build_times[-100:]) / 100
            print(f"  Processed {i + 1} docs, avg time: {avg_time:.3f}s")
    
    total_time = time.time() - start_time
    
    # Get final stats
    if hasattr(agent.l2_memory, 'get_graph_stats'):
        graph_stats = agent.l2_memory.get_graph_stats()
    else:
        graph_stats = {}
    
    return {
        "implementation": "scalable",
        "total_time": total_time,
        "avg_time_per_doc": total_time / len(docs),
        "episodes": len(agent.l2_memory.episodes),
        "graph_nodes": graph_stats.get('nodes', 0),
        "graph_edges": graph_stats.get('edges', 0),
        "conflicts_detected": conflicts_detected,
        "build_times": build_times
    }


def compare_results(original: Dict[str, Any], scalable: Dict[str, Any], n_docs: int):
    """Compare and display results."""
    print("\n=== Performance Comparison ===")
    print(f"Dataset size: {n_docs} documents\n")
    
    print("Time Performance:")
    print(f"  Original:  {original['total_time']:.2f}s total, {original['avg_time_per_doc']:.4f}s per doc")
    print(f"  Scalable:  {scalable['total_time']:.2f}s total, {scalable['avg_time_per_doc']:.4f}s per doc")
    print(f"  Speedup:   {original['total_time'] / scalable['total_time']:.2f}x")
    
    print("\nGraph Structure:")
    print(f"  Original:  {original['graph_nodes']} nodes, {original['graph_edges']} edges")
    print(f"  Scalable:  {scalable['graph_nodes']} nodes, {scalable['graph_edges']} edges")
    
    if original['graph_nodes'] > 0 and scalable['graph_nodes'] > 0:
        orig_density = original['graph_edges'] / (original['graph_nodes'] * (original['graph_nodes'] - 1))
        scal_density = scalable['graph_edges'] / (scalable['graph_nodes'] * (scalable['graph_nodes'] - 1))
        print(f"  Original density: {orig_density:.4f}")
        print(f"  Scalable density: {scal_density:.4f}")
    
    print("\nMemory Management:")
    print(f"  Original episodes: {original['episodes']}")
    print(f"  Scalable episodes: {scalable['episodes']}")
    if 'conflicts_detected' in scalable:
        print(f"  Conflicts detected: {scalable['conflicts_detected']}")
    
    # Save detailed results
    results = {
        "experiment": "Scalability Benchmark",
        "timestamp": datetime.now().isoformat(),
        "dataset_size": n_docs,
        "original": original,
        "scalable": scalable,
        "comparison": {
            "speedup": original['total_time'] / scalable['total_time'],
            "edge_reduction": 1 - (scalable['graph_edges'] / max(1, original['graph_edges']))
        }
    }
    
    filename = f"experiment_6/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {filename}")


def main():
    """Run the scalability benchmark."""
    print("=== Scalability Benchmark ===")
    print(f"Start time: {datetime.now()}")
    
    # Test with different dataset sizes
    for n_docs in [100, 500, 1000]:
        print(f"\n{'='*50}")
        print(f"Testing with {n_docs} documents")
        print('='*50)
        
        # Generate test documents
        docs = generate_test_documents(n_docs)
        
        try:
            # Run benchmarks
            original_results = benchmark_original_implementation(docs[:n_docs])
            scalable_results = benchmark_scalable_implementation(docs[:n_docs])
            
            # Compare results
            compare_results(original_results, scalable_results, n_docs)
            
        except Exception as e:
            print(f"Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nBenchmark completed at: {datetime.now()}")


if __name__ == "__main__":
    main()