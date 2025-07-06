#!/usr/bin/env python3
"""
Final Performance Test - Experiment 8
=====================================

Compare InsightSpike with standard RAG on 1000-scale knowledge base.
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import random

# Import InsightSpike components
from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config
from sentence_transformers import SentenceTransformer


def main():
    """Run the final performance test."""
    print("=== EXPERIMENT 8: FINAL PERFORMANCE TEST ===")
    print(f"Start time: {datetime.now()}")
    
    # Setup
    config = get_config()
    experiment_dir = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiment_8")
    data_dir = experiment_dir / "data"
    
    config.paths.data_dir = str(data_dir)
    config.memory.index_file = str(data_dir / "index.faiss")
    config.reasoning.graph_file = str(data_dir / "graph_pyg.pt")
    
    # Check data files
    print("\n=== Data Files ===")
    for fname in ["episodes.json", "index.faiss", "graph_pyg.pt", "qa_pairs.json"]:
        fpath = data_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"✓ {fname}: {size_mb:.2f} MB")
        else:
            print(f"✗ {fname}: NOT FOUND")
            return
    
    # Load Q&A pairs
    with open(data_dir / "qa_pairs.json", 'r') as f:
        qa_pairs = json.load(f)
    print(f"\nLoaded {len(qa_pairs)} Q&A pairs")
    
    # Initialize agent
    print("\n=== Initializing InsightSpike ===")
    agent = MainAgent(config)
    agent.l2_memory = L2EnhancedScalableMemory(
        dim=config.embedding.dimension,
        config=config,
        use_scalable_graph=True
    )
    
    # Load state
    if not agent.load_state():
        print("✗ Failed to load state")
        return
    
    print(f"✓ Loaded {len(agent.l2_memory.episodes)} episodes")
    
    # Initialize standard embedder
    print("\n=== Initializing Standard RAG ===")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedder initialized")
    
    # Select test questions
    test_size = 100
    test_indices = random.sample(range(len(qa_pairs)), min(test_size, len(qa_pairs)))
    
    # Test 1: Standard RAG (Semantic Search Only)
    print(f"\n=== Testing Standard RAG (n={test_size}) ===")
    rag_correct = 0
    rag_times = []
    
    for i, idx in enumerate(test_indices):
        qa = qa_pairs[idx]
        start = time.time()
        
        # Encode question
        query_vec = embedder.encode(qa['question'])
        
        # Search in FAISS
        if agent.l2_memory.index:
            D, I = agent.l2_memory.index.search(
                query_vec.reshape(1, -1).astype(np.float32),
                k=5
            )
            
            # Check retrieved episodes
            for ep_idx in I[0]:
                if 0 <= ep_idx < len(agent.l2_memory.episodes):
                    ep = agent.l2_memory.episodes[ep_idx]
                    ep_text = ep.text if hasattr(ep, 'text') else ""
                    
                    if qa['answer'].lower() in ep_text.lower():
                        rag_correct += 1
                        break
        
        rag_times.append(time.time() - start)
        
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{test_size}")
    
    rag_accuracy = rag_correct / test_size
    rag_avg_time = np.mean(rag_times)
    
    print(f"\nStandard RAG Results:")
    print(f"  Accuracy: {rag_accuracy:.1%} ({rag_correct}/{test_size})")
    print(f"  Avg time: {rag_avg_time:.3f}s")
    
    # Test 2: InsightSpike (Graph-Enhanced)
    print(f"\n=== Testing InsightSpike (n={test_size}) ===")
    insight_correct = 0
    insight_times = []
    
    for i, idx in enumerate(test_indices):
        qa = qa_pairs[idx]
        start = time.time()
        
        try:
            # Use InsightSpike's full pipeline
            result = agent.process_question(
                qa['question'],
                max_cycles=1,
                verbose=False
            )
            
            response = result.get('response', '')
            
            if qa['answer'].lower() in response.lower():
                insight_correct += 1
                
        except Exception as e:
            pass  # Count as incorrect
        
        insight_times.append(time.time() - start)
        
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{test_size}")
    
    insight_accuracy = insight_correct / test_size
    insight_avg_time = np.mean(insight_times)
    
    print(f"\nInsightSpike Results:")
    print(f"  Accuracy: {insight_accuracy:.1%} ({insight_correct}/{test_size})")
    print(f"  Avg time: {insight_avg_time:.3f}s")
    
    # Comparison
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Method':<25} {'Accuracy':<15} {'Avg Time':<15}")
    print("-"*60)
    print(f"{'Standard RAG':<25} {rag_accuracy:.1%} ({rag_correct}/{test_size}){'':<3} {rag_avg_time:.3f}s")
    print(f"{'InsightSpike':<25} {insight_accuracy:.1%} ({insight_correct}/{test_size}){'':<3} {insight_avg_time:.3f}s")
    print("-"*60)
    
    # Calculate improvements
    accuracy_improvement = (insight_accuracy - rag_accuracy) * 100
    speed_ratio = rag_avg_time / insight_avg_time if insight_avg_time > 0 else 0
    
    print(f"{'Accuracy Improvement:':<25} {accuracy_improvement:+.1f}%")
    print(f"{'Speed Comparison:':<25} {speed_ratio:.2f}x")
    print("="*60)
    
    # Save results
    results = {
        'experiment': 'Experiment 8: 1000-Scale RAG Comparison',
        'timestamp': datetime.now().isoformat(),
        'knowledge_base': {
            'episodes': len(agent.l2_memory.episodes),
            'qa_pairs': len(qa_pairs)
        },
        'test_size': test_size,
        'results': {
            'standard_rag': {
                'accuracy': rag_accuracy,
                'correct': rag_correct,
                'avg_time': rag_avg_time
            },
            'insightspike': {
                'accuracy': insight_accuracy,
                'correct': insight_correct,
                'avg_time': insight_avg_time
            }
        },
        'comparison': {
            'accuracy_improvement': f"{accuracy_improvement:+.1f}%",
            'speed_ratio': f"{speed_ratio:.2f}x",
            'winner': 'InsightSpike' if accuracy_improvement > 0 else 'Standard RAG'
        }
    }
    
    report_file = experiment_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Report saved to {report_file}")
    print("\n✅ Experiment 8 Complete!")


if __name__ == "__main__":
    main()