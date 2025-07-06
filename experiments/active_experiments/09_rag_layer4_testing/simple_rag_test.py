#!/usr/bin/env python3
"""
Simple RAG Test for Experiment 9
=================================
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config
from sentence_transformers import SentenceTransformer


def main():
    """Run simple RAG test."""
    print("=== EXPERIMENT 9: SIMPLE RAG TEST ===")
    
    # Setup
    config = get_config()
    
    # Check if we need to clear data folder first
    data_dir = Path(config.paths.data_dir)
    if (data_dir / "qa_pairs.json").exists():
        print("Found existing data, will use it")
    else:
        print("No data found, rebuilding...")
        # Run the build script again
        os.system("python experiment_9/build_and_test_rag.py")
        return
    
    # Load Q&A pairs
    with open(data_dir / "qa_pairs.json", 'r') as f:
        qa_pairs = json.load(f)
    print(f"\nLoaded {len(qa_pairs)} Q&A pairs")
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = MainAgent(config)
    
    # Load saved state
    if not agent.load_state():
        print("Failed to load state!")
        return
    
    print(f"Loaded {len(agent.l2_memory.episodes)} episodes")
    
    # Initialize embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test 50 random questions
    test_size = 50
    test_indices = random.sample(range(len(qa_pairs)), min(test_size, len(qa_pairs)))
    
    # Test 1: Standard RAG
    print(f"\n=== Testing Standard RAG (n={test_size}) ===")
    rag_correct = 0
    rag_times = []
    
    for i, idx in enumerate(test_indices):
        qa = qa_pairs[idx]
        start = time.time()
        
        # Encode question
        query_vec = embedder.encode(qa['question'])
        
        # Search in FAISS
        if hasattr(agent.l2_memory, 'index') and agent.l2_memory.index:
            D, I = agent.l2_memory.index.search(
                query_vec.reshape(1, -1).astype(np.float32),
                k=5
            )
            
            # Check retrieved episodes
            found = False
            for ep_idx in I[0]:
                if 0 <= ep_idx < len(agent.l2_memory.episodes):
                    ep = agent.l2_memory.episodes[ep_idx]
                    
                    # Handle different episode formats
                    if hasattr(ep, 'text'):
                        ep_text = ep.text
                    elif hasattr(ep, 'content'):
                        ep_text = ep.content
                    elif isinstance(ep, dict) and 'text' in ep:
                        ep_text = ep['text']
                    else:
                        ep_text = str(ep)
                    
                    if qa['answer'].lower() in ep_text.lower():
                        rag_correct += 1
                        found = True
                        break
        
        rag_times.append(time.time() - start)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{test_size} - Accuracy so far: {rag_correct/(i+1):.1%}")
    
    rag_accuracy = rag_correct / test_size
    rag_avg_time = np.mean(rag_times)
    
    # Test 2: InsightSpike
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
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{test_size} - Accuracy so far: {insight_correct/(i+1):.1%}")
    
    insight_accuracy = insight_correct / test_size
    insight_avg_time = np.mean(insight_times)
    
    # Results
    print("\n" + "="*60)
    print("📊 EXPERIMENT 9 RESULTS")
    print("="*60)
    print(f"Knowledge Base: {len(agent.l2_memory.episodes)} episodes from {len(qa_pairs)} Q&A pairs")
    print(f"Integration Rate: {(1 - len(agent.l2_memory.episodes)/(len(qa_pairs)*2))*100:.1f}%")
    print("-"*60)
    print(f"{'Method':<20} {'Accuracy':<15} {'Avg Time':<15}")
    print("-"*60)
    print(f"{'Standard RAG':<20} {rag_accuracy:.1%} ({rag_correct}/{test_size}){'':<3} {rag_avg_time:.3f}s")
    print(f"{'InsightSpike':<20} {insight_accuracy:.1%} ({insight_correct}/{test_size}){'':<3} {insight_avg_time:.3f}s")
    print("-"*60)
    
    # Comparison
    accuracy_diff = (insight_accuracy - rag_accuracy) * 100
    speed_ratio = rag_avg_time / insight_avg_time if insight_avg_time > 0 else 0
    
    print(f"{'Improvement:':<20} {accuracy_diff:+.1f}%{'':<10} {speed_ratio:.2f}x")
    print("="*60)
    
    # Save results
    results = {
        'experiment': 'Experiment 9 Simple RAG Test',
        'timestamp': datetime.now().isoformat(),
        'knowledge_base': {
            'episodes': len(agent.l2_memory.episodes),
            'qa_pairs': len(qa_pairs),
            'integration_rate': (1 - len(agent.l2_memory.episodes)/(len(qa_pairs)*2))
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
            'accuracy_improvement': accuracy_diff,
            'speed_ratio': speed_ratio,
            'winner': 'InsightSpike' if accuracy_diff > 0 else 'Standard RAG'
        }
    }
    
    results_file = Path(__file__).parent / f"simple_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Backup experiment data
    print("\n=== Backing up experiment data ===")
    import shutil
    backup_dir = Path(__file__).parent / "data_final"
    
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    shutil.copytree(data_dir, backup_dir)
    print(f"✓ Data backed up to {backup_dir}")
    
    # List backed up files
    for f in ['episodes.json', 'index.faiss', 'graph_pyg.pt', 'qa_pairs.json']:
        fpath = backup_dir / f
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"  {f}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()