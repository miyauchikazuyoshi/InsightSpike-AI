#!/usr/bin/env python3
"""
Simple test to check episode integration behavior
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
import numpy as np


def test_integration_directly():
    """Test L2 memory integration directly"""
    print("=== Direct Episode Integration Test ===\n")
    
    # Create L2 memory manager
    memory = L2MemoryManager(dim=384)
    memory.initialize()
    
    # Test data
    test_cases = [
        # (text, should_integrate_with_previous)
        ("Research on artificial intelligence and machine learning", False),  # First, no integration
        ("Advanced AI and machine learning research findings", True),  # Very similar, should integrate
        ("Quantum computing breakthrough announced", False),  # Different topic
        ("New quantum computing advances revealed", True),  # Similar to previous
        ("Climate change impacts on agriculture", False),  # Different topic
        ("Machine learning applications in healthcare", False),  # AI topic but different context
    ]
    
    stats = {
        'total': 0,
        'integrated': 0,
        'new_episodes': 0
    }
    
    for i, (text, expected_integration) in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {text[:50]}... ---")
        
        # Generate embedding (simple simulation)
        np.random.seed(i // 2)  # Same seed for similar documents
        base = np.random.randn(384)
        if i % 2 == 1:  # Add small variation for "similar" documents
            base += np.random.randn(384) * 0.1
        embedding = base / np.linalg.norm(base)
        
        # Check current episode count
        episodes_before = len(memory.episodes)
        
        # Add episode
        idx = memory.add_episode(embedding.astype(np.float32), text, c_value=0.5)
        
        # Check if integrated
        episodes_after = len(memory.episodes)
        was_integrated = (episodes_after == episodes_before)
        
        stats['total'] += 1
        if was_integrated:
            stats['integrated'] += 1
            print("Result: INTEGRATED")
        else:
            stats['new_episodes'] += 1
            print("Result: NEW EPISODE")
        
        print(f"Expected: {'INTEGRATE' if expected_integration else 'NEW'}")
        print(f"Current episodes: {episodes_after}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total documents: {stats['total']}")
    print(f"Integrated: {stats['integrated']} ({stats['integrated']/stats['total']*100:.0f}%)")
    print(f"New episodes: {stats['new_episodes']} ({stats['new_episodes']/stats['total']*100:.0f}%)")
    print(f"Final episode count: {len(memory.episodes)}")


def check_experiment_data():
    """Check actual experiment data for integration patterns"""
    print("\n\n=== Checking Experiment Data ===\n")
    
    # Check if we have experiment results
    import glob
    result_files = glob.glob("experiment5_results_*.json")
    
    if result_files:
        print(f"Found {len(result_files)} result files")
        
        # Load most recent
        latest = sorted(result_files)[-1]
        print(f"Loading: {latest}")
        
        with open(latest, 'r') as f:
            results = json.load(f)
        
        # Analyze
        if 'scaling_test' in results:
            print("\nScaling test results:")
            for test in results['scaling_test']:
                print(f"  {test['n_docs']} docs -> {test['nodes']} nodes")
                if test['n_docs'] > 0:
                    integration_rate = 1 - (test['nodes'] / test['n_docs'])
                    print(f"    Integration rate: {integration_rate*100:.1f}%")
    else:
        print("No experiment result files found")
    
    # Check episodes.json
    if os.path.exists("data/episodes.json"):
        with open("data/episodes.json", 'r') as f:
            episodes = json.load(f)
        print(f"\nCurrent episodes.json has {len(episodes)} episodes")
        
        # Sample some episodes to see their content
        print("\nSample episodes:")
        for i in range(min(3, len(episodes))):
            ep = episodes[i]
            print(f"  Episode {i}: {ep.get('text', 'No text')[:50]}...")


if __name__ == "__main__":
    # Run direct test
    test_integration_directly()
    
    # Check experiment data
    check_experiment_data()