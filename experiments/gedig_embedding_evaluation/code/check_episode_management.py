#!/usr/bin/env python3
"""
Check Episode Management in Current Data
=======================================

Analyzes how episode management is working with the current data.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def analyze_episodes():
    """Analyze current episode structure"""
    
    # Load episodes
    episodes_path = Path("data/episodes.json")
    with open(episodes_path, 'r') as f:
        episodes = json.load(f)
    
    print(f"Total episodes: {len(episodes)}")
    
    # Analyze episode characteristics
    c_values = []
    text_lengths = []
    texts = []
    
    for ep in episodes:
        c_values.append(ep['c'])
        text = ep['text']
        text_lengths.append(len(text))
        texts.append(text)
    
    # C-value distribution
    print(f"\nC-value statistics:")
    print(f"  Min: {min(c_values):.3f}")
    print(f"  Max: {max(c_values):.3f}")
    print(f"  Mean: {np.mean(c_values):.3f}")
    print(f"  Std: {np.std(c_values):.3f}")
    
    # Text length distribution
    print(f"\nText length statistics:")
    print(f"  Min: {min(text_lengths)} chars")
    print(f"  Max: {max(text_lengths)} chars")
    print(f"  Mean: {np.mean(text_lengths):.0f} chars")
    
    # Check for duplicates/near-duplicates
    print(f"\nChecking for duplicates...")
    
    # Exact duplicates
    text_counter = Counter(texts)
    duplicates = [(text, count) for text, count in text_counter.items() if count > 1]
    
    if duplicates:
        print(f"Found {len(duplicates)} exact duplicate groups:")
        for text, count in duplicates[:5]:
            print(f"  - '{text[:50]}...' appears {count} times")
    else:
        print("No exact duplicates found")
    
    # Near duplicates (based on first 100 chars)
    text_starts = [text[:100] for text in texts]
    start_counter = Counter(text_starts)
    near_duplicates = [(start, count) for start, count in start_counter.items() if count > 1]
    
    if near_duplicates:
        print(f"\nFound {len(near_duplicates)} near-duplicate groups (same start):")
        for start, count in near_duplicates[:5]:
            print(f"  - '{start[:50]}...' appears {count} times")
    
    # Episodes that might be prunable
    prune_threshold = 0.1
    prunable = [i for i, c in enumerate(c_values) if c < prune_threshold]
    print(f"\nEpisodes below prune threshold ({prune_threshold}): {len(prunable)}")
    
    # Episodes that were split/merged (check metadata)
    split_episodes = []
    merged_episodes = []
    
    for i, ep in enumerate(episodes):
        metadata = ep.get('metadata', {})
        if 'split_from' in metadata:
            split_episodes.append(i)
        if 'merged_from' in metadata:
            merged_episodes.append(i)
    
    print(f"\nEpisodes created from splits: {len(split_episodes)}")
    print(f"Episodes created from merges: {len(merged_episodes)}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Episode Management Analysis", fontsize=16)
    
    # C-value distribution
    ax = axes[0, 0]
    ax.hist(c_values, bins=30, alpha=0.7, color='blue')
    ax.axvline(prune_threshold, color='red', linestyle='--', label=f'Prune threshold ({prune_threshold})')
    ax.set_xlabel('C-value')
    ax.set_ylabel('Count')
    ax.set_title('C-value Distribution')
    ax.legend()
    
    # Text length distribution
    ax = axes[0, 1]
    ax.hist(text_lengths, bins=30, alpha=0.7, color='green')
    ax.set_xlabel('Text Length (chars)')
    ax.set_ylabel('Count')
    ax.set_title('Text Length Distribution')
    ax.set_yscale('log')
    
    # C-value vs text length
    ax = axes[1, 0]
    ax.scatter(text_lengths, c_values, alpha=0.5)
    ax.set_xlabel('Text Length (chars)')
    ax.set_ylabel('C-value')
    ax.set_title('C-value vs Text Length')
    
    # Episode types
    ax = axes[1, 1]
    episode_types = {
        'Normal': len(episodes) - len(split_episodes) - len(merged_episodes),
        'From splits': len(split_episodes),
        'From merges': len(merged_episodes)
    }
    ax.pie(episode_types.values(), labels=episode_types.keys(), autopct='%1.1f%%')
    ax.set_title('Episode Types')
    
    plt.tight_layout()
    plt.savefig('episode_management_analysis.png', dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to episode_management_analysis.png")
    
    # Test similarity between episodes
    print("\nTesting episode similarity...")
    memory = L2MemoryManager(dim=384)
    memory.load()
    
    # Find most similar episode pairs
    high_similarity_pairs = []
    
    for i in range(min(50, len(memory.episodes))):  # Check first 50
        for j in range(i+1, min(50, len(memory.episodes))):
            ep1 = memory.episodes[i]
            ep2 = memory.episodes[j]
            
            similarity = np.dot(ep1.vec, ep2.vec)
            
            if similarity > 0.85:  # Integration threshold
                high_similarity_pairs.append((i, j, similarity))
    
    if high_similarity_pairs:
        print(f"Found {len(high_similarity_pairs)} highly similar pairs (>0.85):")
        for i, j, sim in sorted(high_similarity_pairs, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  Episodes {i} and {j}: similarity {sim:.3f}")
            print(f"    Ep{i}: '{memory.episodes[i].text[:50]}...'")
            print(f"    Ep{j}: '{memory.episodes[j].text[:50]}...'")
    else:
        print("No highly similar pairs found in sample")

def test_management_triggers():
    """Test if episode management is actively working"""
    print("\n" + "="*60)
    print("Testing Episode Management Triggers")
    print("="*60)
    
    agent = MainAgent()
    agent.initialize()
    
    initial_count = len(agent.l2_memory.episodes)
    print(f"Initial episode count: {initial_count}")
    
    # Add a duplicate document
    print("\nAdding duplicate document...")
    duplicate_text = "Machine learning is a field of artificial intelligence that enables systems to learn from data."
    result1 = agent.process_question(f"Learn this: {duplicate_text}", max_cycles=1)
    
    # Add the same document again
    result2 = agent.process_question(f"Learn this: {duplicate_text}", max_cycles=1)
    
    current_count = len(agent.l2_memory.episodes)
    print(f"Episode count after duplicates: {current_count}")
    print(f"Episodes added: {current_count - initial_count}")
    
    # Add a complex document that might be split
    print("\nAdding complex document...")
    complex_text = ("Artificial intelligence encompasses multiple fields. "
                   "Machine learning is one approach. Deep learning uses neural networks. "
                   "Natural language processing handles text. Computer vision processes images. "
                   "Reinforcement learning uses rewards. Each field has unique applications.")
    
    result3 = agent.process_question(f"Learn this complex information: {complex_text}", max_cycles=1)
    
    final_count = len(agent.l2_memory.episodes)
    print(f"Final episode count: {final_count}")
    print(f"Episodes added from complex doc: {final_count - current_count}")
    
    # Check if any pruning happens
    low_c_count = sum(1 for ep in agent.l2_memory.episodes if ep.c < 0.1)
    print(f"\nLow C-value episodes: {low_c_count}")

if __name__ == "__main__":
    analyze_episodes()
    test_management_triggers()