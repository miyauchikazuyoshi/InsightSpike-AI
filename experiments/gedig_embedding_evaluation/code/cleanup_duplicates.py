#!/usr/bin/env python3
"""
Cleanup Duplicate Episodes
==========================

Removes duplicate episodes from the current data structure.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager

import numpy as np
import json
from pathlib import Path
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_current_data():
    """Backup current data before cleanup"""
    backup_dir = Path("data_backup_before_cleanup")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for filename in ["episodes.json", "index.faiss", "graph_pyg.pt"]:
        src = Path("data") / filename
        if src.exists():
            dst = backup_dir / f"{filename}.{timestamp}"
            shutil.copy(src, dst)
            logger.info(f"Backed up {filename} to {dst}")

def cleanup_duplicates():
    """Remove duplicate episodes and rebuild index"""
    
    logger.info("Loading current episodes...")
    memory = L2MemoryManager(dim=384)
    memory.load()
    
    initial_count = len(memory.episodes)
    logger.info(f"Initial episode count: {initial_count}")
    
    # Build a new episode list without duplicates
    unique_episodes = []
    seen_texts = set()
    seen_vectors = []
    
    duplicate_count = 0
    near_duplicate_count = 0
    
    for i, episode in enumerate(memory.episodes):
        # Check for exact text duplicate
        if episode.text in seen_texts:
            duplicate_count += 1
            logger.debug(f"Skipping duplicate episode {i}: {episode.text[:50]}...")
            continue
        
        # Check for near-duplicate vectors
        is_near_duplicate = False
        for seen_vec in seen_vectors:
            similarity = np.dot(episode.vec, seen_vec)
            if similarity > 0.98:  # Very high similarity threshold for near-duplicates
                near_duplicate_count += 1
                logger.debug(f"Skipping near-duplicate episode {i} (similarity: {similarity:.3f})")
                is_near_duplicate = True
                break
        
        if not is_near_duplicate:
            unique_episodes.append(episode)
            seen_texts.add(episode.text)
            seen_vectors.append(episode.vec)
    
    logger.info(f"Removed {duplicate_count} exact duplicates")
    logger.info(f"Removed {near_duplicate_count} near-duplicates")
    logger.info(f"Unique episodes: {len(unique_episodes)}")
    
    # Create new memory manager with unique episodes
    new_memory = L2MemoryManager(dim=384)
    
    # Add unique episodes
    for episode in unique_episodes:
        # Use direct addition to avoid re-encoding
        new_memory.episodes.append(episode)
    
    # Train index with unique episodes
    if len(new_memory.episodes) >= 2:
        new_memory._train_index()
        logger.info("Trained new index with unique episodes")
    
    # Save cleaned data
    new_memory.save()
    logger.info("Saved cleaned data")
    
    # Also update the graph if needed
    agent = MainAgent()
    agent.initialize()
    agent.save_state()
    
    return initial_count, len(unique_episodes)

def verify_cleanup():
    """Verify the cleanup results"""
    
    # Load cleaned data
    memory = L2MemoryManager(dim=384)
    memory.load()
    
    logger.info(f"\nVerifying cleaned data:")
    logger.info(f"Episode count: {len(memory.episodes)}")
    
    # Check for remaining duplicates
    texts = [ep.text for ep in memory.episodes]
    unique_texts = set(texts)
    
    if len(texts) == len(unique_texts):
        logger.info("✓ No duplicate texts found")
    else:
        logger.warning(f"⚠ Still have {len(texts) - len(unique_texts)} duplicate texts")
    
    # Check C-value distribution
    c_values = [ep.c for ep in memory.episodes]
    logger.info(f"C-value range: {min(c_values):.3f} - {max(c_values):.3f}")
    logger.info(f"Mean C-value: {np.mean(c_values):.3f}")
    
    # Test retrieval
    logger.info("\nTesting retrieval on cleaned data:")
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks"
    ]
    
    for query in test_queries:
        results = memory.search_episodes(query, k=5)
        if results:
            logger.info(f"Query: '{query[:30]}...' - Top result similarity: {results[0]['similarity']:.3f}")

def main():
    """Run the cleanup process"""
    print("="*60)
    print("Episode Deduplication and Cleanup")
    print("="*60)
    
    # Backup first
    print("\nBacking up current data...")
    backup_current_data()
    
    # Run cleanup
    print("\nCleaning up duplicates...")
    initial, final = cleanup_duplicates()
    
    # Verify results
    print("\nVerifying cleanup...")
    verify_cleanup()
    
    print("\n" + "="*60)
    print("Cleanup Summary")
    print("="*60)
    print(f"Initial episodes: {initial}")
    print(f"Final episodes: {final}")
    print(f"Removed: {initial - final} ({(initial - final) / initial * 100:.1f}%)")
    print("\nBackup saved to: data_backup_before_cleanup/")

if __name__ == "__main__":
    main()