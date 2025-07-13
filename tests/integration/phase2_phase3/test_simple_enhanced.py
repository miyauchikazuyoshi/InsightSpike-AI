#!/usr/bin/env python3
"""
Simple test for enhanced episode management
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.core.layers.layer2_enhanced import EnhancedL2MemoryManager


def test_basic_functionality():
    """Test basic enhanced L2 functionality"""

    print("=== Testing Enhanced L2 Memory Manager ===\n")

    # Create enhanced manager
    manager = EnhancedL2MemoryManager(dim=10)  # Small dimension for testing

    # Configure for aggressive integration
    manager.integration_config.similarity_threshold = 0.7
    manager.integration_config.graph_weight = 0.3

    # Test episodes
    episodes = [
        ("AI research paper", np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Machine learning study", np.array([0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Climate science report", np.array([0, 0, 0.9, 0.1, 0, 0, 0, 0, 0, 0])),
        (
            "AI and ML advances",
            np.array([0.85, 0.15, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),  # Should integrate
    ]

    print("Adding episodes...")
    for text, vec in episodes:
        vec = vec / np.linalg.norm(vec)
        idx = manager.add_episode(vec.astype(np.float32), text, 0.5)
        print(f"  Added '{text}' -> Total episodes: {len(manager.episodes)}")

    # Check statistics
    stats = manager.get_enhanced_stats()

    print(f"\nIntegration Statistics:")
    print(f"  Total attempts: {stats['integration_stats']['total_attempts']}")
    print(
        f"  Successful integrations: {stats['integration_stats']['successful_integrations']}"
    )

    print(f"\nFinal episode count: {len(manager.episodes)}")
    print("\nEpisodes:")
    for i, ep in enumerate(manager.episodes):
        print(f"  {i}: {ep.text[:50]}...")


def test_splitting():
    """Test episode splitting"""

    print("\n\n=== Testing Episode Splitting ===\n")

    manager = EnhancedL2MemoryManager(dim=10)
    manager.splitting_config.max_episode_length = 100  # Low for testing
    manager.splitting_config.enable_auto_split = True

    # Long episode that should split
    long_text = (
        "This is about AI. "
        "This is about climate. "
        "This is about quantum computing. "
        "This is about biology. "
        "This is about robotics."
    )

    vec = np.random.randn(10).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    print(f"Adding long episode (length: {len(long_text)})...")
    idx = manager.add_episode(vec, long_text, 0.5)

    print(f"Result: {len(manager.episodes)} episodes")

    # Manual split test
    if len(manager.episodes) == 1:
        print("\nTesting manual split...")
        manager._check_and_split_if_needed(0)
        print(f"After manual split: {len(manager.episodes)} episodes")

    print("\nFinal episodes:")
    for i, ep in enumerate(manager.episodes):
        print(f"  {i}: {ep.text}")


if __name__ == "__main__":
    test_basic_functionality()
    test_splitting()
