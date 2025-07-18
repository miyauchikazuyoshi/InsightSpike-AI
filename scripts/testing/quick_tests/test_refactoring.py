#!/usr/bin/env python3
"""
Test script for refactored code
================================

Tests the major refactoring changes:
1. DataStore abstraction
2. Dependency injection
3. Cosine similarity search
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_datastore():
    """Test DataStore abstraction"""
    print("Testing DataStore abstraction...")
    
    from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
    from insightspike.core.exceptions import DataStoreSaveError, DataStoreLoadError
    
    # Create a test datastore
    store = FileSystemDataStore(base_path="./test_data")
    
    # Test episode save/load
    episodes = [
        {
            "text": "Test episode 1",
            "vec": np.random.rand(384).astype(np.float32),
            "c_value": 0.5,
            "timestamp": 12345.0
        },
        {
            "text": "Test episode 2",
            "vec": np.random.rand(384).astype(np.float32),
            "c_value": 0.7,
            "timestamp": 12346.0
        }
    ]
    
    try:
        # Save episodes
        store.save_episodes(episodes, namespace="test")
        print("✓ Episodes saved successfully")
        
        # Load episodes
        loaded = store.load_episodes(namespace="test")
        print(f"✓ Loaded {len(loaded)} episodes")
        
        # Verify content
        assert len(loaded) == len(episodes)
        assert loaded[0]["text"] == episodes[0]["text"]
        print("✓ Episode content verified")
        
    except (DataStoreSaveError, DataStoreLoadError) as e:
        print(f"✗ DataStore error: {e}")
        return False
    
    return True


def test_vector_search():
    """Test cosine similarity vector search"""
    print("\nTesting vector search with cosine similarity...")
    
    from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
    
    store = FileSystemDataStore(base_path="./test_data")
    
    # Create test vectors
    vectors = np.random.rand(10, 384).astype(np.float32)
    metadata = [{"id": i, "text": f"Document {i}"} for i in range(10)]
    
    # Save vectors
    store.save_vectors(vectors, metadata, namespace="test_vectors")
    print("✓ Vectors saved")
    
    # Search with a query vector
    query = np.random.rand(384).astype(np.float32)
    indices, distances = store.search_vectors(query, k=5, namespace="test_vectors")
    
    print(f"✓ Found {len(indices)} nearest neighbors")
    print(f"  Indices: {indices}")
    print(f"  Distances: {[f'{d:.3f}' for d in distances]}")
    
    # Verify that distances are in [0, 2] range (cosine distance)
    assert all(0 <= d <= 2 for d in distances), "Distances should be in [0, 2] range"
    print("✓ Distance range verified")
    
    return True


def test_constants():
    """Test constants and enums"""
    print("\nTesting constants and enums...")
    
    from insightspike.config.constants import FileFormat, DataType, Defaults
    
    # Test enums
    assert FileFormat.JSON.value == ".json"
    assert DataType.EPISODES.value == "episodes"
    print("✓ Enums working correctly")
    
    # Test defaults
    assert Defaults.SPIKE_GED_THRESHOLD == -0.5
    assert Defaults.REWARD_WEIGHT_IG == 0.5
    print("✓ Default constants accessible")
    
    return True


def test_exceptions():
    """Test custom exceptions"""
    print("\nTesting custom exceptions...")
    
    from insightspike.core.exceptions import (
        DataStoreSaveError,
        DataStoreLoadError,
        ConfigurationError
    )
    
    # Test exception creation
    try:
        raise DataStoreSaveError("Test error", details={"path": "/test/path"})
    except DataStoreSaveError as e:
        assert e.message == "Test error"
        assert e.details["path"] == "/test/path"
        print("✓ Custom exceptions working correctly")
    
    return True


def test_text_utils():
    """Test text utilities"""
    print("\nTesting text utilities...")
    
    from insightspike.utils.text_utils import jaccard_similarity
    
    # Test Jaccard similarity
    text1 = "The quick brown fox"
    text2 = "The quick brown dog"
    
    similarity = jaccard_similarity(text1, text2)
    print(f"  Jaccard similarity: {similarity:.3f}")
    
    # Should be 0.6 (3 common words out of 5 total unique words)
    expected = 3 / 5
    assert abs(similarity - expected) < 0.01, f"Expected {expected}, got {similarity}"
    print("✓ Jaccard similarity calculation correct")
    
    return True


def test_graph_analyzer():
    """Test GraphAnalyzer separation"""
    print("\nTesting GraphAnalyzer (SRP refactoring)...")
    
    from insightspike.features.graph_reasoning import GraphAnalyzer, RewardCalculator
    
    # Create analyzer
    analyzer = GraphAnalyzer()
    
    # Test spike detection
    metrics = {"delta_ged": -0.6, "delta_ig": 0.3}
    conflicts = {"total": 0.2}
    thresholds = {"ged": -0.5, "ig": 0.2, "conflict": 0.5}
    
    spike = analyzer.detect_spike(metrics, conflicts, thresholds)
    assert spike == True, "Should detect spike with these metrics"
    print("✓ GraphAnalyzer spike detection working")
    
    # Test reward calculator
    calc = RewardCalculator()
    rewards = calc.calculate_reward(metrics, conflicts)
    
    assert "base" in rewards
    assert "total" in rewards
    print(f"✓ RewardCalculator computed: {rewards}")
    
    return True


def cleanup():
    """Clean up test data"""
    import shutil
    test_dir = Path("./test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\n✓ Cleaned up test data")


def main():
    """Run all tests"""
    print("=== Testing Refactored InsightSpike Code ===\n")
    
    tests = [
        test_datastore,
        test_vector_search,
        test_constants,
        test_exceptions,
        test_text_utils,
        test_graph_analyzer
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    cleanup()
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)