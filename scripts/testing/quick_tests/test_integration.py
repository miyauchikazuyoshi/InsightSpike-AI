#!/usr/bin/env python3
"""
Integration test for refactored components
"""

import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_datastore_integration():
    """Test the complete DataStore workflow"""
    print("=== DataStore Integration Test ===\n")
    
    from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
    
    # Create store
    store = FileSystemDataStore(base_path="./test_integration_data")
    
    # 1. Test Episode Save/Load
    print("1. Testing Episode Save/Load...")
    episodes = [
        {
            "text": "The theory of relativity changed physics forever.",
            "vec": np.random.rand(384).astype(np.float32),
            "c_value": 0.8,
            "timestamp": 1234567890.0,
            "metadata": {"source": "physics_book", "chapter": 1}
        },
        {
            "text": "Quantum mechanics describes the behavior of particles.",
            "vec": np.random.rand(384).astype(np.float32),
            "c_value": 0.7,
            "timestamp": 1234567891.0,
            "metadata": {"source": "physics_book", "chapter": 2}
        }
    ]
    
    try:
        # Save
        store.save_episodes(episodes, namespace="physics")
        print("✓ Episodes saved")
        
        # Load
        loaded = store.load_episodes(namespace="physics")
        print(f"✓ Loaded {len(loaded)} episodes")
        print(f"  First episode: '{loaded[0]['text'][:50]}...'")
        
    except Exception as e:
        print(f"✗ Episode test failed: {e}")
        return False
    
    # 2. Test Vector Search with Cosine Similarity
    print("\n2. Testing Vector Search...")
    
    # Create some test vectors
    num_docs = 100
    dim = 384
    vectors = np.random.rand(num_docs, dim).astype(np.float32)
    metadata = [
        {"id": i, "text": f"Document about topic {i % 10}"} 
        for i in range(num_docs)
    ]
    
    try:
        # Save vectors
        store.save_vectors(vectors, metadata, namespace="documents")
        print(f"✓ Saved {num_docs} vectors")
        
        # Search with a query
        query_vector = np.random.rand(dim).astype(np.float32)
        k = 10
        indices, distances = store.search_vectors(query_vector, k=k, namespace="documents")
        
        print(f"✓ Found {len(indices)} nearest neighbors")
        print(f"  Top 3 indices: {indices[:3]}")
        print(f"  Top 3 distances: {[f'{d:.3f}' for d in distances[:3]]}")
        
        # Verify distances are in cosine distance range [0, 2]
        assert all(0 <= d <= 2 for d in distances), "Invalid distance range"
        print("✓ Distance range verified (cosine distance)")
        
    except Exception as e:
        print(f"✗ Vector search test failed: {e}")
        return False
    
    # 3. Test Graph Save/Load
    print("\n3. Testing Graph Save/Load...")
    
    # Create a mock graph
    class MockGraph:
        def __init__(self):
            self.num_nodes = 5
            self.edges = [(0, 1), (1, 2), (2, 3)]
            self.data = {"metrics": {"delta_ged": -0.5, "delta_ig": 0.3}}
    
    graph = MockGraph()
    
    try:
        # Save graph
        store.save_graph(graph, graph_id="test_graph", namespace="graphs")
        print("✓ Graph saved")
        
        # Load graph
        loaded_graph = store.load_graph(graph_id="test_graph", namespace="graphs")
        print(f"✓ Graph loaded: {loaded_graph.num_nodes} nodes")
        
    except Exception as e:
        print(f"✗ Graph test failed: {e}")
        return False
    
    return True


def test_config_and_exceptions():
    """Test configuration and exception handling"""
    print("\n=== Configuration and Exception Test ===\n")
    
    from insightspike.config.constants import FileFormat, DataType, FILE_FORMAT_MAPPING
    from insightspike.core.exceptions import DataStoreSaveError, DataStoreLoadError
    
    # Test file format mapping
    print("1. File Format Mapping:")
    for data_type, file_format in FILE_FORMAT_MAPPING.items():
        print(f"  {data_type.value} -> {file_format.value}")
    
    # Test exception handling
    print("\n2. Exception Handling:")
    try:
        # Simulate an error
        raise DataStoreSaveError(
            "Disk full", 
            details={"path": "/data/episodes.json", "size": "1GB"}
        )
    except DataStoreSaveError as e:
        print(f"✓ Caught DataStoreSaveError: {e.message}")
        print(f"  Details: {e.details}")
    
    return True


def test_graph_components():
    """Test refactored graph components"""
    print("\n=== Graph Components Test ===\n")
    
    from insightspike.features.graph_reasoning import GraphAnalyzer, RewardCalculator
    
    # Test GraphAnalyzer
    print("1. GraphAnalyzer:")
    analyzer = GraphAnalyzer()
    
    metrics = {"delta_ged": -0.6, "delta_ig": 0.4}
    conflicts = {"total": 0.1}
    thresholds = {"ged": -0.5, "ig": 0.2, "conflict": 0.5}
    
    spike = analyzer.detect_spike(metrics, conflicts, thresholds)
    quality = analyzer.assess_quality(metrics, conflicts)
    
    print(f"  Spike detected: {spike}")
    print(f"  Reasoning quality: {quality:.3f}")
    
    # Test RewardCalculator
    print("\n2. RewardCalculator:")
    calculator = RewardCalculator()
    
    rewards = calculator.calculate_reward(metrics, conflicts)
    print(f"  Base reward: {rewards['base']:.3f}")
    print(f"  Total reward: {rewards['total']:.3f}")
    
    return True


def cleanup():
    """Clean up test data"""
    import shutil
    test_dir = Path("./test_integration_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\n✓ Cleaned up test data")


def main():
    """Run integration tests"""
    print("InsightSpike Refactoring Integration Tests")
    print("==========================================\n")
    
    tests = [
        test_datastore_integration,
        test_config_and_exceptions,
        test_graph_components
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    cleanup()
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{len(tests)}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)