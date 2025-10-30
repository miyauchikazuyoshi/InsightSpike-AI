"""
Simple SQLiteDataStore Test
===========================

Test core functionality without conflicting with existing event loops.
"""

import os

import numpy as np

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore


def main():
    """Test SQLiteDataStore functionality"""

    # Initialize datastore
    db_path = "./test_data/test_datastore_simple.db"
    os.makedirs("./test_data", exist_ok=True)

    # Remove existing database for clean test
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=== Testing SQLiteDataStore ===")
    store = SQLiteDataStore(db_path, vector_dim=384)

    # Test 1: Add episodes using sync method
    print("\n1. Testing sync episode operations...")
    episodes = [
        {
            "text": "The sky is blue",
            "vec": np.random.rand(384).astype(np.float32),
            "c": 0.7,
            "metadata": {"source": "test1"},
        },
        {
            "text": "Water is essential for life",
            "vec": np.random.rand(384).astype(np.float32),
            "c": 0.8,
            "metadata": {"source": "test2"},
        },
        {
            "text": "Python is a programming language",
            "vec": np.random.rand(384).astype(np.float32),
            "c": 0.9,
            "metadata": {"source": "test3"},
        },
    ]

    # Use sync method
    success = store.save_episodes(episodes, namespace="test")
    print(f"✓ Saved episodes: {success}")

    # Test 2: Load episodes using sync method
    print("\n2. Testing sync episode load...")
    loaded = store.load_episodes(namespace="test")
    print(f"✓ Loaded {len(loaded)} episodes")
    for ep in loaded[:2]:
        print(f"  - {ep['text'][:30]}... (c={ep['c']})")

    # Test 3: Metadata operations (sync)
    print("\n3. Testing sync metadata operations...")
    metadata = {"experiment": "test", "version": "1.0", "config": {"param1": 10}}
    success = store.save_metadata(metadata, "experiment_config", namespace="test_meta")
    print(f"✓ Saved metadata: {success}")

    loaded_meta = store.load_metadata("experiment_config", namespace="test_meta")
    print(f"✓ Loaded metadata: {loaded_meta}")

    # Test 4: Vector operations (sync)
    print("\n4. Testing sync vector operations...")
    vectors = np.random.rand(5, 384).astype(np.float32)
    vector_metadata = [{"id": f"vec_{i}", "text": f"Vector {i}"} for i in range(5)]
    success = store.save_vectors(vectors, vector_metadata, namespace="test_vectors")
    print(f"✓ Saved vectors: {success}")

    indices, distances = store.search_vectors(vectors[0], k=3, namespace="test_vectors")
    print(f"✓ Vector search found {len(indices)} results")

    # Test 5: Graph operations (sync)
    print("\n5. Testing sync graph operations...")
    graph_data = {
        "nodes": {
            "node1": {"type": "concept", "attributes": {"name": "Node 1"}},
            "node2": {"type": "concept", "attributes": {"name": "Node 2"}},
            "node3": {"type": "concept", "attributes": {"name": "Node 3"}},
        },
        "edges": [
            {
                "source": "node1",
                "target": "node2",
                "type": "related",
                "attributes": {"weight": 0.5},
            },
            {
                "source": "node2",
                "target": "node3",
                "type": "related",
                "attributes": {"weight": 0.7},
            },
        ],
    }
    success = store.save_graph(graph_data, "test_graph", namespace="graphs")
    print(f"✓ Saved graph: {success}")

    loaded_graph = store.load_graph("test_graph", namespace="graphs")
    if loaded_graph:
        print(
            f"✓ Loaded graph with {len(loaded_graph['nodes'])} nodes and {len(loaded_graph['edges'])} edges"
        )

    # Test 6: Namespace operations (sync)
    print("\n6. Testing sync namespace operations...")
    keys = store.list_keys("test")
    print(f"✓ Found {len(keys)} keys in 'test' namespace")

    # Cleanup
    store.close()
    print("\n=== All tests completed successfully! ===")


if __name__ == "__main__":
    main()
