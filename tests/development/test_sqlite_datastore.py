"""
Test SQLiteDataStore Implementation
==================================

Simple test script to verify SQLiteDataStore functionality.
"""

import asyncio
import os
import importlib.util
import pytest

import numpy as np

if importlib.util.find_spec("pytest_asyncio") is None:
    pytest.skip("pytest_asyncio not available; skipping async datastore test", allow_module_level=True)

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore


async def test_sqlite_datastore():
    """Test SQLiteDataStore functionality"""

    # Initialize datastore
    db_path = "./test_data/test_datastore.db"
    os.makedirs("./test_data", exist_ok=True)

    print("=== Testing SQLiteDataStore ===")
    store = SQLiteDataStore(db_path, vector_dim=384)

    # Test 1: Add episodes
    print("\n1. Testing episode operations...")
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

    # Add episodes using async method
    episode_ids = await store.batch_add_episodes(episodes, namespace="test")
    print(f"Added {len(episode_ids)} episodes: {episode_ids}")

    # Test 2: Search by vector
    print("\n2. Testing vector search...")
    query_vec = np.random.rand(384).astype(np.float32)
    results = await store.search_episodes_by_vector(
        query_vec, k=2, threshold=0.0, namespace="test"
    )
    print(f"Found {len(results)} similar episodes:")
    for r in results:
        print(f"  - {r['text'][:50]}... (c={r['c']})")

    # Test 3: Graph operations
    print("\n3. Testing graph operations...")
    edges_to_add = [
        ("node1", "node2", {"weight": 0.5}),
        ("node2", "node3", {"weight": 0.7}),
        ("node1", "node3", {"weight": 0.3}),
    ]
    success = await store.update_graph_edges(edges_to_add, [], namespace="test_graph")
    print(f"Added graph edges: {success}")

    # Get neighbors
    neighbors = await store.get_graph_neighbors("node1", hop=2, namespace="test_graph")
    print(f"Neighbors of node1: {neighbors}")

    # Test 4: Metadata operations
    print("\n4. Testing metadata operations...")
    metadata = {"experiment": "test", "version": "1.0", "config": {"param1": 10}}
    store.save_metadata(metadata, "experiment_config", namespace="test_meta")
    loaded = store.load_metadata("experiment_config", namespace="test_meta")
    print(f"Saved and loaded metadata: {loaded}")

    # Test 5: Stats
    print("\n5. Getting datastore stats...")
    stats = store.get_stats()
    print(f"Stats: {stats}")

    # Test 6: Streaming
    print("\n6. Testing streaming operations...")
    print("Streaming episodes:")
    count = 0
    async for batch in store.stream_episodes(batch_size=2, namespace="test"):
        count += len(batch)
        print(f"  Got batch of {len(batch)} episodes")
    print(f"Total streamed: {count} episodes")

    # Cleanup
    store.close()
    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(test_sqlite_datastore())
