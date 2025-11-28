"""
Standalone Test for SQLiteDataStore Implementation
=================================================

Direct test without importing from the full package.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite
import faiss
import numpy as np
import importlib.util
import pytest

if importlib.util.find_spec("pytest_asyncio") is None:
    pytest.skip("pytest_asyncio not available; skipping async datastore test", allow_module_level=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.insightspike.core.base.async_datastore import AsyncDataStore

# Import only the minimal dependencies
from src.insightspike.core.base.datastore import DataStore, VectorIndex

# Import the SQLiteDataStore
from src.insightspike.implementations.datastore.sqlite_store import (
    FAISSVectorIndex,
    SQLiteDataStore,
)


async def test_sqlite_datastore():
    """Test SQLiteDataStore functionality"""

    # Initialize datastore
    db_path = "./test_data/test_datastore.db"
    os.makedirs("./test_data", exist_ok=True)

    # Remove existing database for clean test
    if os.path.exists(db_path):
        os.remove(db_path)

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
    print(f"✓ Added {len(episode_ids)} episodes: {episode_ids[:2]}...")

    # Test 2: Search by vector
    print("\n2. Testing vector search...")
    query_vec = np.random.rand(384).astype(np.float32)
    results = await store.search_episodes_by_vector(
        query_vec, k=2, threshold=0.0, namespace="test"
    )
    print(f"✓ Found {len(results)} similar episodes:")
    for r in results:
        print(f"  - {r['text'][:50]}... (c={r['c']})")

    # Test 3: Get episodes by IDs
    print("\n3. Testing get by IDs...")
    retrieved = await store.get_episodes_by_ids(episode_ids[:2], namespace="test")
    print(f"✓ Retrieved {len(retrieved)} episodes by ID")

    # Test 4: Update episode
    print("\n4. Testing episode update...")
    updates = {"c": 0.95, "metadata": {"source": "test1", "updated": True}}
    success = await store.update_episode(episode_ids[0], updates, namespace="test")
    print(f"✓ Updated episode: {success}")

    # Verify update
    updated = await store.get_episodes_by_ids([episode_ids[0]], namespace="test")
    if updated:
        print(f"  - New c value: {updated[0]['c']}")
        print(f"  - Metadata: {updated[0]['metadata']}")

    # Test 5: Graph operations
    print("\n5. Testing graph operations...")
    edges_to_add = [
        ("node1", "node2", {"weight": 0.5}),
        ("node2", "node3", {"weight": 0.7}),
        ("node1", "node3", {"weight": 0.3}),
    ]
    success = await store.update_graph_edges(edges_to_add, [], namespace="test_graph")
    print(f"✓ Added graph edges: {success}")

    # Get neighbors
    neighbors = await store.get_graph_neighbors("node1", hop=2, namespace="test_graph")
    print(f"✓ Neighbors of node1: {neighbors}")

    # Get subgraph
    subgraph = await store.get_subgraph(
        ["node1", "node2", "node3"], namespace="test_graph"
    )
    print(
        f"✓ Subgraph has {len(subgraph['nodes'])} nodes and {len(subgraph['edges'])} edges"
    )

    # Test 6: Metadata operations
    print("\n6. Testing metadata operations...")
    metadata = {"experiment": "test", "version": "1.0", "config": {"param1": 10}}
    store.save_metadata(metadata, "experiment_config", namespace="test_meta")
    loaded = store.load_metadata("experiment_config", namespace="test_meta")
    print(f"✓ Saved and loaded metadata: {loaded}")

    # Test 7: Batch vector search
    print("\n7. Testing batch vector search...")
    query_vectors = np.random.rand(3, 384).astype(np.float32)
    batch_results = await store.search_vectors_batch(
        query_vectors, k=2, namespace="test"
    )
    print(f"✓ Batch search returned {len(batch_results)} results")

    # Test 8: Streaming
    print("\n8. Testing streaming operations...")
    count = 0
    async for batch in store.stream_episodes(batch_size=2, namespace="test"):
        count += len(batch)
        print(f"  - Got batch of {len(batch)} episodes")
    print(f"✓ Total streamed: {count} episodes")

    # Test 9: Stats
    print("\n9. Getting datastore stats...")
    stats = store.get_stats()
    print(f"✓ Stats:")
    print(f"  - Episodes: {stats['episodes']}")
    print(f"  - Graphs: {stats['graphs']}")
    print(f"  - Indices: {stats['indices']}")
    if "db_size_bytes" in stats:
        print(f"  - DB size: {stats['db_size_bytes']} bytes")

    # Test 10: Namespace operations
    print("\n10. Testing namespace operations...")
    keys = store.list_keys("test")
    print(f"✓ Found {len(keys)} keys in 'test' namespace")

    # Test sync methods
    print("\n11. Testing sync methods...")
    sync_episodes = store.load_episodes("test")
    print(f"✓ Loaded {len(sync_episodes)} episodes via sync method")

    # Save and search vectors
    vectors = np.random.rand(5, 384).astype(np.float32)
    vector_metadata = [{"id": f"vec_{i}", "text": f"Vector {i}"} for i in range(5)]
    store.save_vectors(vectors, vector_metadata, namespace="test_vectors")

    indices, distances = store.search_vectors(vectors[0], k=3, namespace="test_vectors")
    print(f"✓ Vector search found {len(indices)} results")

    # Cleanup
    store.close()
    print("\n=== All tests completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(test_sqlite_datastore())
