#!/usr/bin/env python3
"""
DataStore Integration Tests
===========================

End-to-end tests for DataStore-centric architecture.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.insightspike.config.presets import ConfigPresets
from src.insightspike.implementations.agents.datastore_agent import DataStoreMainAgent
from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore


class IntegrationTester:
    """Run integration tests for DataStore architecture"""

    async def test_full_pipeline(self):
        """Test complete pipeline from input to insight detection"""
        print("=== DataStore Integration Test ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Setup
            db_path = Path(tmpdir) / "test.db"
            datastore = SQLiteDataStore(str(db_path))
            config = ConfigPresets.get_experiment_config()
            agent = DataStoreMainAgent(datastore, config)

            print("\n1. Testing initial state...")
            stats = datastore.get_stats()
            assert stats["episodes"]["total"] == 0
            print("✓ DataStore initialized empty")

            # 2. Process first input
            print("\n2. Processing first input...")
            response1 = await agent.process_async(
                "Quantum entanglement demonstrates non-local correlations"
            )

            assert response1 is not None
            assert "response" in response1
            stats = datastore.get_stats()
            assert stats["episodes"]["total"] > 0
            print(
                f"✓ First input processed, {stats['episodes']['total']} episodes stored"
            )

            # 3. Process related input
            print("\n3. Processing related input...")
            response2 = await agent.process_async(
                "Bell inequalities test quantum entanglement experimentally"
            )

            assert response2 is not None
            initial_count = stats["episodes"]["total"]
            stats = datastore.get_stats()
            assert stats["episodes"]["total"] > initial_count
            print(
                f"✓ Related input processed, {stats['episodes']['total']} total episodes"
            )

            # 4. Test insight detection
            print("\n4. Testing insight detection...")
            response3 = await agent.process_async(
                "Consciousness might emerge from quantum entanglement in microtubules"
            )

            assert response3 is not None
            if "spike_detected" in response3:
                print(
                    f"✓ Insight spike detected: {response3.get('spike_confidence', 0):.2f} confidence"
                )
            else:
                print("✓ Response generated (no spike detected)")

            # 5. Test persistence
            print("\n5. Testing persistence...")
            # Create new agent with same datastore
            agent2 = DataStoreMainAgent(datastore, config)

            # Should find similar content
            test_vec = np.random.rand(384).astype(np.float32)
            similar = datastore.search_episodes_by_vector(test_vec, k=5)
            assert len(similar) > 0
            print(f"✓ Persistence verified, found {len(similar)} similar episodes")

            # 6. Test working memory
            print("\n6. Testing working memory...")
            working_set_size = len(agent.memory_manager._working_set)
            print(f"✓ Working set contains {working_set_size} episodes")

            # 7. Test graph operations
            print("\n7. Testing graph operations...")
            graph_data = {
                "nodes": {
                    "concept1": {"type": "concept", "attributes": {"text": "quantum"}},
                    "concept2": {
                        "type": "concept",
                        "attributes": {"text": "entanglement"},
                    },
                },
                "edges": [
                    {"source": "concept1", "target": "concept2", "type": "related"}
                ],
            }

            success = datastore.save_graph(graph_data, "test_graph")
            assert success

            loaded_graph = datastore.load_graph("test_graph")
            assert loaded_graph is not None
            assert len(loaded_graph["nodes"]) == 2
            print("✓ Graph operations working")

            # 8. Final statistics
            print("\n8. Final statistics:")
            final_stats = datastore.get_stats()
            print(f"  Episodes: {final_stats['episodes']['total']}")
            print(
                f"  Namespaces: {list(final_stats['episodes']['by_namespace'].keys())}"
            )
            print(f"  DB Size: {final_stats['db_size_bytes'] / 1024:.1f} KB")

            print("\n✅ All integration tests passed!")

    async def test_error_handling(self):
        """Test error handling and recovery"""
        print("\n=== Error Handling Tests ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            datastore = SQLiteDataStore(str(db_path))
            agent = DataStoreMainAgent(datastore)

            # Test with invalid input
            print("\n1. Testing empty input...")
            response = await agent.process_async("")
            assert response is not None
            print("✓ Handled empty input gracefully")

            # Test with very long input
            print("\n2. Testing very long input...")
            long_text = " ".join(["word"] * 10000)
            response = await agent.process_async(long_text)
            assert response is not None
            print("✓ Handled long input gracefully")

            # Test concurrent access
            print("\n3. Testing concurrent access...")
            tasks = []
            for i in range(10):
                task = agent.process_async(f"Concurrent test {i}")
                tasks.append(task)

            responses = await asyncio.gather(*tasks)
            assert all(r is not None for r in responses)
            print("✓ Handled concurrent requests")

            print("\n✅ Error handling tests passed!")

    async def test_migration(self):
        """Test data migration from legacy format"""
        print("\n=== Migration Test ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy data
            legacy_dir = Path(tmpdir) / "legacy"
            legacy_dir.mkdir()

            legacy_episodes = [
                {
                    "text": "Legacy episode 1",
                    "vector": np.random.rand(384).tolist(),
                    "c_value": 0.5,
                    "metadata": {"source": "legacy"},
                },
                {
                    "text": "Legacy episode 2",
                    "vec": np.random.rand(384).tolist(),
                    "c": 0.7,
                    "metadata": {"source": "legacy"},
                },
            ]

            with open(legacy_dir / "episodes.json", "w") as f:
                json.dump({"episodes": legacy_episodes}, f)

            # Run migration
            from scripts.migrate_to_sqlite import DataMigrator

            db_path = Path(tmpdir) / "migrated.db"
            migrator = DataMigrator(str(legacy_dir), str(db_path))

            episodes = migrator.load_legacy_episodes(str(legacy_dir / "episodes.json"))
            assert len(episodes) == 2
            print(f"✓ Loaded {len(episodes)} legacy episodes")

            await migrator.migrate_episodes(episodes, namespace="migrated")

            # Verify migration
            datastore = SQLiteDataStore(str(db_path))
            stats = datastore.get_stats()
            assert stats["episodes"]["total"] == 2
            print("✓ Episodes migrated successfully")

            # Check content
            all_episodes = datastore.load_episodes(namespace="migrated")
            assert len(all_episodes) == 2
            assert any("Legacy episode 1" in ep["text"] for ep in all_episodes)
            print("✓ Content preserved correctly")

            print("\n✅ Migration test passed!")


async def main():
    """Run all integration tests"""
    tester = IntegrationTester()

    # Run tests
    await tester.test_full_pipeline()
    await tester.test_error_handling()
    await tester.test_migration()

    print("\n\n🎉 All integration tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
