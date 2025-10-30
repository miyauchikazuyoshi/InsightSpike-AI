#!/usr/bin/env python3
"""
DataStore Performance Tests
===========================

Benchmarks for DataStore implementation performance.
"""

import asyncio
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.insightspike.implementations.datastore.filesystem_store import (
    FileSystemDataStore,
)
from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore


class PerformanceTester:
    """Run performance benchmarks for DataStore implementations"""

    def __init__(self):
        self.results = {}

    def generate_episodes(
        self, count: int, text_length: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate test episodes"""
        episodes = []
        for i in range(count):
            text = f"Test episode {i}: " + " ".join(
                [f"word{j}" for j in range(text_length)]
            )
            vec = np.random.rand(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize

            episodes.append(
                {
                    "text": text,
                    "vec": vec,
                    "c": random.random(),
                    "metadata": {
                        "index": i,
                        "category": f"cat_{i % 10}",
                        "timestamp": time.time(),
                    },
                }
            )
        return episodes

    async def benchmark_save_episodes(
        self, datastore, episodes: List[Dict[str, Any]], name: str
    ):
        """Benchmark episode saving"""
        print(f"\n{name} - Save {len(episodes)} episodes:")

        # Test different batch sizes
        batch_sizes = [1, 10, 100, 1000]

        for batch_size in batch_sizes:
            if batch_size > len(episodes):
                continue

            start_time = time.time()

            # Save in batches
            for i in range(0, len(episodes), batch_size):
                batch = episodes[i : i + batch_size]
                datastore.save_episodes(batch)

            elapsed = time.time() - start_time
            eps = len(episodes) / elapsed

            print(f"  Batch size {batch_size}: {elapsed:.2f}s ({eps:.0f} episodes/sec)")

            self.results[f"{name}_save_batch_{batch_size}"] = {
                "episodes": len(episodes),
                "time": elapsed,
                "eps": eps,
            }

    async def benchmark_vector_search(
        self, datastore, query_count: int, k: int, name: str
    ):
        """Benchmark vector search"""
        print(f"\n{name} - Vector search ({query_count} queries, k={k}):")

        # Generate query vectors
        query_vectors = []
        for _ in range(query_count):
            vec = np.random.rand(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            query_vectors.append(vec)

        # Warm up
        _ = datastore.search_episodes_by_vector(query_vectors[0], k=k)

        # Benchmark
        start_time = time.time()

        results = []
        for vec in query_vectors:
            result = datastore.search_episodes_by_vector(vec, k=k)
            results.append(result)

        elapsed = time.time() - start_time
        qps = query_count / elapsed

        print(f"  Total time: {elapsed:.2f}s ({qps:.0f} queries/sec)")
        print(f"  Avg query time: {elapsed/query_count*1000:.1f}ms")

        self.results[f"{name}_search_k{k}"] = {
            "queries": query_count,
            "k": k,
            "time": elapsed,
            "qps": qps,
            "avg_ms": elapsed / query_count * 1000,
        }

    async def benchmark_text_search(self, datastore, query_count: int, name: str):
        """Benchmark text search"""
        print(f"\n{name} - Text search ({query_count} queries):")

        # Generate queries
        queries = [f"word{random.randint(0, 99)}" for _ in range(query_count)]

        # Benchmark
        start_time = time.time()

        results = []
        for query in queries:
            result = datastore.search_episodes_by_text(query)
            results.append(result)

        elapsed = time.time() - start_time
        qps = query_count / elapsed

        print(f"  Total time: {elapsed:.2f}s ({qps:.0f} queries/sec)")
        print(f"  Avg query time: {elapsed/query_count*1000:.1f}ms")

        self.results[f"{name}_text_search"] = {
            "queries": query_count,
            "time": elapsed,
            "qps": qps,
            "avg_ms": elapsed / query_count * 1000,
        }

    async def benchmark_load_all(self, datastore, name: str):
        """Benchmark loading all episodes"""
        print(f"\n{name} - Load all episodes:")

        start_time = time.time()
        episodes = datastore.load_episodes()
        elapsed = time.time() - start_time

        count = len(episodes)
        eps = count / elapsed if elapsed > 0 else 0

        print(f"  Loaded {count} episodes in {elapsed:.2f}s ({eps:.0f} episodes/sec)")

        self.results[f"{name}_load_all"] = {
            "episodes": count,
            "time": elapsed,
            "eps": eps,
        }

    async def run_benchmarks(self):
        """Run all benchmarks"""
        print("=== DataStore Performance Benchmarks ===")

        # Generate test data
        episode_counts = [100, 1000, 10000]

        for count in episode_counts:
            print(f"\n\n--- Testing with {count} episodes ---")
            episodes = self.generate_episodes(count)

            # Test SQLite DataStore
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "test.db"
                sqlite_store = SQLiteDataStore(str(db_path))

                await self.benchmark_save_episodes(
                    sqlite_store, episodes, f"SQLite_{count}"
                )
                await self.benchmark_vector_search(
                    sqlite_store, 100, k=10, name=f"SQLite_{count}"
                )
                await self.benchmark_text_search(
                    sqlite_store, 100, name=f"SQLite_{count}"
                )
                await self.benchmark_load_all(sqlite_store, name=f"SQLite_{count}")

                # Get stats
                stats = sqlite_store.get_stats()
                print(f"\nSQLite stats: {stats}")

            # Test FileSystem DataStore
            with tempfile.TemporaryDirectory() as tmpdir:
                fs_store = FileSystemDataStore(tmpdir)

                await self.benchmark_save_episodes(
                    fs_store, episodes, f"FileSystem_{count}"
                )
                await self.benchmark_vector_search(
                    fs_store, 100, k=10, name=f"FileSystem_{count}"
                )
                # Note: FileSystem doesn't support text search
                await self.benchmark_load_all(fs_store, name=f"FileSystem_{count}")

        # Save results
        self.save_results()

    def save_results(self):
        """Save benchmark results"""
        output_path = Path("tests/performance/results/datastore_benchmark_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n\nResults saved to: {output_path}")

        # Print summary
        print("\n=== Summary ===")
        print("\nSave performance (episodes/sec):")
        for key, value in self.results.items():
            if "_save_" in key and "_batch_100" in key:
                print(f"  {key}: {value['eps']:.0f}")

        print("\nVector search performance (queries/sec):")
        for key, value in self.results.items():
            if "_search_k" in key:
                print(f"  {key}: {value['qps']:.0f} ({value['avg_ms']:.1f}ms avg)")


async def main():
    """Run performance tests"""
    tester = PerformanceTester()
    await tester.run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
