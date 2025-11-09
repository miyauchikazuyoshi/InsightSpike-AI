#!/usr/bin/env python3
"""
Memory Usage Tests
==================

Compare memory usage between in-memory and DataStore approaches.
"""

import asyncio
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
pytest.importorskip("psutil")
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
from src.insightspike.implementations.layers.layer2_memory_manager import (
    L2MemoryManager,
)
from src.insightspike.implementations.layers.layer2_working_memory import (
    L2WorkingMemoryManager,
)


class MemoryProfiler:
    """Profile memory usage of different approaches"""

    def __init__(self):
        self.process = psutil.Process()
        self.results = {}

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        gc.collect()  # Force garbage collection
        return self.process.memory_info().rss / 1024 / 1024

    def generate_episodes(self, count: int) -> List[Dict[str, Any]]:
        """Generate test episodes"""
        episodes = []
        for i in range(count):
            text = f"Episode {i}: " + " ".join([f"content_{j}" for j in range(50)])
            vec = np.random.rand(384).astype(np.float32)

            episodes.append(
                {"text": text, "vec": vec, "c": 0.5, "metadata": {"index": i}}
            )
        return episodes

    async def profile_in_memory_approach(self, episode_count: int):
        """Profile traditional in-memory approach"""
        print(f"\n--- In-Memory Approach ({episode_count} episodes) ---")

        initial_mem = self.get_memory_usage()
        print(f"Initial memory: {initial_mem:.1f} MB")

        # Create in-memory manager
        from src.insightspike.config.presets import ConfigPresets

        config = ConfigPresets.get_experiment_config()
        memory_manager = L2MemoryManager(config)

        # Generate and load all episodes
        episodes = self.generate_episodes(episode_count)

        start_time = time.time()

        # Add all episodes (simulating load_episodes)
        for ep in episodes:
            memory_manager.add_episode(ep["text"], ep["vec"], ep["c"])

        loaded_mem = self.get_memory_usage()
        load_time = time.time() - start_time

        print(
            f"After loading: {loaded_mem:.1f} MB (Δ{loaded_mem - initial_mem:.1f} MB)"
        )
        print(f"Load time: {load_time:.2f}s")

        # Simulate some operations
        query_vec = np.random.rand(384).astype(np.float32)

        start_time = time.time()
        for _ in range(100):
            _ = memory_manager.find_similar_episodes(query_vec, k=10)
        query_time = time.time() - start_time

        peak_mem = self.get_memory_usage()

        self.results[f"in_memory_{episode_count}"] = {
            "initial_mb": initial_mem,
            "loaded_mb": loaded_mem,
            "peak_mb": peak_mem,
            "delta_mb": peak_mem - initial_mem,
            "load_time": load_time,
            "query_time": query_time,
        }

        # Cleanup
        del memory_manager
        del episodes
        gc.collect()

    async def profile_datastore_approach(self, episode_count: int):
        """Profile DataStore approach with working memory"""
        print(f"\n--- DataStore Approach ({episode_count} episodes) ---")

        initial_mem = self.get_memory_usage()
        print(f"Initial memory: {initial_mem:.1f} MB")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create DataStore and working memory
            db_path = Path(tmpdir) / "test.db"
            datastore = SQLiteDataStore(str(db_path))
            working_memory = L2WorkingMemoryManager(datastore)

            # Generate episodes
            episodes = self.generate_episodes(episode_count)

            start_time = time.time()

            # Save episodes to DataStore (not keeping in memory)
            batch_size = 1000
            for i in range(0, len(episodes), batch_size):
                batch = episodes[i : i + batch_size]
                datastore.save_episodes(batch)

            # Clear episodes from memory
            del episodes
            gc.collect()

            loaded_mem = self.get_memory_usage()
            load_time = time.time() - start_time

            print(
                f"After saving: {loaded_mem:.1f} MB (Δ{loaded_mem - initial_mem:.1f} MB)"
            )
            print(f"Save time: {load_time:.2f}s")

            # Simulate operations with working memory
            query_vec = np.random.rand(384).astype(np.float32)

            start_time = time.time()
            for _ in range(100):
                # This loads only needed episodes
                _ = working_memory.find_similar(query_vec, k=10)
            query_time = time.time() - start_time

            peak_mem = self.get_memory_usage()

            # Check working memory size
            working_set_size = len(working_memory._working_set)
            print(f"Working set size: {working_set_size} episodes")

            self.results[f"datastore_{episode_count}"] = {
                "initial_mb": initial_mem,
                "loaded_mb": loaded_mem,
                "peak_mb": peak_mem,
                "delta_mb": peak_mem - initial_mem,
                "load_time": load_time,
                "query_time": query_time,
                "working_set_size": working_set_size,
            }

    async def run_comparison(self):
        """Run memory comparison tests"""
        print("=== Memory Usage Comparison ===")

        episode_counts = [1000, 10000, 50000]

        for count in episode_counts:
            await self.profile_in_memory_approach(count)
            await asyncio.sleep(1)  # Let memory settle

            await self.profile_datastore_approach(count)
            await asyncio.sleep(1)

        # Save and display results
        self.save_results()
        self.print_comparison()

    def save_results(self):
        """Save results to file"""
        output_path = Path("tests/performance/results/memory_usage_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def print_comparison(self):
        """Print comparison summary"""
        print("\n\n=== Memory Usage Summary ===")
        print("\nMemory Delta (Peak - Initial):")

        for key in sorted(self.results.keys()):
            value = self.results[key]
            print(f"  {key}: {value['delta_mb']:.1f} MB")

        print("\nMemory Savings:")
        counts = [1000, 10000, 50000]
        for count in counts:
            in_mem_key = f"in_memory_{count}"
            ds_key = f"datastore_{count}"

            if in_mem_key in self.results and ds_key in self.results:
                in_mem = self.results[in_mem_key]["delta_mb"]
                ds_mem = self.results[ds_key]["delta_mb"]
                savings = (1 - ds_mem / in_mem) * 100 if in_mem > 0 else 0

                print(f"  {count} episodes: {savings:.1f}% memory saved")


async def main():
    """Run memory profiling"""
    profiler = MemoryProfiler()
    await profiler.run_comparison()


if __name__ == "__main__":
    asyncio.run(main())
