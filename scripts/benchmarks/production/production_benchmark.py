#!/usr/bin/env python3
"""
Production Environment Benchmark
================================

Comprehensive benchmarks for InsightSpike in production scenarios.
"""

import asyncio
import time
import statistics
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
from src.insightspike.implementations.agents.datastore_agent import DataStoreMainAgent
from src.insightspike.config.presets import ConfigPresets


class ProductionBenchmark:
    """Production environment benchmark suite"""
    
    def __init__(self, db_path: str = "./data/sqlite/benchmark.db"):
        self.db_path = db_path
        self.results = {
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
        
    def _get_system_info(self):
        """Get system information"""
        import platform
        import os
        
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            cpu_count = os.cpu_count()
            memory_gb = "N/A"
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': cpu_count,
            'memory_gb': memory_gb,
            'python_version': platform.python_version()
        }
    
    async def benchmark_knowledge_addition(self, agent: DataStoreMainAgent, count: int = 100):
        """Benchmark knowledge addition performance"""
        print(f"\n=== Knowledge Addition Benchmark ({count} items) ===")
        
        times = []
        test_texts = [
            f"Scientific fact {i}: {self._generate_test_text(i)}"
            for i in range(count)
        ]
        
        for i, text in enumerate(test_texts):
            start = time.time()
            # Use sync method (DataStoreMainAgent doesn't have async methods)
            agent.process(text)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{count} items...")
        
        # Calculate statistics
        stats = {
            'count': count,
            'total_time': sum(times),
            'avg_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'min_time': min(times),
            'max_time': max(times),
            'items_per_second': count / sum(times)
        }
        
        print(f"\nResults:")
        print(f"  Average time: {stats['avg_time']*1000:.1f}ms")
        print(f"  Throughput: {stats['items_per_second']:.1f} items/sec")
        
        return stats
    
    async def benchmark_question_processing(self, agent: DataStoreMainAgent, count: int = 50):
        """Benchmark question processing performance"""
        print(f"\n=== Question Processing Benchmark ({count} queries) ===")
        
        # First, ensure we have some knowledge
        print("  Preparing knowledge base...")
        for i in range(20):
            agent.process(f"Background knowledge {i}: {self._generate_test_text(i)}")
        
        times = []
        test_questions = [
            f"What is the relationship between concept {i} and {i+1}?"
            for i in range(count)
        ]
        
        for i, question in enumerate(test_questions):
            start = time.time()
            agent.process(question)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{count} questions...")
        
        # Calculate statistics
        stats = {
            'count': count,
            'total_time': sum(times),
            'avg_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'min_time': min(times),
            'max_time': max(times),
            'queries_per_second': count / sum(times),
            'p95_time': statistics.quantiles(times, n=20)[18] if count >= 20 else max(times),
            'p99_time': statistics.quantiles(times, n=100)[98] if count >= 100 else max(times)
        }
        
        print(f"\nResults:")
        print(f"  Average time: {stats['avg_time']*1000:.1f}ms")
        print(f"  P95 time: {stats['p95_time']*1000:.1f}ms")
        print(f"  Throughput: {stats['queries_per_second']:.1f} queries/sec")
        
        return stats
    
    async def benchmark_concurrent_operations(self, agent: DataStoreMainAgent, concurrent: int = 10):
        """Benchmark concurrent operations"""
        print(f"\n=== Concurrent Operations Benchmark ({concurrent} concurrent) ===")
        
        # Since DataStoreMainAgent is sync, we'll simulate concurrent operations
        # by measuring sequential operations and calculating theoretical concurrency
        
        start = time.time()
        for i in range(concurrent):
            if i % 2 == 0:
                # Knowledge addition
                agent.process(f"Concurrent knowledge {i}: {self._generate_test_text(i)}")
            else:
                # Question processing  
                agent.process(f"Concurrent question {i}?")
        elapsed = time.time() - start
        
        stats = {
            'concurrent_count': concurrent,
            'total_time': elapsed,
            'operations_per_second': concurrent / elapsed
        }
        
        print(f"\nResults:")
        print(f"  Total time: {stats['total_time']:.2f}s")
        print(f"  Throughput: {stats['operations_per_second']:.1f} ops/sec")
        
        return stats
    
    async def benchmark_memory_usage(self, agent: DataStoreMainAgent, episodes: int = 1000):
        """Benchmark memory usage with increasing data"""
        print(f"\n=== Memory Usage Benchmark ({episodes} episodes) ===")
        
        try:
            import psutil
            process = psutil.Process()
            # Initial memory
            initial_mem = process.memory_info().rss / 1024 / 1024  # MB
            print(f"  Initial memory: {initial_mem:.1f} MB")
        except ImportError:
            print("  Warning: psutil not installed, skipping memory benchmark")
            return {
                'episodes': episodes,
                'initial_memory_mb': 'N/A',
                'final_memory_mb': 'N/A',
                'memory_increase_mb': 'N/A',
                'memory_per_episode_kb': 'N/A',
                'memory_points': []
            }
        
        # Add episodes in batches
        batch_size = 100
        memory_points = []
        
        for i in range(0, episodes, batch_size):
            # Add batch
            for j in range(batch_size):
                if i + j < episodes:
                    agent.process(
                        f"Memory test {i+j}: {self._generate_test_text(i+j)}"
                    )
            
            # Measure memory
            current_mem = process.memory_info().rss / 1024 / 1024
            memory_points.append({
                'episodes': i + batch_size,
                'memory_mb': current_mem,
                'delta_mb': current_mem - initial_mem
            })
            
            print(f"  {i + batch_size} episodes: {current_mem:.1f} MB (Δ{current_mem - initial_mem:.1f} MB)")
        
        # Final stats
        final_mem = memory_points[-1]['memory_mb']
        stats = {
            'episodes': episodes,
            'initial_memory_mb': initial_mem,
            'final_memory_mb': final_mem,
            'memory_increase_mb': final_mem - initial_mem,
            'memory_per_episode_kb': (final_mem - initial_mem) * 1024 / episodes,
            'memory_points': memory_points
        }
        
        print(f"\nResults:")
        print(f"  Memory increase: {stats['memory_increase_mb']:.1f} MB")
        print(f"  Per episode: {stats['memory_per_episode_kb']:.1f} KB")
        
        return stats
    
    async def benchmark_database_size(self, datastore: SQLiteDataStore, episodes: int = 1000):
        """Benchmark database size growth"""
        print(f"\n=== Database Size Benchmark ===")
        
        import os
        
        # Get initial size
        initial_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        # Get stats
        stats = datastore.get_stats()
        
        final_size = stats.get('db_size_bytes', 0)
        
        db_stats = {
            'episodes': stats['episodes']['total'],
            'initial_size_mb': initial_size / 1024 / 1024,
            'final_size_mb': final_size / 1024 / 1024,
            'size_per_episode_kb': (final_size / stats['episodes']['total'] / 1024) if stats['episodes']['total'] > 0 else 0
        }
        
        print(f"  Database size: {db_stats['final_size_mb']:.1f} MB")
        print(f"  Episodes: {db_stats['episodes']}")
        print(f"  Per episode: {db_stats['size_per_episode_kb']:.1f} KB")
        
        return db_stats
    
    def _generate_test_text(self, index: int) -> str:
        """Generate test text with some variety"""
        templates = [
            "The quantum mechanics principle states that particles exhibit wave-particle duality",
            "Neural networks learn through backpropagation of error gradients",
            "Evolution operates through natural selection of favorable traits",
            "Thermodynamics governs the flow of energy in physical systems",
            "Information theory provides the mathematical foundation for communication"
        ]
        return f"{templates[index % len(templates)]} (variation {index})"
    
    async def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("=== InsightSpike Production Benchmarks ===")
        print(f"System: {self.results['system_info']['platform']}")
        print(f"CPU: {self.results['system_info']['cpu_count']} cores")
        print(f"Memory: {self.results['system_info']['memory_gb']} GB")
        
        # Setup
        import os
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        datastore = SQLiteDataStore(self.db_path)
        config = ConfigPresets.get_experiment_config()
        agent = DataStoreMainAgent(datastore, config)
        
        # Run benchmarks
        self.results['benchmarks']['knowledge_addition'] = await self.benchmark_knowledge_addition(agent, count=100)
        self.results['benchmarks']['question_processing'] = await self.benchmark_question_processing(agent, count=50)
        self.results['benchmarks']['concurrent_operations'] = await self.benchmark_concurrent_operations(agent)
        self.results['benchmarks']['memory_usage'] = await self.benchmark_memory_usage(agent)
        self.results['benchmarks']['database_size'] = await self.benchmark_database_size(datastore)
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save benchmark results"""
        output_path = Path("benchmarks/results/production_benchmark_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        b = self.results['benchmarks']
        
        print("\nPerformance Metrics:")
        print(f"  Knowledge Addition: {b['knowledge_addition']['items_per_second']:.1f} items/sec")
        print(f"  Question Processing: {b['question_processing']['queries_per_second']:.1f} queries/sec")
        print(f"  Concurrent Operations: {b['concurrent_operations']['operations_per_second']:.1f} ops/sec")
        
        print("\nResource Usage:")
        print(f"  Memory per 1000 episodes: {b['memory_usage']['memory_increase_mb']:.1f} MB")
        print(f"  Database size per episode: {b['database_size']['size_per_episode_kb']:.1f} KB")
        
        print("\nLatency (Question Processing):")
        print(f"  Average: {b['question_processing']['avg_time']*1000:.1f}ms")
        print(f"  P95: {b['question_processing']['p95_time']*1000:.1f}ms")


async def main():
    """Run production benchmarks"""
    parser = argparse.ArgumentParser(description="Production environment benchmarks")
    parser.add_argument("--db-path", default="./data/sqlite/benchmark.db", help="Database path")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks")
    
    args = parser.parse_args()
    
    benchmark = ProductionBenchmark(args.db_path)
    
    if args.quick:
        # Quick benchmarks with smaller counts
        print("Running quick benchmarks...")
        datastore = SQLiteDataStore(args.db_path)
        agent = DataStoreMainAgent(datastore)
        
        await benchmark.benchmark_knowledge_addition(agent, count=20)
        await benchmark.benchmark_question_processing(agent, count=10)
    else:
        # Full benchmark suite
        await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())