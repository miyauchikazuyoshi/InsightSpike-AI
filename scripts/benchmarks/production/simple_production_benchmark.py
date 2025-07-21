#!/usr/bin/env python3
"""
Simple Production Benchmark
===========================

Lightweight production benchmarks focusing on core DataStore performance.
"""

import time
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
import numpy as np


def benchmark_datastore_operations():
    """Benchmark core DataStore operations"""
    print("=== Simple Production Benchmark ===\n")
    
    # Setup
    db_path = "./data/sqlite/simple_benchmark.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    datastore = SQLiteDataStore(db_path)
    results = {}
    
    # 1. Episode Save Performance
    print("1. Episode Save Performance")
    episode_counts = [10, 100, 1000]
    
    for count in episode_counts:
        episodes = []
        for i in range(count):
            vec = np.random.rand(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            
            episodes.append({
                'text': f"Test episode {i}: This is a sample text for benchmarking purposes.",
                'vec': vec,
                'c': 0.5,
                'metadata': {'index': i, 'category': f'cat_{i % 5}'}
            })
        
        start = time.time()
        success = datastore.save_episodes(episodes)
        elapsed = time.time() - start
        
        if success:
            rate = count / elapsed
            print(f"  {count} episodes: {elapsed:.3f}s ({rate:.0f} episodes/sec)")
            results[f'save_{count}'] = {'time': elapsed, 'rate': rate}
    
    # 2. Vector Search Performance
    print("\n2. Vector Search Performance")
    query_counts = [10, 50, 100]
    k_values = [5, 10, 20]
    
    for queries in query_counts:
        for k in k_values:
            # Generate query vectors
            times = []
            for _ in range(queries):
                query_vec = np.random.rand(384).astype(np.float32)
                query_vec = query_vec / np.linalg.norm(query_vec)
                
                start = time.time()
                results_list = datastore.search_episodes_by_vector(query_vec, k=k)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            qps = 1 / avg_time if avg_time > 0 else 0
            
            print(f"  {queries} queries, k={k}: avg {avg_time*1000:.1f}ms ({qps:.0f} queries/sec)")
            results[f'search_{queries}_k{k}'] = {'avg_ms': avg_time*1000, 'qps': qps}
    
    # 3. Text Search Performance
    print("\n3. Text Search Performance")
    text_queries = ["test", "episode", "sample", "benchmark", "purpose"]
    
    times = []
    for query in text_queries * 10:  # 50 queries total
        start = time.time()
        results_list = datastore.search_episodes_by_text(query)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    qps = 1 / avg_time if avg_time > 0 else 0
    
    print(f"  50 text searches: avg {avg_time*1000:.1f}ms ({qps:.0f} queries/sec)")
    results['text_search'] = {'avg_ms': avg_time*1000, 'qps': qps}
    
    # 4. Database Statistics
    print("\n4. Database Statistics")
    stats = datastore.get_stats()
    
    print(f"  Total episodes: {stats['episodes']['total']}")
    print(f"  Database size: {stats['db_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Size per episode: {stats['db_size_bytes'] / stats['episodes']['total'] / 1024:.1f} KB")
    
    results['db_stats'] = {
        'episodes': stats['episodes']['total'],
        'size_mb': stats['db_size_bytes'] / 1024 / 1024,
        'kb_per_episode': stats['db_size_bytes'] / stats['episodes']['total'] / 1024
    }
    
    # 5. Concurrent Access Test
    print("\n5. Concurrent Access Simulation")
    operations = 100
    start = time.time()
    
    for i in range(operations):
        if i % 3 == 0:
            # Save operation
            vec = np.random.rand(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            datastore.save_episodes([{
                'text': f'Concurrent test {i}',
                'vec': vec,
                'c': 0.5
            }])
        else:
            # Search operation
            query_vec = np.random.rand(384).astype(np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)
            datastore.search_episodes_by_vector(query_vec, k=5)
    
    elapsed = time.time() - start
    ops_per_sec = operations / elapsed
    
    print(f"  {operations} mixed operations: {elapsed:.2f}s ({ops_per_sec:.0f} ops/sec)")
    results['concurrent'] = {'operations': operations, 'time': elapsed, 'ops_per_sec': ops_per_sec}
    
    # Save results
    output_path = Path("benchmarks/results/simple_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Episode save rate: {results['save_1000']['rate']:.0f} episodes/sec")
    print(f"Vector search rate: {results['search_100_k10']['qps']:.0f} queries/sec")
    print(f"Text search rate: {results['text_search']['qps']:.0f} queries/sec")
    print(f"Mixed operations: {results['concurrent']['ops_per_sec']:.0f} ops/sec")
    print(f"Storage efficiency: {results['db_stats']['kb_per_episode']:.1f} KB/episode")


if __name__ == "__main__":
    benchmark_datastore_operations()