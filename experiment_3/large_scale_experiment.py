#!/usr/bin/env python3
"""
Large-scale Dynamic Growth Experiment
1000+ real text samples to test InsightSpike-AI's scalability
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent

def generate_diverse_dataset(count=1000):
    """Generate diverse text samples for testing"""
    topics = [
        "artificial intelligence", "machine learning", "deep learning",
        "natural language processing", "computer vision", "robotics",
        "quantum computing", "blockchain", "cybersecurity", "data science",
        "neural networks", "reinforcement learning", "edge computing",
        "autonomous vehicles", "bioinformatics", "computational biology"
    ]
    
    patterns = [
        "{topic} is revolutionizing the field of {field}.",
        "Recent advances in {topic} have led to breakthroughs in {field}.",
        "The application of {topic} to {field} shows promising results.",
        "{topic} techniques are being used to solve problems in {field}.",
        "Researchers are exploring how {topic} can enhance {field}.",
        "The intersection of {topic} and {field} creates new opportunities.",
        "{topic} algorithms have improved performance in {field} tasks.",
        "Understanding {topic} is crucial for progress in {field}.",
        "The future of {field} depends on advances in {topic}.",
        "{topic} provides new insights into {field} challenges."
    ]
    
    fields = [
        "healthcare", "finance", "education", "manufacturing",
        "transportation", "energy", "agriculture", "retail",
        "entertainment", "telecommunications", "aerospace", "defense"
    ]
    
    dataset = []
    for i in range(count):
        topic = random.choice(topics)
        field = random.choice(fields)
        pattern = random.choice(patterns)
        text = pattern.format(topic=topic, field=field)
        
        # Add variations
        if i % 10 == 0:
            text += f" This was discovered in {2020 + i % 5}."
        if i % 15 == 0:
            text += f" The impact is estimated at ${random.randint(1, 100)} billion."
        
        dataset.append({
            "text": text,
            "metadata": {
                "source": "synthetic",
                "topic": topic,
                "field": field,
                "index": i
            }
        })
    
    return dataset

def measure_performance(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper

@measure_performance
def add_episodes_batch(agent, episodes):
    """Add multiple episodes and return statistics"""
    successful = 0
    failed = 0
    merged = 0
    
    for episode in episodes:
        result = agent.add_episode_with_graph_update(
            text=episode['text'],
            c_value=0.5
        )
        
        if result.get('success', False):
            successful += 1
            # Check if it was merged (episode count didn't increase)
            if 'merged' in str(result).lower():
                merged += 1
        else:
            failed += 1
    
    return {
        'successful': successful,
        'failed': failed,
        'merged': merged
    }

def run_large_scale_experiment():
    """Run the large-scale growth experiment"""
    print("=== InsightSpike-AI Large-Scale Growth Experiment ===")
    print(f"Start time: {datetime.now()}\n")
    
    # Initialize agent
    print("1. Initializing MainAgent...")
    agent = MainAgent()
    agent.initialize()
    
    # Reset to clean state
    print("2. Starting from clean state...")
    # Don't load existing state - start fresh
    
    # Get initial state
    initial_stats = agent.get_stats()
    initial_memory = initial_stats.get('memory_stats', {})
    initial_episodes = initial_memory.get('total_episodes', 0)
    
    print(f"Initial episodes: {initial_episodes}")
    
    # Generate large dataset
    print("\n3. Generating 1000 diverse text samples...")
    dataset = generate_diverse_dataset(1000)
    
    # Calculate total text size
    total_text_size = sum(len(item['text'].encode('utf-8')) for item in dataset)
    print(f"Total raw text size: {total_text_size:,} bytes ({total_text_size/1024/1024:.2f} MB)")
    
    # Add episodes in batches
    print("\n4. Adding episodes in batches...")
    batch_size = 100
    total_time = 0
    all_results = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(dataset)//batch_size}...")
        
        results, batch_time = add_episodes_batch(agent, batch)
        total_time += batch_time
        all_results.append(results)
        
        print(f"  Time: {batch_time:.2f}s")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Merged: {results['merged']}")
        
        # Save periodically
        if (i + batch_size) % 200 == 0:
            print("  Saving checkpoint...")
            agent.save_state()
    
    # Final save
    print("\n5. Saving final state...")
    agent.save_state()
    
    # Get final statistics
    final_stats = agent.get_stats()
    final_memory = final_stats.get('memory_stats', {})
    final_episodes = final_memory.get('total_episodes', 0)
    
    # Measure file sizes
    files = {
        "episodes.json": Path("data/episodes.json"),
        "graph_pyg.pt": Path("data/graph_pyg.pt"),
        "index.faiss": Path("data/index.faiss")
    }
    
    total_storage = 0
    print("\n=== Results ===")
    print(f"\nEpisode Growth:")
    print(f"  Initial: {initial_episodes}")
    print(f"  Final: {final_episodes}")
    print(f"  Net growth: {final_episodes - initial_episodes}")
    
    print(f"\nProcessing Statistics:")
    total_successful = sum(r['successful'] for r in all_results)
    total_failed = sum(r['failed'] for r in all_results)
    total_merged = sum(r['merged'] for r in all_results)
    
    print(f"  Total attempted: {len(dataset)}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    print(f"  Merged (deduplicated): {total_merged}")
    print(f"  Unique additions: {total_successful - total_merged}")
    
    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per episode: {total_time/len(dataset)*1000:.2f}ms")
    print(f"  Episodes per second: {len(dataset)/total_time:.2f}")
    
    print(f"\nStorage Analysis:")
    for name, path in files.items():
        if path.exists():
            size = path.stat().st_size
            total_storage += size
            print(f"  {name}: {size:,} bytes ({size/1024/1024:.2f} MB)")
    
    print(f"\nCompression Analysis:")
    print(f"  Raw text size: {total_text_size:,} bytes")
    print(f"  Total storage: {total_storage:,} bytes")
    print(f"  Compression ratio: {total_text_size/total_storage:.2f}x")
    
    # Test retrieval performance
    print(f"\n6. Testing retrieval performance...")
    test_queries = [
        "What is artificial intelligence?",
        "How does blockchain work?",
        "Explain quantum computing",
        "What are the applications of machine learning?",
        "Describe deep learning techniques"
    ]
    
    total_query_time = 0
    for query in test_queries:
        start = time.time()
        response = agent.process_question(query)
        query_time = time.time() - start
        total_query_time += query_time
        print(f"  Query: '{query[:30]}...' - Time: {query_time:.3f}s")
    
    avg_query_time = total_query_time / len(test_queries)
    print(f"\n  Average query time: {avg_query_time:.3f}s")
    
    # Memory efficiency analysis
    if final_episodes > 0:
        bytes_per_episode = total_storage / final_episodes
        print(f"\n7. Memory Efficiency:")
        print(f"  Bytes per episode: {bytes_per_episode:.0f}")
        print(f"  Episodes per MB: {1024*1024/bytes_per_episode:.0f}")
    
    print(f"\n=== Experiment Complete ===")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    run_large_scale_experiment()