#!/usr/bin/env python3
"""
Experiment 4: Dynamic Graph Growth with RAG
Properly implements cumulative document tracking for graph growth
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent

class DynamicGraphExperiment:
    def __init__(self):
        self.agent = None
        self.all_documents = []  # Critical: Track ALL documents for graph growth
        self.metrics = {
            'episodes': [],
            'graph_nodes': [],
            'file_sizes': [],
            'timestamps': [],
            'processing_times': []
        }
    
    def initialize(self):
        """Initialize agent and verify clean state"""
        print("=== Experiment 4: Dynamic Graph Growth ===")
        print(f"Start time: {datetime.now()}\n")
        
        self.agent = MainAgent()
        self.agent.initialize()
        self.agent.load_state()
        
        # Get initial state
        initial_state = self._get_system_state()
        print(f"Initial State:")
        print(f"  Episodes: {initial_state['episodes']}")
        print(f"  Graph nodes: {initial_state['graph_nodes']}")
        print(f"  File sizes: {initial_state['file_sizes']}")
        
        # Record initial metrics
        self._record_metrics(0)
        
        return initial_state
    
    def generate_dataset(self, count: int) -> List[str]:
        """Generate diverse dataset for testing"""
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
            
            dataset.append(text)
        
        return dataset
    
    def add_episode_with_graph_growth(self, text: str) -> Dict[str, Any]:
        """Add episode and properly update graph with ALL documents"""
        start_time = time.time()
        
        # Add episode
        result = self.agent.add_episode_with_graph_update(text)
        
        if result['success']:
            # Create document for graph
            doc = {
                "text": text,
                "embedding": result['vector'],
                "c_value": result['c_value'],
                "episode_idx": result['episode_idx']
            }
            self.all_documents.append(doc)
            
            # CRITICAL: Update graph with ALL documents
            if self.agent.l3_graph:
                graph_analysis = self.agent.l3_graph.analyze_documents(self.all_documents)
                result['graph_analysis'] = graph_analysis
                result['graph_nodes'] = self.agent.l3_graph.previous_graph.num_nodes if self.agent.l3_graph.previous_graph else 0
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def run_phase1_small_scale(self):
        """Phase 1: Demonstrate graph growth with 100 documents"""
        print("\n=== Phase 1: Small Scale Graph Growth Demo ===")
        
        dataset = self.generate_dataset(100)
        successful = 0
        failed = 0
        
        for i, text in enumerate(dataset):
            result = self.add_episode_with_graph_growth(text)
            
            if result['success']:
                successful += 1
                
                # Report progress every 10 documents
                if (i + 1) % 10 == 0:
                    state = self._get_system_state()
                    print(f"\nProgress: {i + 1}/100")
                    print(f"  Episodes: {state['episodes']}")
                    print(f"  Graph nodes: {state['graph_nodes']}")
                    print(f"  Unique documents tracked: {len(self.all_documents)}")
                    
                    self._record_metrics(i + 1)
            else:
                failed += 1
        
        # Save state
        self.agent.save_state()
        
        print(f"\nPhase 1 Complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        return successful, failed
    
    def run_phase2_large_scale(self):
        """Phase 2: Large scale experiment with 1000 documents"""
        print("\n=== Phase 2: Large Scale Experiment (1000 documents) ===")
        
        dataset = self.generate_dataset(1000)
        batch_size = 100
        total_successful = 0
        total_failed = 0
        
        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{len(dataset)//batch_size}")
            
            batch_start = time.time()
            successful = 0
            failed = 0
            
            for text in batch:
                result = self.add_episode_with_graph_growth(text)
                if result['success']:
                    successful += 1
                else:
                    failed += 1
            
            batch_time = time.time() - batch_start
            total_successful += successful
            total_failed += failed
            
            # Report batch results
            state = self._get_system_state()
            print(f"  Batch time: {batch_time:.2f}s")
            print(f"  Successful: {successful}, Failed: {failed}")
            print(f"  Current episodes: {state['episodes']}")
            print(f"  Current graph nodes: {state['graph_nodes']}")
            
            # Save checkpoint
            if (batch_idx + batch_size) % 200 == 0:
                print("  Saving checkpoint...")
                self.agent.save_state()
                
            self._record_metrics(batch_idx + batch_size)
        
        # Final save
        self.agent.save_state()
        
        print(f"\nPhase 2 Complete:")
        print(f"  Total successful: {total_successful}")
        print(f"  Total failed: {total_failed}")
        
        return total_successful, total_failed
    
    def analyze_compression(self):
        """Phase 3: Analyze compression efficiency"""
        print("\n=== Phase 3: Compression Analysis ===")
        
        # Calculate raw text size
        total_text_size = sum(len(doc['text'].encode('utf-8')) for doc in self.all_documents)
        
        # Get file sizes
        files = {
            "episodes.json": Path("data/episodes.json"),
            "graph_pyg.pt": Path("data/graph_pyg.pt"),
            "index.faiss": Path("data/index.faiss")
        }
        
        total_storage = 0
        for name, path in files.items():
            if path.exists():
                size = path.stat().st_size
                total_storage += size
                print(f"  {name}: {size:,} bytes ({size/1024/1024:.2f} MB)")
        
        compression_ratio = total_text_size / total_storage if total_storage > 0 else 0
        
        print(f"\nCompression Results:")
        print(f"  Raw text size: {total_text_size:,} bytes")
        print(f"  Total storage: {total_storage:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Bytes per document: {total_storage / len(self.all_documents):.0f}")
        
        return compression_ratio
    
    def plot_growth_metrics(self):
        """Visualize growth metrics"""
        print("\n=== Growth Metrics Visualization ===")
        
        # Simple text-based visualization
        print("\nEpisode Growth:")
        for i, (docs, episodes) in enumerate(zip(self.metrics['timestamps'], self.metrics['episodes'])):
            bar = '█' * (episodes // 10)
            print(f"  {docs:4d} docs: {bar} {episodes}")
        
        print("\nGraph Node Growth:")
        for i, (docs, nodes) in enumerate(zip(self.metrics['timestamps'], self.metrics['graph_nodes'])):
            bar = '█' * (nodes // 2)
            print(f"  {docs:4d} docs: {bar} {nodes}")
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        stats = self.agent.get_stats()
        memory_stats = stats.get('memory_stats', {})
        graph_state = self.agent.get_memory_graph_state()
        
        files = {
            "episodes.json": Path("data/episodes.json"),
            "graph_pyg.pt": Path("data/graph_pyg.pt"),
            "index.faiss": Path("data/index.faiss")
        }
        
        file_sizes = {}
        for name, path in files.items():
            file_sizes[name] = path.stat().st_size if path.exists() else 0
        
        return {
            'episodes': memory_stats.get('total_episodes', 0),
            'graph_nodes': graph_state['graph'].get('num_nodes', 0),
            'file_sizes': file_sizes
        }
    
    def _record_metrics(self, document_count: int):
        """Record current metrics"""
        state = self._get_system_state()
        self.metrics['timestamps'].append(document_count)
        self.metrics['episodes'].append(state['episodes'])
        self.metrics['graph_nodes'].append(state['graph_nodes'])
        self.metrics['file_sizes'].append(state['file_sizes'])
        self.metrics['processing_times'].append(time.time())

def main():
    """Run the complete experiment"""
    experiment = DynamicGraphExperiment()
    
    # Initialize
    initial_state = experiment.initialize()
    
    # Phase 1: Small scale demo
    experiment.run_phase1_small_scale()
    
    # Phase 2: Large scale
    experiment.run_phase2_large_scale()
    
    # Phase 3: Compression analysis
    compression_ratio = experiment.analyze_compression()
    
    # Visualize growth
    experiment.plot_growth_metrics()
    
    # Final summary
    final_state = experiment._get_system_state()
    print("\n=== Final Summary ===")
    print(f"Initial → Final:")
    print(f"  Episodes: {initial_state['episodes']} → {final_state['episodes']}")
    print(f"  Graph nodes: {initial_state['graph_nodes']} → {final_state['graph_nodes']}")
    print(f"  Episodes growth: {(final_state['episodes'] / max(1, initial_state['episodes']) - 1) * 100:.1f}%")
    print(f"  Graph growth: {(final_state['graph_nodes'] / max(1, initial_state['graph_nodes']) - 1) * 100:.1f}%")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    print(f"\nExperiment completed at: {datetime.now()}")

if __name__ == "__main__":
    main()