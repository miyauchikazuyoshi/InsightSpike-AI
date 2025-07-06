#!/usr/bin/env python3
"""
Experiment 5: Scalable Graph Growth with FAISS-based GraphBuilder
Tests the new ScalableGraphBuilder with large-scale data
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

class ScalableGraphExperiment:
    def __init__(self):
        self.agent = None
        self.all_documents = []  # Critical: Track ALL documents for graph growth
        self.metrics = {
            'episodes': [],
            'graph_nodes': [],
            'graph_edges': [],
            'file_sizes': [],
            'timestamps': [],
            'processing_times': [],
            'build_times': []  # Track graph build times separately
        }
    
    def initialize(self):
        """Initialize agent and verify clean state"""
        print("=== Experiment 5: Scalable Graph Growth with FAISS ===")
        print(f"Start time: {datetime.now()}\n")
        
        self.agent = MainAgent()
        self.agent.initialize()
        self.agent.load_state()
        
        # Get initial state
        initial_state = self._get_system_state()
        print(f"Initial State:")
        print(f"  Episodes: {initial_state['episodes']}")
        print(f"  Graph nodes: {initial_state['graph_nodes']}")
        print(f"  Graph edges: {initial_state['graph_edges']}")
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
            "autonomous vehicles", "bioinformatics", "computational biology",
            "distributed systems", "cloud computing", "IoT", "5G networks",
            "augmented reality", "virtual reality", "digital twins",
            "knowledge graphs", "semantic web", "ontology engineering"
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
            "{topic} provides new insights into {field} challenges.",
            "Integrating {topic} with {field} yields unexpected benefits.",
            "The synergy between {topic} and {field} drives innovation."
        ]
        
        fields = [
            "healthcare", "finance", "education", "manufacturing",
            "transportation", "energy", "agriculture", "retail",
            "entertainment", "telecommunications", "aerospace", "defense",
            "pharmaceuticals", "logistics", "construction", "mining",
            "insurance", "real estate", "hospitality", "media"
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
            if i % 20 == 0:
                text += f" Over {random.randint(100, 1000)} organizations are adopting this approach."
            
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
                graph_start = time.time()
                graph_analysis = self.agent.l3_graph.analyze_documents(self.all_documents)
                graph_build_time = time.time() - graph_start
                
                result['graph_analysis'] = graph_analysis
                result['graph_nodes'] = self.agent.l3_graph.previous_graph.num_nodes if self.agent.l3_graph.previous_graph else 0
                result['graph_edges'] = self.agent.l3_graph.previous_graph.edge_index.size(1) if self.agent.l3_graph.previous_graph else 0
                result['graph_build_time'] = graph_build_time
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def run_phase1_performance_comparison(self):
        """Phase 1: Compare performance with original experiment"""
        print("\n=== Phase 1: Performance Comparison (300 documents) ===")
        
        dataset = self.generate_dataset(300)
        successful = 0
        failed = 0
        total_build_time = 0
        
        for i, text in enumerate(dataset):
            result = self.add_episode_with_graph_growth(text)
            
            if result['success']:
                successful += 1
                if 'graph_build_time' in result:
                    total_build_time += result['graph_build_time']
                
                # Report progress every 50 documents
                if (i + 1) % 50 == 0:
                    state = self._get_system_state()
                    avg_build_time = total_build_time / (i + 1)
                    print(f"\nProgress: {i + 1}/300")
                    print(f"  Episodes: {state['episodes']}")
                    print(f"  Graph nodes: {state['graph_nodes']}")
                    print(f"  Graph edges: {state['graph_edges']}")
                    print(f"  Avg graph build time: {avg_build_time:.3f}s")
                    print(f"  Unique documents tracked: {len(self.all_documents)}")
                    
                    self._record_metrics(i + 1)
            else:
                failed += 1
        
        # Save state
        self.agent.save_state()
        
        print(f"\nPhase 1 Complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Average graph build time: {total_build_time/successful:.3f}s")
        
        return successful, failed
    
    def run_phase2_large_scale(self):
        """Phase 2: Test with much larger scale (5000 documents)"""
        print("\n=== Phase 2: Large Scale Experiment (5000 documents) ===")
        
        dataset = self.generate_dataset(5000)
        batch_size = 100
        total_successful = 0
        total_failed = 0
        total_build_time = 0
        
        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{len(dataset)//batch_size}")
            
            batch_start = time.time()
            successful = 0
            failed = 0
            batch_build_time = 0
            
            for text in batch:
                result = self.add_episode_with_graph_growth(text)
                if result['success']:
                    successful += 1
                    if 'graph_build_time' in result:
                        batch_build_time += result['graph_build_time']
                else:
                    failed += 1
            
            batch_time = time.time() - batch_start
            total_successful += successful
            total_failed += failed
            total_build_time += batch_build_time
            
            # Report batch results
            state = self._get_system_state()
            print(f"  Batch time: {batch_time:.2f}s")
            print(f"  Batch graph build time: {batch_build_time:.2f}s")
            print(f"  Successful: {successful}, Failed: {failed}")
            print(f"  Current episodes: {state['episodes']}")
            print(f"  Current graph nodes: {state['graph_nodes']}")
            print(f"  Current graph edges: {state['graph_edges']}")
            print(f"  Edges per node: {state['graph_edges']/max(1, state['graph_nodes']):.1f}")
            
            # Save checkpoint every 500 documents
            if (batch_idx + batch_size) % 500 == 0:
                print("  Saving checkpoint...")
                self.agent.save_state()
                
            self._record_metrics(batch_idx + batch_size)
        
        # Final save
        self.agent.save_state()
        
        print(f"\nPhase 2 Complete:")
        print(f"  Total successful: {total_successful}")
        print(f"  Total failed: {total_failed}")
        print(f"  Total graph build time: {total_build_time:.2f}s")
        print(f"  Average graph build time: {total_build_time/total_successful:.3f}s")
        
        return total_successful, total_failed
    
    def analyze_scalability(self):
        """Phase 3: Analyze scalability metrics"""
        print("\n=== Phase 3: Scalability Analysis ===")
        
        # Calculate graph density over time
        print("\nGraph Density Analysis:")
        for i, (docs, nodes, edges) in enumerate(zip(
            self.metrics['timestamps'], 
            self.metrics['graph_nodes'], 
            self.metrics['graph_edges']
        )):
            if nodes > 0:
                density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
                avg_degree = edges / nodes
                print(f"  {docs:4d} docs: {nodes:4d} nodes, {edges:5d} edges, "
                      f"density: {density:.3f}, avg degree: {avg_degree:.1f}")
        
        # Performance analysis
        if self.metrics['build_times']:
            avg_build_time = sum(self.metrics['build_times']) / len(self.metrics['build_times'])
            max_build_time = max(self.metrics['build_times'])
            print(f"\nPerformance Metrics:")
            print(f"  Average graph build time: {avg_build_time:.3f}s")
            print(f"  Maximum graph build time: {max_build_time:.3f}s")
            
            # Estimate for larger scales
            current_nodes = self.metrics['graph_nodes'][-1] if self.metrics['graph_nodes'] else 0
            if current_nodes > 0:
                # Assuming O(n log n) complexity with FAISS
                time_10k = avg_build_time * (10000 / current_nodes) * np.log(10000) / np.log(current_nodes)
                time_100k = avg_build_time * (100000 / current_nodes) * np.log(100000) / np.log(current_nodes)
                
                print(f"\nProjected Performance (based on O(n log n)):")
                print(f"  10,000 nodes: ~{time_10k:.1f}s per update")
                print(f"  100,000 nodes: ~{time_100k:.1f}s per update")
        
    def save_results(self):
        """Save detailed results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        final_state = self._get_system_state()
        
        results = {
            'experiment': 'Scalable Graph Growth with FAISS',
            'timestamp': timestamp,
            'initial_state': self._get_system_state(),  # Will be the final state, but structure is same
            'final_state': final_state,
            'metrics': self.metrics,
            'summary': {
                'total_documents': len(self.all_documents),
                'final_episodes': final_state['episodes'],
                'final_graph_nodes': final_state['graph_nodes'],
                'final_graph_edges': final_state['graph_edges'],
                'graph_implementation': 'ScalableGraphBuilder with FAISS',
                'topk_setting': 50
            }
        }
        
        filename = f'scalable_graph_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        
        # Print final summary
        print("\n=== Final Summary ===")
        print(f"Documents processed: {len(self.all_documents)}")
        print(f"Final graph: {final_state['graph_nodes']} nodes, {final_state['graph_edges']} edges")
        print(f"Average edges per node: {final_state['graph_edges']/max(1, final_state['graph_nodes']):.1f}")
        print(f"Graph density: {final_state['graph_edges']/(final_state['graph_nodes']*(final_state['graph_nodes']-1)):.4f}")
    
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
        
        # Get edge count from graph
        graph_edges = 0
        if self.agent.l3_graph and self.agent.l3_graph.previous_graph:
            graph_edges = self.agent.l3_graph.previous_graph.edge_index.size(1)
        
        return {
            'episodes': memory_stats.get('total_episodes', 0),
            'graph_nodes': graph_state['graph'].get('num_nodes', 0),
            'graph_edges': graph_edges,
            'file_sizes': file_sizes
        }
    
    def _record_metrics(self, document_count: int):
        """Record current metrics"""
        state = self._get_system_state()
        self.metrics['timestamps'].append(document_count)
        self.metrics['episodes'].append(state['episodes'])
        self.metrics['graph_nodes'].append(state['graph_nodes'])
        self.metrics['graph_edges'].append(state['graph_edges'])
        self.metrics['file_sizes'].append(state['file_sizes'])
        self.metrics['processing_times'].append(time.time())

def main():
    """Run the complete experiment"""
    experiment = ScalableGraphExperiment()
    
    # Initialize
    initial_state = experiment.initialize()
    
    # Phase 1: Performance comparison with 300 docs
    experiment.run_phase1_performance_comparison()
    
    # Phase 2: Large scale test with 5000 docs
    experiment.run_phase2_large_scale()
    
    # Phase 3: Scalability analysis
    experiment.analyze_scalability()
    
    # Save results
    experiment.save_results()
    
    print(f"\nExperiment completed at: {datetime.now()}")

if __name__ == "__main__":
    main()