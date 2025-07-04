#!/usr/bin/env python3
"""
Experiment 5: Scalability Performance Test
Compare original GraphBuilder vs ScalableGraphBuilder performance
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.layers.layer3_graph_reasoner import GraphBuilder
from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder


class ScalabilityTest:
    def __init__(self):
        self.results = {
            'original': {},
            'scalable': {},
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'test_sizes': []
            }
        }
        
    def generate_test_data(self, n_documents: int) -> list:
        """Generate test documents with embeddings"""
        documents = []
        embedding_dim = 384
        
        # Generate random embeddings (simulating real document embeddings)
        embeddings = np.random.randn(n_documents, embedding_dim)
        
        # Normalize to simulate sentence-transformers output
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        for i in range(n_documents):
            doc = {
                'text': f'Document {i}: Sample text content',
                'embedding': embeddings[i],
                'id': i
            }
            documents.append(doc)
            
        return documents
        
    def test_original_builder(self, documents: list) -> dict:
        """Test original GraphBuilder performance"""
        builder = GraphBuilder()
        
        start_time = time.time()
        graph = builder.build_graph(documents)
        build_time = time.time() - start_time
        
        return {
            'build_time': build_time,
            'num_nodes': graph.num_nodes if hasattr(graph, 'num_nodes') else len(documents),
            'num_edges': graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0,
            'memory_estimate': self._estimate_memory_usage(graph)
        }
        
    def test_scalable_builder(self, documents: list) -> dict:
        """Test ScalableGraphBuilder performance"""
        builder = ScalableGraphBuilder()
        
        start_time = time.time()
        graph = builder.build_graph(documents)
        build_time = time.time() - start_time
        
        return {
            'build_time': build_time,
            'num_nodes': graph.num_nodes,
            'num_edges': graph.edge_index.size(1),
            'memory_estimate': self._estimate_memory_usage(graph)
        }
        
    def _estimate_memory_usage(self, graph) -> float:
        """Estimate memory usage in MB"""
        if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
            return 0.0
            
        # Node features memory
        node_memory = graph.x.element_size() * graph.x.nelement() / (1024 * 1024)
        
        # Edge index memory
        edge_memory = graph.edge_index.element_size() * graph.edge_index.nelement() / (1024 * 1024)
        
        return node_memory + edge_memory
        
    def run_comparison(self):
        """Run performance comparison across different scales"""
        test_sizes = [100, 300, 500, 1000, 2000, 3000, 5000]
        
        print("=== Scalability Performance Test ===")
        print(f"Start time: {datetime.now()}")
        print(f"Test sizes: {test_sizes}\n")
        
        for size in test_sizes:
            print(f"\nTesting with {size} documents...")
            
            # Generate test data
            documents = self.generate_test_data(size)
            
            # Test original builder (skip for very large sizes)
            if size <= 1000:
                try:
                    print(f"  Testing original GraphBuilder...")
                    original_results = self.test_original_builder(documents)
                    self.results['original'][size] = original_results
                    print(f"    Build time: {original_results['build_time']:.2f}s")
                    print(f"    Edges: {original_results['num_edges']}")
                except Exception as e:
                    print(f"    Failed: {e}")
                    self.results['original'][size] = {'error': str(e)}
            else:
                print(f"  Skipping original GraphBuilder (too large)")
                self.results['original'][size] = {'skipped': True}
            
            # Test scalable builder
            try:
                print(f"  Testing ScalableGraphBuilder...")
                scalable_results = self.test_scalable_builder(documents)
                self.results['scalable'][size] = scalable_results
                print(f"    Build time: {scalable_results['build_time']:.2f}s")
                print(f"    Edges: {scalable_results['num_edges']}")
                
                # Calculate speedup
                if size in self.results['original'] and 'build_time' in self.results['original'][size]:
                    speedup = self.results['original'][size]['build_time'] / scalable_results['build_time']
                    print(f"    Speedup: {speedup:.2f}x")
            except Exception as e:
                print(f"    Failed: {e}")
                self.results['scalable'][size] = {'error': str(e)}
                
        self.results['metadata']['test_sizes'] = test_sizes
        self.results['metadata']['end_time'] = datetime.now().isoformat()
        
    def visualize_results(self):
        """Create performance comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        sizes = self.results['metadata']['test_sizes']
        
        # Build time comparison
        ax = axes[0, 0]
        original_times = [self.results['original'].get(s, {}).get('build_time', None) for s in sizes]
        scalable_times = [self.results['scalable'].get(s, {}).get('build_time', None) for s in sizes]
        
        # Filter out None values
        valid_original = [(s, t) for s, t in zip(sizes, original_times) if t is not None]
        valid_scalable = [(s, t) for s, t in zip(sizes, scalable_times) if t is not None]
        
        if valid_original:
            ax.plot(*zip(*valid_original), 'o-', label='Original', linewidth=2)
        if valid_scalable:
            ax.plot(*zip(*valid_scalable), 's-', label='Scalable', linewidth=2)
        
        ax.set_xlabel('Number of Documents')
        ax.set_ylabel('Build Time (seconds)')
        ax.set_title('Graph Build Time Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Edge count comparison
        ax = axes[0, 1]
        original_edges = [self.results['original'].get(s, {}).get('num_edges', None) for s in sizes]
        scalable_edges = [self.results['scalable'].get(s, {}).get('num_edges', None) for s in sizes]
        
        valid_original_edges = [(s, e) for s, e in zip(sizes, original_edges) if e is not None]
        valid_scalable_edges = [(s, e) for s, e in zip(sizes, scalable_edges) if e is not None]
        
        if valid_original_edges:
            ax.plot(*zip(*valid_original_edges), 'o-', label='Original', linewidth=2)
        if valid_scalable_edges:
            ax.plot(*zip(*valid_scalable_edges), 's-', label='Scalable', linewidth=2)
        
        ax.set_xlabel('Number of Documents')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Graph Edge Count Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Speedup plot
        ax = axes[1, 0]
        speedups = []
        speedup_sizes = []
        
        for size in sizes:
            if (size in self.results['original'] and 
                'build_time' in self.results['original'][size] and
                size in self.results['scalable'] and 
                'build_time' in self.results['scalable'][size]):
                speedup = self.results['original'][size]['build_time'] / self.results['scalable'][size]['build_time']
                speedups.append(speedup)
                speedup_sizes.append(size)
        
        if speedups:
            ax.bar(range(len(speedup_sizes)), speedups, color='green', alpha=0.7)
            ax.set_xticks(range(len(speedup_sizes)))
            ax.set_xticklabels(speedup_sizes)
            ax.set_xlabel('Number of Documents')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('ScalableGraphBuilder Speedup')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Memory usage
        ax = axes[1, 1]
        scalable_memory = [self.results['scalable'].get(s, {}).get('memory_estimate', None) for s in sizes]
        valid_memory = [(s, m) for s, m in zip(sizes, scalable_memory) if m is not None]
        
        if valid_memory:
            ax.plot(*zip(*valid_memory), 'g^-', label='Memory Usage', linewidth=2)
            ax.set_xlabel('Number of Documents')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Graph Memory Usage')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'scalability_performance_{timestamp}.png', dpi=150)
        print(f"\nPlot saved as scalability_performance_{timestamp}.png")
        
    def save_results(self):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'scalability_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Results saved to {filename}")
        
        # Print summary
        print("\n=== Performance Summary ===")
        for size in self.results['metadata']['test_sizes']:
            print(f"\n{size} documents:")
            
            if size in self.results['original'] and 'build_time' in self.results['original'][size]:
                orig_time = self.results['original'][size]['build_time']
                print(f"  Original: {orig_time:.2f}s")
            
            if size in self.results['scalable'] and 'build_time' in self.results['scalable'][size]:
                scale_time = self.results['scalable'][size]['build_time']
                print(f"  Scalable: {scale_time:.2f}s")
                
                if size in self.results['original'] and 'build_time' in self.results['original'][size]:
                    speedup = self.results['original'][size]['build_time'] / scale_time
                    print(f"  Speedup: {speedup:.2f}x")


def main():
    """Run the scalability test"""
    # Create experiment directory
    Path("experiment_5").mkdir(exist_ok=True)
    os.chdir("experiment_5")
    
    # Run test
    test = ScalabilityTest()
    test.run_comparison()
    test.visualize_results()
    test.save_results()
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()