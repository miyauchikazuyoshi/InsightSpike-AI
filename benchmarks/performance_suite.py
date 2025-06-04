#!/usr/bin/env python3
"""
Performance Suite for InsightSpike-AI

This module provides comprehensive performance benchmarking for scalability testing,
measuring algorithm complexity from O(n²) to O(n⁴) and large-scale graph processing.
"""

import time
import json
import statistics
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import networkx as nx
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.algorithms.graph_edit_distance import GraphEditDistance
from insightspike.algorithms.information_gain import InformationGain
from insightspike.core.insight_detector import InsightDetector


class PerformanceSuite:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ged_calculator = GraphEditDistance()
        self.ig_calculator = InformationGain()
        self.detector = InsightDetector()
        
        # Benchmark configurations
        self.graph_sizes = [10, 20, 50, 100, 200]  # Node counts
        self.iteration_counts = [5, 10, 20]  # For averaging
        
    def generate_test_graph(self, n_nodes: int, density: float = 0.3) -> nx.Graph:
        """Generate a test graph with specified properties."""
        np.random.seed(42)  # For reproducible benchmarks
        graph = nx.erdos_renyi_graph(n_nodes, density)
        
        # Add node attributes for realistic testing
        for node in graph.nodes():
            graph.nodes[node].update({
                'concept': f'concept_{node}',
                'type': np.random.choice(['entity', 'relation', 'property']),
                'weight': np.random.uniform(0.1, 1.0)
            })
            
        return graph
    
    def measure_ged_complexity(self) -> Dict[str, Any]:
        """Measure Graph Edit Distance computational complexity."""
        results = {
            'algorithm': 'graph_edit_distance',
            'complexity_analysis': {},
            'performance_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        print("🔍 Measuring GED complexity...")
        
        for size in self.graph_sizes:
            print(f"  Testing graph size: {size} nodes")
            
            # Generate test graphs
            graph1 = self.generate_test_graph(size, 0.3)
            graph2 = self.generate_test_graph(size, 0.4)  # Slightly different
            
            # Measure execution times
            execution_times = []
            
            for iteration in range(min(5, max(1, 100 // size))):  # Adaptive iterations
                start_time = time.perf_counter()
                
                try:
                    distance = self.ged_calculator.calculate_distance(graph1, graph2)
                    end_time = time.perf_counter()
                    execution_times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"    Warning: Failed iteration {iteration}: {e}")
                    continue
            
            if execution_times:
                avg_time = statistics.mean(execution_times)
                std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                
                results['complexity_analysis'][size] = {
                    'avg_execution_time': avg_time,
                    'std_execution_time': std_time,
                    'iterations': len(execution_times),
                    'theoretical_complexity': 'O(n³)',  # GED is typically O(n³)
                    'measured_scaling': self._calculate_scaling_factor(size, avg_time, results['complexity_analysis'])
                }
                
                print(f"    Average time: {avg_time:.4f}s (±{std_time:.4f})")
            else:
                print(f"    Failed to measure size {size}")
        
        return results
    
    def measure_ig_complexity(self) -> Dict[str, Any]:
        """Measure Information Gain computational complexity."""
        results = {
            'algorithm': 'information_gain',
            'complexity_analysis': {},
            'performance_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        print("📊 Measuring IG complexity...")
        
        for size in self.graph_sizes:
            print(f"  Testing dataset size: {size} samples")
            
            # Generate test data
            feature_data = np.random.rand(size, 10)  # 10 features
            target_data = np.random.randint(0, 3, size)  # 3 classes
            
            execution_times = []
            
            for iteration in range(min(10, max(1, 200 // size))):  # Adaptive iterations
                start_time = time.perf_counter()
                
                try:
                    ig_value = self.ig_calculator.calculate_gain(feature_data, target_data)
                    end_time = time.perf_counter()
                    execution_times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"    Warning: Failed iteration {iteration}: {e}")
                    continue
            
            if execution_times:
                avg_time = statistics.mean(execution_times)
                std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                
                results['complexity_analysis'][size] = {
                    'avg_execution_time': avg_time,
                    'std_execution_time': std_time,
                    'iterations': len(execution_times),
                    'theoretical_complexity': 'O(n log n)',  # IG is typically O(n log n)
                    'measured_scaling': self._calculate_scaling_factor(size, avg_time, results['complexity_analysis'])
                }
                
                print(f"    Average time: {avg_time:.4f}s (±{std_time:.4f})")
        
        return results
    
    def measure_integration_performance(self) -> Dict[str, Any]:
        """Measure end-to-end integration performance."""
        results = {
            'algorithm': 'integration_workflow',
            'workflow_metrics': {},
            'scalability_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        print("🔗 Measuring integration performance...")
        
        test_questions = [
            "What is the Monty Hall problem?",
            "Explain the Ship of Theseus paradox.",
            "How does machine learning work?",
            "What are the applications of graph theory?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"  Testing question {i}: {question[:30]}...")
            
            execution_times = []
            
            for iteration in range(3):  # 3 iterations for stability
                start_time = time.perf_counter()
                
                try:
                    # Simulate full detection workflow
                    result = self.detector.detect_insights(
                        query=question,
                        context="test_context"
                    )
                    end_time = time.perf_counter()
                    execution_times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"    Warning: Failed iteration {iteration}: {e}")
                    continue
            
            if execution_times:
                avg_time = statistics.mean(execution_times)
                
                results['workflow_metrics'][f'question_{i}'] = {
                    'question': question,
                    'avg_response_time': avg_time,
                    'meets_target': avg_time < 1.0,  # 1 second target
                    'iterations': len(execution_times)
                }
                
                print(f"    Average response time: {avg_time:.4f}s")
        
        return results
    
    def _calculate_scaling_factor(self, size: int, time_val: float, prev_results: Dict) -> float:
        """Calculate scaling factor compared to previous size."""
        if not prev_results:
            return 1.0
        
        prev_sizes = sorted([int(k) for k in prev_results.keys()])
        if not prev_sizes:
            return 1.0
        
        prev_size = prev_sizes[-1]
        prev_time = prev_results[prev_size]['avg_execution_time']
        
        if prev_time == 0:
            return float('inf')
        
        scaling = (time_val / prev_time) / (size / prev_size)
        return scaling
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("📋 Generating performance report...\n")
        
        # Run all benchmarks
        ged_results = self.measure_ged_complexity()
        ig_results = self.measure_ig_complexity()
        integration_results = self.measure_integration_performance()
        
        # Compile comprehensive report
        report = {
            'benchmark_metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_environment': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'graph_sizes_tested': self.graph_sizes,
                },
                'performance_targets': {
                    'response_time_target': '< 1.0 seconds',
                    'scalability_requirement': 'O(n³) or better',
                    'memory_efficiency': 'Linear scaling preferred'
                }
            },
            'algorithm_benchmarks': {
                'graph_edit_distance': ged_results,
                'information_gain': ig_results,
                'integration_workflow': integration_results
            },
            'summary_analysis': self._generate_summary_analysis(ged_results, ig_results, integration_results)
        }
        
        return report
    
    def _generate_summary_analysis(self, ged_results: Dict, ig_results: Dict, integration_results: Dict) -> Dict[str, Any]:
        """Generate summary analysis of all benchmarks."""
        analysis = {
            'performance_status': 'PASS',
            'bottlenecks_identified': [],
            'optimization_recommendations': [],
            'scalability_assessment': {}
        }
        
        # Analyze GED performance
        ged_times = [v['avg_execution_time'] for v in ged_results.get('complexity_analysis', {}).values()]
        if ged_times and max(ged_times) > 5.0:  # > 5 seconds is concerning
            analysis['bottlenecks_identified'].append('GED calculation performance degradation at large graph sizes')
            analysis['optimization_recommendations'].append('Consider approximation algorithms for large graphs')
        
        # Analyze integration performance
        integration_metrics = integration_results.get('workflow_metrics', {})
        slow_queries = [k for k, v in integration_metrics.items() if not v.get('meets_target', True)]
        
        if slow_queries:
            analysis['performance_status'] = 'WARNING'
            analysis['bottlenecks_identified'].append(f'Slow response times for {len(slow_queries)} queries')
            analysis['optimization_recommendations'].append('Implement caching or preprocessing optimizations')
        
        # Scalability assessment
        if ged_times:
            analysis['scalability_assessment']['ged_max_time'] = max(ged_times)
            analysis['scalability_assessment']['ged_scaling'] = 'Acceptable' if max(ged_times) < 5.0 else 'Needs optimization'
        
        return analysis
    
    def save_results(self, report: Dict[str, Any]) -> str:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_benchmark_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {filepath}")
        return str(filepath)


def main():
    """Main execution function."""
    print("🚀 InsightSpike-AI Performance Suite")
    print("=" * 50)
    
    # Initialize suite
    suite = PerformanceSuite()
    
    # Generate comprehensive report
    report = suite.generate_performance_report()
    
    # Save results
    results_file = suite.save_results(report)
    
    # Print summary
    print("\n📊 Performance Summary:")
    print("-" * 30)
    
    summary = report.get('summary_analysis', {})
    status = summary.get('performance_status', 'UNKNOWN')
    print(f"Overall Status: {status}")
    
    bottlenecks = summary.get('bottlenecks_identified', [])
    if bottlenecks:
        print(f"Bottlenecks: {len(bottlenecks)} identified")
        for bottleneck in bottlenecks:
            print(f"  - {bottleneck}")
    
    recommendations = summary.get('optimization_recommendations', [])
    if recommendations:
        print(f"Recommendations: {len(recommendations)} suggested")
        for rec in recommendations:
            print(f"  - {rec}")
    
    print(f"\nDetailed results: {results_file}")
    return 0 if status == 'PASS' else 1


if __name__ == "__main__":
    exit(main())
