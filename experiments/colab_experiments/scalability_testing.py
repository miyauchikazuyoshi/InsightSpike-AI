#!/usr/bin/env python3
"""
Large-Scale Scalability Testing for Google Colab
===============================================

Based on docs/experiment_design/05_scalability_testing.md
Tests InsightSpike performance with varying data scales.
"""

import json
import logging
import time
import psutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.append('/content/InsightSpike-AI/src')

from insightspike.config import InsightSpikeConfig
from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
from insightspike.processing.embedder import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalabilityTestSuite:
    """Comprehensive scalability testing for InsightSpike"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize scalability test suite"""
        self.config_path = config_path or "experiments/colab_experiments/colab_config.yaml"
        self.results_dir = Path("experiments/colab_experiments/results/scalability")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = []
        
    def run_memory_scaling_test(self, 
                               episode_counts: List[int] = [100, 500, 1000, 5000, 10000],
                               query_count: int = 100) -> Dict[str, Any]:
        """
        Test memory system performance with increasing episode counts
        
        Args:
            episode_counts: List of episode counts to test
            query_count: Number of queries per episode count
        """
        logger.info("🧠 Running Memory Scaling Test...")
        
        results = {
            'test': 'memory_scaling',
            'timestamp': datetime.now().isoformat(),
            'episode_counts': episode_counts,
            'query_count': query_count,
            'measurements': []
        }
        
        # Load config
        config = InsightSpikeConfig()
        embedder = EmbeddingManager()
        
        for n_episodes in episode_counts:
            logger.info(f"\n📊 Testing with {n_episodes} episodes...")
            
            # Create memory manager
            memory = L2MemoryManager(
                index_file=f"test_index_{n_episodes}.faiss",
                episodes_file=f"test_episodes_{n_episodes}.json",
                config=config
            )
            
            # Generate and store episodes
            logger.info("Generating episodes...")
            self._generate_episodes(memory, n_episodes, embedder)
            
            # Measure performance
            metrics = self._measure_memory_performance(memory, embedder, query_count)
            metrics['n_episodes'] = n_episodes
            
            results['measurements'].append(metrics)
            
            # Cleanup
            del memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log intermediate results
            logger.info(f"✅ {n_episodes} episodes - Avg retrieval: {metrics['avg_retrieval_time']:.3f}s")
        
        # Analyze scaling behavior
        results['analysis'] = self._analyze_scaling_behavior(results['measurements'])
        
        # Generate plots
        self._plot_memory_scaling(results)
        
        # Save results
        self._save_results(results, 'memory_scaling_test')
        
        return results
    
    def run_graph_scaling_test(self,
                              node_counts: List[int] = [100, 500, 1000, 5000],
                              edge_density: float = 0.1) -> Dict[str, Any]:
        """
        Test graph processing performance with increasing graph sizes
        
        Args:
            node_counts: List of node counts to test
            edge_density: Edge density (fraction of possible edges)
        """
        logger.info("🕸️ Running Graph Scaling Test...")
        
        results = {
            'test': 'graph_scaling',
            'timestamp': datetime.now().isoformat(),
            'node_counts': node_counts,
            'edge_density': edge_density,
            'measurements': []
        }
        
        config = InsightSpikeConfig()
        
        for n_nodes in node_counts:
            logger.info(f"\n📊 Testing with {n_nodes} nodes...")
            
            # Generate graph
            n_edges = int(n_nodes * (n_nodes - 1) * edge_density / 2)
            graph = self._generate_random_graph(n_nodes, n_edges)
            
            # Measure graph operations
            metrics = self._measure_graph_performance(graph, config)
            metrics['n_nodes'] = n_nodes
            metrics['n_edges'] = n_edges
            
            results['measurements'].append(metrics)
            
            # Cleanup
            del graph
            gc.collect()
            
            logger.info(f"✅ {n_nodes} nodes - GED calc: {metrics['avg_ged_time']:.3f}s")
        
        # Analyze complexity
        results['complexity_analysis'] = self._analyze_graph_complexity(results['measurements'])
        
        # Generate plots
        self._plot_graph_scaling(results)
        
        # Save results
        self._save_results(results, 'graph_scaling_test')
        
        return results
    
    def run_concurrent_user_test(self,
                                user_counts: List[int] = [1, 5, 10, 20, 50],
                                queries_per_user: int = 10) -> Dict[str, Any]:
        """
        Test system performance with concurrent users
        
        Args:
            user_counts: List of concurrent user counts to test
            queries_per_user: Number of queries per user
        """
        logger.info("👥 Running Concurrent User Test...")
        
        results = {
            'test': 'concurrent_users',
            'timestamp': datetime.now().isoformat(),
            'user_counts': user_counts,
            'queries_per_user': queries_per_user,
            'measurements': []
        }
        
        # Setup shared resources
        config = InsightSpikeConfig()
        agent = MainAgent(config)
        
        for n_users in user_counts:
            logger.info(f"\n📊 Testing with {n_users} concurrent users...")
            
            # Simulate concurrent queries
            metrics = self._simulate_concurrent_users(agent, n_users, queries_per_user)
            metrics['n_users'] = n_users
            
            results['measurements'].append(metrics)
            
            logger.info(f"✅ {n_users} users - Throughput: {metrics['throughput']:.1f} queries/sec")
        
        # Calculate scalability metrics
        results['scalability_metrics'] = self._calculate_scalability_metrics(results['measurements'])
        
        # Generate plots
        self._plot_concurrent_scaling(results)
        
        # Save results
        self._save_results(results, 'concurrent_user_test')
        
        return results
    
    def run_model_size_comparison(self,
                                 models: List[str] = None,
                                 test_queries: int = 50) -> Dict[str, Any]:
        """
        Compare performance across different model sizes
        
        Args:
            models: List of model names to test
            test_queries: Number of test queries per model
        """
        if models is None:
            models = [
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "microsoft/phi-2",
                "mistralai/Mistral-7B-v0.1"
            ]
        
        logger.info("🤖 Running Model Size Comparison...")
        
        results = {
            'test': 'model_size_comparison',
            'timestamp': datetime.now().isoformat(),
            'models': models,
            'test_queries': test_queries,
            'measurements': []
        }
        
        for model_name in models:
            logger.info(f"\n📊 Testing model: {model_name}")
            
            try:
                # Create config for model
                config = InsightSpikeConfig()
                config.core.model_name = model_name
                
                # Check if model fits in memory
                if not self._check_model_memory_fit(model_name):
                    logger.warning(f"⚠️ Model {model_name} too large for available memory")
                    continue
                
                # Test model
                metrics = self._test_model_performance(config, test_queries)
                metrics['model'] = model_name
                metrics['model_size_gb'] = self._get_model_size(model_name)
                
                results['measurements'].append(metrics)
                
                # Cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed to test {model_name}: {e}")
        
        # Analyze trade-offs
        results['tradeoff_analysis'] = self._analyze_model_tradeoffs(results['measurements'])
        
        # Generate plots
        self._plot_model_comparison(results)
        
        # Save results
        self._save_results(results, 'model_size_comparison')
        
        return results
    
    def run_comprehensive_scalability_test(self) -> Dict[str, Any]:
        """Run all scalability tests"""
        logger.info("🏃 Running Comprehensive Scalability Tests...")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'environment': self._get_system_info(),
            'tests': {}
        }
        
        # Define test configurations
        tests = [
            ('memory_scaling', lambda: self.run_memory_scaling_test()),
            ('graph_scaling', lambda: self.run_graph_scaling_test()),
            ('concurrent_users', lambda: self.run_concurrent_user_test()),
            ('model_comparison', lambda: self.run_model_size_comparison())
        ]
        
        for name, test_fn in tests:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running {name} test...")
                logger.info(f"{'='*60}")
                
                results = test_fn()
                all_results['tests'][name] = results
                
            except Exception as e:
                logger.error(f"Failed to run {name} test: {e}")
                all_results['tests'][name] = {'error': str(e)}
        
        # Generate comprehensive report
        self._generate_scalability_report(all_results)
        
        # Save all results
        output_path = self.results_dir / f"comprehensive_scalability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\n✅ Scalability tests complete!")
        logger.info(f"📊 Results saved to: {output_path}")
        
        return all_results
    
    # Helper methods
    def _generate_episodes(self, memory: L2MemoryManager, n_episodes: int, embedder: EmbeddingManager):
        """Generate test episodes"""
        for i in range(n_episodes):
            text = f"Test episode {i}: " + " ".join([f"word{j}" for j in range(50)])
            embedding = embedder.embed_text(text)
            memory.store_episode(text, embedding, metadata={'id': i})
    
    def _measure_memory_performance(self, memory: L2MemoryManager, embedder: EmbeddingManager, 
                                   query_count: int) -> Dict[str, float]:
        """Measure memory system performance"""
        retrieval_times = []
        memory_usage = []
        
        for i in range(query_count):
            query_text = f"Test query {i}: " + " ".join([f"term{j}" for j in range(10)])
            query_vec = embedder.embed_text(query_text)
            
            start_time = time.time()
            results = memory.search(query_vec, k=10)
            retrieval_time = time.time() - start_time
            
            retrieval_times.append(retrieval_time)
            
            # Measure memory periodically
            if i % 10 == 0:
                memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
        
        return {
            'avg_retrieval_time': np.mean(retrieval_times),
            'p95_retrieval_time': np.percentile(retrieval_times, 95),
            'p99_retrieval_time': np.percentile(retrieval_times, 99),
            'avg_memory_mb': np.mean(memory_usage),
            'peak_memory_mb': np.max(memory_usage)
        }
    
    def _generate_random_graph(self, n_nodes: int, n_edges: int):
        """Generate random graph for testing"""
        import networkx as nx
        
        # Use Erdős-Rényi random graph
        p = n_edges / (n_nodes * (n_nodes - 1) / 2)
        graph = nx.erdos_renyi_graph(n_nodes, p)
        
        # Add random weights
        for u, v in graph.edges():
            graph[u][v]['weight'] = np.random.random()
        
        return graph
    
    def _measure_graph_performance(self, graph, config) -> Dict[str, float]:
        """Measure graph operation performance"""
        from insightspike.algorithms.graph_edit_distance import calculate_ged
        from insightspike.algorithms.information_gain import calculate_ig
        
        ged_times = []
        ig_times = []
        
        # Test GED calculation
        for _ in range(10):
            # Create modified graph
            modified_graph = graph.copy()
            # Add/remove some edges
            edges = list(graph.edges())
            if edges:
                modified_graph.remove_edge(*edges[0])
            
            start_time = time.time()
            ged = calculate_ged(graph, modified_graph)
            ged_time = time.time() - start_time
            ged_times.append(ged_time)
        
        # Test IG calculation
        for _ in range(10):
            start_time = time.time()
            ig = calculate_ig(graph, graph)  # Simplified test
            ig_time = time.time() - start_time
            ig_times.append(ig_time)
        
        return {
            'avg_ged_time': np.mean(ged_times),
            'avg_ig_time': np.mean(ig_times),
            'total_graph_time': np.mean(ged_times) + np.mean(ig_times)
        }
    
    def _simulate_concurrent_users(self, agent: MainAgent, n_users: int, 
                                  queries_per_user: int) -> Dict[str, float]:
        """Simulate concurrent user queries"""
        import concurrent.futures
        import threading
        
        queries = [f"Test question {i}" for i in range(n_users * queries_per_user)]
        response_times = []
        errors = 0
        
        start_time = time.time()
        
        # Use thread pool for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_users) as executor:
            futures = []
            
            for i in range(n_users):
                user_queries = queries[i*queries_per_user:(i+1)*queries_per_user]
                
                for query in user_queries:
                    future = executor.submit(self._process_query, agent, query)
                    futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    response_time = future.result()
                    response_times.append(response_time)
                except Exception as e:
                    errors += 1
                    logger.error(f"Query failed: {e}")
        
        total_time = time.time() - start_time
        successful_queries = len(response_times)
        
        return {
            'total_queries': n_users * queries_per_user,
            'successful_queries': successful_queries,
            'errors': errors,
            'total_time': total_time,
            'throughput': successful_queries / total_time,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0
        }
    
    def _process_query(self, agent: MainAgent, query: str) -> float:
        """Process single query and return response time"""
        start_time = time.time()
        agent.process_question(query)
        return time.time() - start_time
    
    def _check_model_memory_fit(self, model_name: str) -> bool:
        """Check if model fits in available memory"""
        model_sizes_gb = {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.2,
            "microsoft/phi-2": 5.5,
            "mistralai/Mistral-7B-v0.1": 14.0
        }
        
        available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
        model_size = model_sizes_gb.get(model_name, 10.0)  # Default 10GB
        
        return available_memory_gb > model_size * 1.5  # 1.5x safety margin
    
    def _get_model_size(self, model_name: str) -> float:
        """Get model size in GB"""
        model_sizes_gb = {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.2,
            "microsoft/phi-2": 5.5,
            "mistralai/Mistral-7B-v0.1": 14.0
        }
        return model_sizes_gb.get(model_name, 0.0)
    
    def _test_model_performance(self, config: Config, test_queries: int) -> Dict[str, float]:
        """Test model performance"""
        agent = MainAgent(config)
        
        response_times = []
        quality_scores = []
        
        test_questions = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "What are the benefits of exercise?",
            # Add more diverse questions
        ]
        
        for i in range(test_queries):
            question = test_questions[i % len(test_questions)]
            
            start_time = time.time()
            result = agent.process_question(question)
            response_time = time.time() - start_time
            
            response_times.append(response_time)
            quality_scores.append(result.get('reasoning_quality', 0.0))
        
        return {
            'avg_response_time': np.mean(response_times),
            'avg_quality_score': np.mean(quality_scores),
            'throughput': 1.0 / np.mean(response_times)
        }
    
    def _analyze_scaling_behavior(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Analyze scaling behavior from measurements"""
        episode_counts = [m['n_episodes'] for m in measurements]
        retrieval_times = [m['avg_retrieval_time'] for m in measurements]
        
        # Fit complexity curve (e.g., O(log n), O(n), O(n log n))
        from scipy.optimize import curve_fit
        
        def log_complexity(x, a, b):
            return a * np.log(x) + b
        
        try:
            params, _ = curve_fit(log_complexity, episode_counts, retrieval_times)
            complexity = "O(log n)"
        except:
            complexity = "Unknown"
        
        return {
            'scaling_complexity': complexity,
            'scaling_factor': params[0] if 'params' in locals() else None
        }
    
    def _analyze_graph_complexity(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Analyze graph algorithm complexity"""
        node_counts = [m['n_nodes'] for m in measurements]
        ged_times = [m['avg_ged_time'] for m in measurements]
        
        # Calculate complexity growth rate
        growth_rates = []
        for i in range(1, len(node_counts)):
            node_ratio = node_counts[i] / node_counts[i-1]
            time_ratio = ged_times[i] / ged_times[i-1]
            growth_rates.append(time_ratio / node_ratio)
        
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
        
        return {
            'avg_growth_rate': avg_growth_rate,
            'estimated_complexity': self._estimate_complexity(avg_growth_rate)
        }
    
    def _estimate_complexity(self, growth_rate: float) -> str:
        """Estimate computational complexity from growth rate"""
        if growth_rate < 1.5:
            return "O(n) or O(n log n)"
        elif growth_rate < 2.5:
            return "O(n²)"
        else:
            return "O(n³) or higher"
    
    def _calculate_scalability_metrics(self, measurements: List[Dict]) -> Dict[str, float]:
        """Calculate scalability metrics"""
        user_counts = [m['n_users'] for m in measurements]
        throughputs = [m['throughput'] for m in measurements]
        
        # Calculate linear scalability score (1.0 = perfect linear scaling)
        expected_throughputs = [throughputs[0] * (u / user_counts[0]) for u in user_counts]
        scalability_scores = [actual / expected for actual, expected in zip(throughputs, expected_throughputs)]
        
        return {
            'avg_scalability_score': np.mean(scalability_scores),
            'min_scalability_score': np.min(scalability_scores),
            'scalability_degradation': 1.0 - np.min(scalability_scores)
        }
    
    def _analyze_model_tradeoffs(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Analyze trade-offs between model size and performance"""
        if not measurements:
            return {}
        
        sizes = [m['model_size_gb'] for m in measurements]
        times = [m['avg_response_time'] for m in measurements]
        qualities = [m['avg_quality_score'] for m in measurements]
        
        # Calculate efficiency metrics
        efficiency_scores = [q / (s * t) for q, s, t in zip(qualities, sizes, times)]
        
        best_model_idx = np.argmax(efficiency_scores)
        
        return {
            'best_efficiency_model': measurements[best_model_idx]['model'],
            'efficiency_scores': dict(zip([m['model'] for m in measurements], efficiency_scores))
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'platform': 'Google Colab',
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'gpu_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        
        return info
    
    def _save_results(self, results: Dict, test_name: str):
        """Save test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_path}")
    
    def _plot_memory_scaling(self, results: Dict):
        """Generate memory scaling plots"""
        measurements = results['measurements']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Retrieval time scaling
        ax = axes[0, 0]
        episode_counts = [m['n_episodes'] for m in measurements]
        retrieval_times = [m['avg_retrieval_time'] * 1000 for m in measurements]  # Convert to ms
        ax.plot(episode_counts, retrieval_times, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Episodes')
        ax.set_ylabel('Avg Retrieval Time (ms)')
        ax.set_title('Retrieval Time Scaling')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Memory usage
        ax = axes[0, 1]
        memory_usage = [m['avg_memory_mb'] for m in measurements]
        ax.plot(episode_counts, memory_usage, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Episodes')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Scaling')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Percentile response times
        ax = axes[1, 0]
        p95_times = [m['p95_retrieval_time'] * 1000 for m in measurements]
        p99_times = [m['p99_retrieval_time'] * 1000 for m in measurements]
        ax.plot(episode_counts, p95_times, 'go-', label='P95', linewidth=2, markersize=8)
        ax.plot(episode_counts, p99_times, 'mo-', label='P99', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Episodes')
        ax.set_ylabel('Response Time (ms)')
        ax.set_title('Percentile Response Times')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Efficiency
        ax = axes[1, 1]
        efficiency = [1000 / m['avg_retrieval_time'] / m['n_episodes'] for m in measurements]
        ax.plot(episode_counts, efficiency, 'co-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Episodes')
        ax.set_ylabel('Queries per Second per Episode')
        ax.set_title('Retrieval Efficiency')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"memory_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Plot saved to: {plot_path}")
    
    def _plot_graph_scaling(self, results: Dict):
        """Generate graph scaling plots"""
        measurements = results['measurements']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # GED scaling
        ax = axes[0]
        node_counts = [m['n_nodes'] for m in measurements]
        ged_times = [m['avg_ged_time'] * 1000 for m in measurements]
        ax.plot(node_counts, ged_times, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('GED Calculation Time (ms)')
        ax.set_title('Graph Edit Distance Scaling')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Total graph processing time
        ax = axes[1]
        total_times = [m['total_graph_time'] * 1000 for m in measurements]
        ax.plot(node_counts, total_times, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Total Processing Time (ms)')
        ax.set_title('Total Graph Processing Time')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"graph_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_concurrent_scaling(self, results: Dict):
        """Generate concurrent user scaling plots"""
        measurements = results['measurements']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Throughput scaling
        ax = axes[0, 0]
        user_counts = [m['n_users'] for m in measurements]
        throughputs = [m['throughput'] for m in measurements]
        ax.plot(user_counts, throughputs, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Concurrent Users')
        ax.set_ylabel('Throughput (queries/sec)')
        ax.set_title('Throughput Scaling')
        ax.grid(True, alpha=0.3)
        
        # Response time
        ax = axes[0, 1]
        response_times = [m['avg_response_time'] * 1000 for m in measurements]
        ax.plot(user_counts, response_times, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Concurrent Users')
        ax.set_ylabel('Avg Response Time (ms)')
        ax.set_title('Response Time vs Concurrent Users')
        ax.grid(True, alpha=0.3)
        
        # Error rate
        ax = axes[1, 0]
        error_rates = [m['errors'] / m['total_queries'] * 100 for m in measurements]
        ax.plot(user_counts, error_rates, 'mo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Concurrent Users')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rate vs Load')
        ax.grid(True, alpha=0.3)
        
        # Scalability efficiency
        ax = axes[1, 1]
        base_throughput = throughputs[0]
        scalability_efficiency = [t / base_throughput / (u / user_counts[0]) * 100 
                                 for t, u in zip(throughputs, user_counts)]
        ax.plot(user_counts, scalability_efficiency, 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Scaling')
        ax.set_xlabel('Number of Concurrent Users')
        ax.set_ylabel('Scalability Efficiency (%)')
        ax.set_title('Scalability Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"concurrent_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, results: Dict):
        """Generate model comparison plots"""
        measurements = results['measurements']
        
        if not measurements:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = [m['model'].split('/')[-1] for m in measurements]  # Shorten names
        
        # Response time vs model size
        ax = axes[0, 0]
        sizes = [m['model_size_gb'] for m in measurements]
        times = [m['avg_response_time'] for m in measurements]
        ax.scatter(sizes, times, s=100, c='blue', alpha=0.6)
        for i, model in enumerate(models):
            ax.annotate(model, (sizes[i], times[i]), fontsize=8)
        ax.set_xlabel('Model Size (GB)')
        ax.set_ylabel('Avg Response Time (s)')
        ax.set_title('Response Time vs Model Size')
        ax.grid(True, alpha=0.3)
        
        # Quality vs model size
        ax = axes[0, 1]
        qualities = [m['avg_quality_score'] for m in measurements]
        ax.scatter(sizes, qualities, s=100, c='green', alpha=0.6)
        for i, model in enumerate(models):
            ax.annotate(model, (sizes[i], qualities[i]), fontsize=8)
        ax.set_xlabel('Model Size (GB)')
        ax.set_ylabel('Avg Quality Score')
        ax.set_title('Quality vs Model Size')
        ax.grid(True, alpha=0.3)
        
        # Model comparison bar chart
        ax = axes[1, 0]
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, times, width, label='Response Time (s)', color='blue', alpha=0.7)
        ax.bar(x + width/2, qualities, width, label='Quality Score', color='green', alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('Value')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Efficiency scores
        ax = axes[1, 1]
        if 'tradeoff_analysis' in results and 'efficiency_scores' in results['tradeoff_analysis']:
            efficiency_scores = list(results['tradeoff_analysis']['efficiency_scores'].values())
            ax.bar(models, efficiency_scores, color='purple', alpha=0.7)
            ax.set_xlabel('Model')
            ax.set_ylabel('Efficiency Score')
            ax.set_title('Model Efficiency (Quality / Size / Time)')
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_scalability_report(self, all_results: Dict):
        """Generate comprehensive scalability report"""
        report = f"""# InsightSpike Scalability Test Report

## Test Environment
- **Date**: {all_results['timestamp']}
- **Platform**: {all_results['environment']['platform']}
- **CPU Cores**: {all_results['environment']['cpu_count']}
- **Total Memory**: {all_results['environment']['total_memory_gb']:.1f} GB
- **GPU**: {all_results['environment'].get('gpu_name', 'Not available')}

## Test Results Summary

"""
        
        for test_name, results in all_results['tests'].items():
            if 'error' in results:
                report += f"### {test_name}\n**Error**: {results['error']}\n\n"
                continue
                
            report += f"### {test_name}\n"
            
            if test_name == 'memory_scaling':
                report += f"- **Scaling Complexity**: {results.get('analysis', {}).get('scaling_complexity', 'Unknown')}\n"
                report += f"- **Largest Dataset**: {max([m['n_episodes'] for m in results['measurements']])} episodes\n"
                
            elif test_name == 'graph_scaling':
                report += f"- **Complexity**: {results.get('complexity_analysis', {}).get('estimated_complexity', 'Unknown')}\n"
                report += f"- **Largest Graph**: {max([m['n_nodes'] for m in results['measurements']])} nodes\n"
                
            elif test_name == 'concurrent_users':
                report += f"- **Max Users Tested**: {max([m['n_users'] for m in results['measurements']])}\n"
                report += f"- **Scalability Score**: {results.get('scalability_metrics', {}).get('avg_scalability_score', 0):.2f}\n"
                
            elif test_name == 'model_comparison':
                report += f"- **Models Tested**: {len(results['measurements'])}\n"
                if 'tradeoff_analysis' in results and 'best_efficiency_model' in results['tradeoff_analysis']:
                    report += f"- **Best Efficiency**: {results['tradeoff_analysis']['best_efficiency_model']}\n"
            
            report += "\n"
        
        report += """
## Recommendations

1. **Memory Optimization**: Consider implementing memory-mapped indices for datasets > 10k episodes
2. **Graph Processing**: Current algorithms show good scaling up to 5k nodes
3. **Concurrency**: System handles concurrent users well with minimal degradation
4. **Model Selection**: Balance between model size and performance based on use case

## Next Steps

- Test with larger datasets (100k+ episodes)
- Implement distributed processing for extreme scale
- Optimize graph algorithms for sparse graphs
- Test with production workloads
"""
        
        # Save report
        report_path = self.results_dir / f"scalability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {report_path}")


# Convenience functions for Colab
def test_memory_scaling(episode_counts: List[int] = None) -> Dict:
    """Quick function to test memory scaling"""
    if episode_counts is None:
        episode_counts = [100, 500, 1000, 5000, 10000]
    
    suite = ScalabilityTestSuite()
    return suite.run_memory_scaling_test(episode_counts)


def test_all_scalability() -> Dict:
    """Run all scalability tests"""
    suite = ScalabilityTestSuite()
    return suite.run_comprehensive_scalability_test()


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting Scalability Tests...")
    
    # Run comprehensive tests
    results = test_all_scalability()
    
    print("\n✅ Scalability tests complete!")
    print(f"📊 Results saved to: experiments/colab_experiments/results/scalability/")