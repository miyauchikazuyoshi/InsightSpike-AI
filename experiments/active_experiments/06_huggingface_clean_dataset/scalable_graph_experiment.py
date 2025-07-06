#!/usr/bin/env python3
"""
Experiment 6: Scalable Graph Growth with Improved Implementation
================================================================

Tests the new scalable graph features implemented in src based on experiment_5 learnings.
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.monitoring.graph_monitor import GraphOperationMonitor, create_default_monitor
from insightspike.utils.graph_importance import GraphImportanceCalculator
from insightspike.core.config import get_config


class ScalableGraphExperiment:
    """Run scalability experiments with new implementation."""
    
    def __init__(self):
        self.config = get_config()
        self.monitor = create_default_monitor()
        self.agent = None
        self.all_documents = []
        self.metrics = {
            'episodes': [],
            'graph_nodes': [],
            'graph_edges': [],
            'build_times': [],
            'conflicts_detected': [],
            'splits_performed': [],
            'importance_scores': [],
            'timestamps': []
        }
        
    def initialize(self):
        """Initialize experiment with enhanced components."""
        print("=== Experiment 6: Scalable Graph with Enhanced Implementation ===")
        print(f"Start time: {datetime.now()}")
        print("\nFeatures enabled:")
        print("  ✓ FAISS-based scalable graph builder")
        print("  ✓ Conflict detection and auto-splitting")
        print("  ✓ Graph-based dynamic importance")
        print("  ✓ Performance monitoring")
        print("  ✓ Enhanced memory management\n")
        
        # Initialize agent with enhanced memory
        self.agent = MainAgent()
        
        # Replace memory manager with enhanced version
        if hasattr(self.agent, 'l2_memory'):
            print("Upgrading to enhanced scalable memory manager...")
            old_memory = self.agent.l2_memory
            self.agent.l2_memory = L2EnhancedScalableMemory(
                dim=self.config.embedding.dimension,
                config=self.config,
                use_scalable_graph=True
            )
            # Copy existing episodes if any
            if hasattr(old_memory, 'episodes'):
                self.agent.l2_memory.episodes = old_memory.episodes
        
        self.agent.initialize()
        
        # Get initial state
        initial_state = self._get_system_state()
        print(f"Initial State:")
        print(f"  Episodes: {initial_state['episodes']}")
        print(f"  Graph nodes: {initial_state['graph_nodes']}")
        print(f"  Graph edges: {initial_state['graph_edges']}")
        
        return initial_state
    
    def generate_dataset(self, count: int) -> List[str]:
        """Generate diverse dataset for testing."""
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
            
            # Add variations including conflicts
            if i % 10 == 0:
                text += f" This was discovered in {2020 + i % 5}."
            if i % 15 == 0:
                text += f" The impact is estimated at ${random.randint(1, 100)} billion."
            if i % 20 == 0:
                # Add conflicting information occasionally
                if "revolutionizing" in text:
                    text += " However, some experts believe the impact is overstated."
                elif "breakthrough" in text:
                    text += " Critics argue these advances are incremental at best."
            
            dataset.append(text)
        
        return dataset
    
    def add_episode_with_monitoring(self, text: str) -> Dict[str, Any]:
        """Add episode with full monitoring and analysis."""
        start_time = time.time()
        
        try:
            # Add episode through agent
            result = self.agent.add_episode_with_graph_update(text)
            
            # Get additional metrics from enhanced memory
            if hasattr(self.agent.l2_memory, 'scalable_graph'):
                graph_stats = self.agent.l2_memory.get_graph_stats()
                result['graph_stats'] = graph_stats
                
                # Check for recent conflicts
                if hasattr(self.agent.l2_memory, 'recent_conflicts'):
                    result['conflicts'] = len(self.agent.l2_memory.recent_conflicts)
            
            result['processing_time'] = time.time() - start_time
            
            # Track document for graph analysis
            if result.get('success', False):
                self.all_documents.append({
                    "text": text,
                    "timestamp": time.time(),
                    "processing_time": result['processing_time']
                })
            
            return result
            
        except Exception as e:
            print(f"Error adding episode: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def run_phase1_basic_test(self):
        """Phase 1: Basic functionality test (100 documents)."""
        print("\n=== Phase 1: Basic Functionality Test (100 documents) ===")
        
        dataset = self.generate_dataset(100)
        successful = 0
        failed = 0
        total_conflicts = 0
        total_build_time = 0
        
        for i, text in enumerate(dataset):
            result = self.add_episode_with_monitoring(text)
            
            if result.get('success', False):
                successful += 1
                total_build_time += result.get('processing_time', 0)
                
                if 'conflicts' in result:
                    total_conflicts += result['conflicts']
                
                # Report progress every 20 documents
                if (i + 1) % 20 == 0:
                    state = self._get_system_state()
                    print(f"\nProgress: {i + 1}/100")
                    print(f"  Episodes: {state['episodes']}")
                    print(f"  Graph nodes: {state['graph_nodes']}")
                    print(f"  Graph edges: {state['graph_edges']}")
                    print(f"  Conflicts detected: {total_conflicts}")
                    print(f"  Avg processing time: {total_build_time/(i+1):.3f}s")
                    
                    self._record_metrics(i + 1)
            else:
                failed += 1
                print(f"Failed to add episode {i}: {result.get('error', 'Unknown error')}")
        
        print(f"\nPhase 1 Complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total conflicts: {total_conflicts}")
        print(f"  Average processing time: {total_build_time/max(1, successful):.3f}s")
        
        return successful, failed
    
    def run_phase2_performance_test(self):
        """Phase 2: Performance test (1000 documents)."""
        print("\n=== Phase 2: Performance Test (1000 documents) ===")
        
        dataset = self.generate_dataset(1000)
        batch_size = 100
        
        phase_start = time.time()
        successful = 0
        failed = 0
        
        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset[batch_idx:batch_idx + batch_size]
            batch_start = time.time()
            
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{len(dataset)//batch_size}")
            
            for text in batch:
                result = self.add_episode_with_monitoring(text)
                if result.get('success', False):
                    successful += 1
                else:
                    failed += 1
            
            batch_time = time.time() - batch_start
            state = self._get_system_state()
            
            print(f"  Batch time: {batch_time:.2f}s ({batch_time/len(batch):.3f}s per doc)")
            print(f"  Total episodes: {state['episodes']}")
            print(f"  Graph: {state['graph_nodes']} nodes, {state['graph_edges']} edges")
            
            # Check monitoring metrics
            if self.monitor:
                summary = self.monitor.get_operation_summary()
                if summary:
                    print(f"  Operation stats: {len(summary)} operation types tracked")
            
            self._record_metrics(batch_idx + len(batch))
        
        phase_time = time.time() - phase_start
        
        print(f"\nPhase 2 Complete:")
        print(f"  Total time: {phase_time:.2f}s")
        print(f"  Documents processed: {successful + failed}")
        print(f"  Success rate: {successful/(successful + failed)*100:.1f}%")
        print(f"  Average time per document: {phase_time/(successful + failed):.3f}s")
        
        return successful, failed
    
    def run_phase3_scalability_analysis(self):
        """Phase 3: Analyze scalability and performance."""
        print("\n=== Phase 3: Scalability Analysis ===")
        
        # Get final state
        final_state = self._get_system_state()
        
        # Analyze graph density
        if final_state['graph_nodes'] > 1:
            max_edges = final_state['graph_nodes'] * (final_state['graph_nodes'] - 1)
            density = final_state['graph_edges'] / max_edges
            avg_degree = final_state['graph_edges'] / final_state['graph_nodes']
            
            print(f"\nGraph Structure Analysis:")
            print(f"  Nodes: {final_state['graph_nodes']}")
            print(f"  Edges: {final_state['graph_edges']}")
            print(f"  Density: {density:.4f}")
            print(f"  Average degree: {avg_degree:.2f}")
        
        # Analyze monitoring data
        if self.monitor:
            print(f"\nPerformance Analysis:")
            summary = self.monitor.get_operation_summary()
            
            for op, stats in summary.items():
                print(f"\n  {op}:")
                print(f"    Count: {stats['count']}")
                print(f"    Avg duration: {stats['avg_duration']:.3f}s")
                print(f"    Min/Max: {stats['min_duration']:.3f}s / {stats['max_duration']:.3f}s")
            
            # Check for anomalies
            anomalies = self.monitor.detect_anomalies()
            if anomalies:
                print(f"\n  Detected {len(anomalies)} anomalies:")
                for anomaly in anomalies[:5]:  # Show first 5
                    print(f"    - {anomaly['type']}: {anomaly.get('operation', 'N/A')}")
        
        # Test graph importance calculation
        if hasattr(self.agent, 'l3_graph') and self.agent.l3_graph:
            print(f"\nTesting Graph Importance Calculation...")
            calculator = GraphImportanceCalculator()
            
            # Get current graph
            if hasattr(self.agent.l3_graph, 'previous_graph'):
                graph = self.agent.l3_graph.previous_graph
                if graph and graph.num_nodes > 0:
                    # Get top important nodes
                    top_nodes = calculator.get_top_k_important(graph, k=5)
                    print(f"  Top 5 most important nodes:")
                    for idx, (node, score) in enumerate(top_nodes):
                        print(f"    {idx+1}. Node {node}: importance={score:.3f}")
    
    def save_results(self):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'experiment': 'Scalable Graph Growth with Enhanced Implementation',
            'timestamp': timestamp,
            'config': {
                'use_scalable_graph': True,
                'top_k': self.config.scalable_graph.top_k_neighbors if hasattr(self.config, 'scalable_graph') else 50,
                'batch_size': self.config.scalable_graph.batch_size if hasattr(self.config, 'scalable_graph') else 1000,
                'conflict_detection': True,
                'graph_importance': True
            },
            'metrics': self.metrics,
            'final_state': self._get_system_state(),
            'total_documents': len(self.all_documents)
        }
        
        # Export monitoring data
        if self.monitor:
            monitor_file = Path(f'experiment_6/monitor_data_{timestamp}.json')
            self.monitor.export_metrics(monitor_file)
            results['monitor_data_file'] = str(monitor_file)
        
        # Save main results
        filename = f'experiment_6/results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        
        # Print summary
        print("\n=== Experiment Summary ===")
        print(f"Documents processed: {len(self.all_documents)}")
        print(f"Final graph: {results['final_state']['graph_nodes']} nodes, "
              f"{results['final_state']['graph_edges']} edges")
        
        if self.metrics['build_times']:
            avg_time = sum(self.metrics['build_times']) / len(self.metrics['build_times'])
            print(f"Average processing time: {avg_time:.3f}s")
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        state = {
            'episodes': 0,
            'graph_nodes': 0,
            'graph_edges': 0
        }
        
        try:
            # Get episode count
            if hasattr(self.agent, 'l2_memory') and hasattr(self.agent.l2_memory, 'episodes'):
                state['episodes'] = len(self.agent.l2_memory.episodes)
            
            # Get graph stats
            if hasattr(self.agent.l2_memory, 'get_graph_stats'):
                graph_stats = self.agent.l2_memory.get_graph_stats()
                if graph_stats.get('graph_enabled', False):
                    state['graph_nodes'] = graph_stats.get('nodes', 0)
                    state['graph_edges'] = graph_stats.get('edges', 0)
            
            # Alternative: check L3 graph
            elif hasattr(self.agent, 'l3_graph') and hasattr(self.agent.l3_graph, 'previous_graph'):
                graph = self.agent.l3_graph.previous_graph
                if graph:
                    state['graph_nodes'] = graph.num_nodes
                    state['graph_edges'] = graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0
        
        except Exception as e:
            print(f"Warning: Error getting system state: {e}")
        
        return state
    
    def _record_metrics(self, document_count: int):
        """Record current metrics."""
        state = self._get_system_state()
        
        self.metrics['timestamps'].append(document_count)
        self.metrics['episodes'].append(state['episodes'])
        self.metrics['graph_nodes'].append(state['graph_nodes'])
        self.metrics['graph_edges'].append(state['graph_edges'])
        
        # Record timing if available
        if self.all_documents:
            recent_times = [d['processing_time'] for d in self.all_documents[-10:]]
            avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
            self.metrics['build_times'].append(avg_time)


def main():
    """Run the experiment."""
    print("Starting Scalable Graph Experiment with Enhanced Implementation")
    print("=" * 60)
    
    experiment = ScalableGraphExperiment()
    
    try:
        # Initialize
        experiment.initialize()
        
        # Run phases
        experiment.run_phase1_basic_test()
        experiment.run_phase2_performance_test()
        experiment.run_phase3_scalability_analysis()
        
        # Save results
        experiment.save_results()
        
        print(f"\nExperiment completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial results
        try:
            experiment.save_results()
        except:
            print("Failed to save results")


if __name__ == "__main__":
    main()