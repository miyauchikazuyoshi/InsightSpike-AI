#!/usr/bin/env python3
"""
Run Comprehensive Comparison Experiment
Compare InsightSpike-AI with Baseline RAG Systems
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.insightspike.core.agents.main_agent import InsightSpikeAI
from src.insightspike.core.agents.agent_builder import AgentBuilder
from build_baseline_rag import BaselineRAG, HybridBaselineRAG


class RAGComparison:
    """Compare different RAG systems"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize systems
        self.systems = {}
        self.test_data = []
        self.results = {
            'accuracy': {},
            'speed': {},
            'compression': {}
        }
    
    def load_test_questions(self) -> List[Dict]:
        """Load test questions and answers"""
        # Try to load from existing test data
        test_file = Path("experiments/gedig_embedding_evaluation/data/processed/test_questions.json")
        
        if test_file.exists():
            with open(test_file, 'r') as f:
                return json.load(f)
        
        # Create synthetic test questions
        test_questions = []
        for i in range(50):
            test_questions.append({
                'question': f"What is the information about topic {i % 10}?",
                'answer': f"topic {i % 10}",
                'type': 'factual'
            })
        
        return test_questions
    
    def setup_systems(self):
        """Initialize all RAG systems"""
        print("Setting up RAG systems...")
        
        # 1. InsightSpike-AI
        print("1. Initializing InsightSpike-AI...")
        agent_builder = AgentBuilder()
        self.systems['insightspike'] = agent_builder.build_main_agent()
        
        # 2. Standard Baseline RAG
        print("2. Loading Standard Baseline RAG...")
        baseline = BaselineRAG()
        baseline_path = Path("experiment_1/comparison_experiment/data/baseline_rag")
        if baseline_path.exists():
            baseline.load(baseline_path)
        else:
            # Build if not exists
            from build_baseline_rag import prepare_test_data
            docs, meta = prepare_test_data()
            baseline.build_index(docs, meta)
        self.systems['baseline'] = baseline
        
        # 3. Hybrid Baseline RAG
        print("3. Loading Hybrid Baseline RAG...")
        hybrid = HybridBaselineRAG()
        hybrid_path = Path("experiment_1/comparison_experiment/data/hybrid_baseline_rag")
        if hybrid_path.exists():
            hybrid.load(hybrid_path)
        else:
            # Build if not exists
            from build_baseline_rag import prepare_test_data
            docs, meta = prepare_test_data()
            hybrid.build_index(docs, meta)
        self.systems['hybrid_baseline'] = hybrid
    
    def evaluate_accuracy(self, questions: List[Dict], k: int = 5):
        """Evaluate retrieval accuracy"""
        print("\n=== Evaluating Accuracy ===")
        
        accuracy_results = {
            system_name: {
                'correct': 0,
                'total': 0,
                'mrr': [],
                'recall_at_k': []
            }
            for system_name in self.systems.keys()
        }
        
        for q_data in questions[:20]:  # Limit for testing
            question = q_data['question']
            expected = q_data['answer']
            
            for system_name, system in self.systems.items():
                try:
                    if system_name == 'insightspike':
                        # InsightSpike query
                        response = system.process_input(question)
                        # Extract relevant info (simplified)
                        found = expected.lower() in str(response).lower()
                        rank = 1 if found else 0
                    else:
                        # Baseline systems
                        results, _ = system.search(question, k=k)
                        found = False
                        rank = 0
                        
                        for i, (doc, score, meta) in enumerate(results):
                            if expected.lower() in doc.lower():
                                found = True
                                rank = 1 / (i + 1)  # Reciprocal rank
                                break
                    
                    accuracy_results[system_name]['total'] += 1
                    if found:
                        accuracy_results[system_name]['correct'] += 1
                        accuracy_results[system_name]['recall_at_k'].append(1)
                    else:
                        accuracy_results[system_name]['recall_at_k'].append(0)
                    
                    accuracy_results[system_name]['mrr'].append(rank)
                    
                except Exception as e:
                    print(f"Error evaluating {system_name}: {e}")
        
        # Calculate metrics
        for system_name, results in accuracy_results.items():
            if results['total'] > 0:
                results['accuracy'] = results['correct'] / results['total']
                results['mrr_score'] = np.mean(results['mrr'])
                results['recall_at_k_score'] = np.mean(results['recall_at_k'])
            else:
                results['accuracy'] = 0
                results['mrr_score'] = 0
                results['recall_at_k_score'] = 0
        
        self.results['accuracy'] = accuracy_results
        return accuracy_results
    
    def evaluate_speed(self, queries: List[str], iterations: int = 10):
        """Evaluate query speed"""
        print("\n=== Evaluating Speed ===")
        
        speed_results = {
            system_name: {
                'times': [],
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
            for system_name in self.systems.keys()
        }
        
        for query in queries[:5]:  # Limit queries
            for _ in range(iterations):
                for system_name, system in self.systems.items():
                    try:
                        start_time = time.time()
                        
                        if system_name == 'insightspike':
                            _ = system.process_input(query)
                        else:
                            _, _ = system.search(query, k=5)
                        
                        elapsed = time.time() - start_time
                        speed_results[system_name]['times'].append(elapsed)
                        
                    except Exception as e:
                        print(f"Error testing speed for {system_name}: {e}")
        
        # Calculate statistics
        for system_name, results in speed_results.items():
            if results['times']:
                results['avg_time'] = np.mean(results['times'])
                results['min_time'] = np.min(results['times'])
                results['max_time'] = np.max(results['times'])
                results['std_time'] = np.std(results['times'])
        
        self.results['speed'] = speed_results
        return speed_results
    
    def evaluate_compression(self):
        """Evaluate storage compression"""
        print("\n=== Evaluating Compression ===")
        
        compression_results = {}
        
        # InsightSpike compression
        insightspike_data = {
            'episodes': Path("data/episodes.json"),
            'graph': Path("data/graph_pyg.pt"),
            'index': Path("data/index.faiss")
        }
        
        insightspike_size = 0
        for name, path in insightspike_data.items():
            if path.exists():
                size = path.stat().st_size
                insightspike_size += size
        
        compression_results['insightspike'] = {
            'total_size': insightspike_size,
            'size_per_doc': insightspike_size / 100  # Assuming 100 docs
        }
        
        # Baseline compression
        baseline_size = 0
        baseline_path = Path("experiment_1/comparison_experiment/data/baseline_rag")
        if baseline_path.exists():
            for file in baseline_path.iterdir():
                baseline_size += file.stat().st_size
        
        compression_results['baseline'] = {
            'total_size': baseline_size,
            'size_per_doc': baseline_size / 100
        }
        
        # Calculate compression ratios
        if baseline_size > 0:
            compression_results['compression_ratio'] = baseline_size / insightspike_size
        else:
            compression_results['compression_ratio'] = 0
        
        self.results['compression'] = compression_results
        return compression_results
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        print("\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Accuracy comparison
        ax = axes[0, 0]
        systems = list(self.results['accuracy'].keys())
        accuracies = [self.results['accuracy'][s].get('accuracy', 0) for s in systems]
        ax.bar(systems, accuracies)
        ax.set_title('Retrieval Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # 2. Speed comparison
        ax = axes[0, 1]
        avg_times = [self.results['speed'][s].get('avg_time', 0) for s in systems]
        ax.bar(systems, avg_times)
        ax.set_title('Average Query Time')
        ax.set_ylabel('Time (seconds)')
        
        # 3. MRR comparison
        ax = axes[1, 0]
        mrr_scores = [self.results['accuracy'][s].get('mrr_score', 0) for s in systems]
        ax.bar(systems, mrr_scores)
        ax.set_title('Mean Reciprocal Rank (MRR)')
        ax.set_ylabel('MRR Score')
        ax.set_ylim(0, 1)
        
        # 4. Compression comparison
        ax = axes[1, 1]
        compression_systems = ['insightspike', 'baseline']
        sizes = [self.results['compression'][s]['size_per_doc'] for s in compression_systems]
        ax.bar(compression_systems, sizes)
        ax.set_title('Storage Size per Document')
        ax.set_ylabel('Size (bytes)')
        
        plt.tight_layout()
        viz_path = self.results_dir / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Visualization saved to: {viz_path}")
    
    def save_results(self):
        """Save all results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results summary
        summary = {
            'timestamp': timestamp,
            'systems_evaluated': list(self.systems.keys()),
            'accuracy_results': self.results['accuracy'],
            'speed_results': self.results['speed'],
            'compression_results': self.results['compression'],
            'summary': {
                'best_accuracy': max(self.results['accuracy'].items(), 
                                   key=lambda x: x[1].get('accuracy', 0))[0],
                'fastest': min(self.results['speed'].items(), 
                             key=lambda x: x[1].get('avg_time', float('inf')))[0],
                'best_compression': 'insightspike' if self.results['compression'].get('compression_ratio', 0) > 1 else 'baseline'
            }
        }
        
        # Save JSON results
        results_file = self.results_dir / f"comparison_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return summary


def main():
    """Run comparison experiment"""
    print("Starting RAG System Comparison Experiment")
    
    # Initialize comparison
    comparison = RAGComparison("experiment_1/comparison_experiment/results")
    
    # Setup systems
    comparison.setup_systems()
    
    # Load test questions
    test_questions = comparison.load_test_questions()
    print(f"Loaded {len(test_questions)} test questions")
    
    # Run evaluations
    # 1. Accuracy evaluation
    accuracy_results = comparison.evaluate_accuracy(test_questions)
    
    # 2. Speed evaluation
    test_queries = [q['question'] for q in test_questions[:5]]
    speed_results = comparison.evaluate_speed(test_queries)
    
    # 3. Compression evaluation
    compression_results = comparison.evaluate_compression()
    
    # Create visualizations
    comparison.create_visualizations()
    
    # Save results
    summary = comparison.save_results()
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Best Accuracy: {summary['summary']['best_accuracy']}")
    print(f"Fastest System: {summary['summary']['fastest']}")
    print(f"Best Compression: {summary['summary']['best_compression']}")
    
    print("\nDetailed Results:")
    for system in comparison.systems.keys():
        print(f"\n{system}:")
        print(f"  - Accuracy: {accuracy_results[system].get('accuracy', 0):.2%}")
        print(f"  - MRR: {accuracy_results[system].get('mrr_score', 0):.3f}")
        print(f"  - Avg Query Time: {speed_results[system].get('avg_time', 0):.3f}s")


if __name__ == "__main__":
    main()