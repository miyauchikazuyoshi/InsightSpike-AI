#!/usr/bin/env python3
"""
Test RAG Performance Comparison
================================

Compare InsightSpike with standard RAG using saved data.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import random

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config
from sentence_transformers import SentenceTransformer


class RAGPerformanceTester:
    """Test and compare RAG performance."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        self.embedder = None
        
        # Use experiment-specific directory
        self.experiment_dir = Path(__file__).parent
        self.data_dir = self.experiment_dir / "data"
        
        # Configure paths
        self.config.paths.data_dir = str(self.data_dir)
        self.config.memory.index_file = str(self.data_dir / "index.faiss")
        self.config.reasoning.graph_file = str(self.data_dir / "graph_pyg.pt")
        
        # Load Q&A pairs
        qa_file = self.data_dir / "qa_pairs.json"
        with open(qa_file, 'r') as f:
            self.qa_pairs = json.load(f)
        
        print(f"Loaded {len(self.qa_pairs)} Q&A pairs for testing")
    
    def initialize_for_testing(self):
        """Initialize components for testing."""
        print("\n=== Initializing Components ===")
        
        # Initialize agent
        self.agent = MainAgent(self.config)
        self.agent.l2_memory = L2EnhancedScalableMemory(
            dim=self.config.embedding.dimension,
            config=self.config,
            use_scalable_graph=True
        )
        
        # Load saved state
        if self.agent.load_state():
            print(f"✓ Loaded state with {len(self.agent.l2_memory.episodes)} episodes")
        else:
            print("✗ Failed to load state")
            return False
        
        # Initialize embedder for standard RAG
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        return True
    
    def test_standard_rag(self, test_size: int = 100) -> Dict:
        """Test standard semantic search RAG."""
        print("\n=== Testing Standard RAG (Semantic Search Only) ===")
        
        # Sample test questions
        test_indices = random.sample(range(len(self.qa_pairs)), min(test_size, len(self.qa_pairs)))
        
        correct = 0
        total = 0
        response_times = []
        
        for idx in test_indices:
            qa = self.qa_pairs[idx]
            start_time = time.time()
            
            # Encode question
            query_vec = self.embedder.encode(qa['question'])
            
            # Search in FAISS index
            if hasattr(self.agent.l2_memory, 'index') and self.agent.l2_memory.index:
                # Search for top 5 similar episodes
                D, I = self.agent.l2_memory.index.search(
                    query_vec.reshape(1, -1).astype(np.float32), 
                    k=5
                )
                
                # Check if answer is in retrieved episodes
                found = False
                for i in I[0]:
                    if i >= 0 and i < len(self.agent.l2_memory.episodes):
                        ep = self.agent.l2_memory.episodes[i]
                        ep_text = ep.text if hasattr(ep, 'text') else str(ep)
                        
                        # Check if the answer is in the episode
                        if qa['answer'].lower() in ep_text.lower():
                            correct += 1
                            found = True
                            break
                
                if not found and total < 5:  # Debug first few misses
                    print(f"\nMissed Q: {qa['question'][:50]}...")
                    print(f"Expected A: {qa['answer']}")
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            total += 1
            
            if total % 20 == 0:
                print(f"Progress: {total}/{test_size}")
        
        avg_time = np.mean(response_times)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nStandard RAG Results:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"  Avg response time: {avg_time:.3f}s")
        print(f"  Total time: {sum(response_times):.1f}s")
        
        return {
            'method': 'Standard RAG (Semantic Search)',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_response_time': avg_time,
            'total_time': sum(response_times),
            'response_times': response_times
        }
    
    def test_insightspike(self, test_size: int = 100) -> Dict:
        """Test InsightSpike with full graph-based retrieval."""
        print("\n=== Testing InsightSpike (Graph-Enhanced RAG) ===")
        
        # Use same test indices for fair comparison
        test_indices = random.sample(range(len(self.qa_pairs)), min(test_size, len(self.qa_pairs)))
        
        correct = 0
        total = 0
        response_times = []
        
        for idx in test_indices:
            qa = self.qa_pairs[idx]
            start_time = time.time()
            
            try:
                # Use InsightSpike's question processing
                result = self.agent.process_question(
                    qa['question'], 
                    max_cycles=1,  # Single cycle for speed
                    verbose=False
                )
                
                response = result.get('response', '')
                
                # Check if answer is in response
                if qa['answer'].lower() in response.lower():
                    correct += 1
                elif total < 5:  # Debug first few misses
                    print(f"\nMissed Q: {qa['question'][:50]}...")
                    print(f"Expected A: {qa['answer']}")
                    print(f"Got: {response[:100]}...")
                
            except Exception as e:
                print(f"Error: {e}")
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            total += 1
            
            if total % 20 == 0:
                print(f"Progress: {total}/{test_size}")
        
        avg_time = np.mean(response_times)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nInsightSpike Results:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"  Avg response time: {avg_time:.3f}s")
        print(f"  Total time: {sum(response_times):.1f}s")
        
        return {
            'method': 'InsightSpike (Graph-Enhanced)',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_response_time': avg_time,
            'total_time': sum(response_times),
            'response_times': response_times
        }
    
    def compare_methods(self, test_size: int = 100):
        """Run comparison between methods."""
        print(f"\n=== PERFORMANCE COMPARISON (n={test_size}) ===")
        print(f"Knowledge base: {len(self.agent.l2_memory.episodes)} episodes")
        
        # Test both methods
        rag_results = self.test_standard_rag(test_size)
        insight_results = self.test_insightspike(test_size)
        
        # Calculate improvements
        accuracy_diff = insight_results['accuracy'] - rag_results['accuracy']
        speed_ratio = rag_results['avg_response_time'] / insight_results['avg_response_time']
        
        # Generate report
        report = {
            'experiment': 'Experiment 8: RAG Performance Comparison',
            'timestamp': datetime.now().isoformat(),
            'test_size': test_size,
            'knowledge_base': {
                'episodes': len(self.agent.l2_memory.episodes),
                'qa_pairs': len(self.qa_pairs)
            },
            'results': {
                'standard_rag': rag_results,
                'insightspike': insight_results
            },
            'comparison': {
                'accuracy_improvement': f"{accuracy_diff*100:+.1f}%",
                'speed_ratio': f"{speed_ratio:.2f}x",
                'winner': 'InsightSpike' if accuracy_diff > 0 else 'Standard RAG'
            }
        }
        
        # Print comparison
        print("\n" + "="*60)
        print("📊 COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Method':<30} {'Accuracy':<15} {'Avg Time':<15}")
        print("-"*60)
        print(f"{'Standard RAG':<30} {rag_results['accuracy']:.1%} ({rag_results['correct']}/{rag_results['total']}){'':<5} {rag_results['avg_response_time']:.3f}s")
        print(f"{'InsightSpike':<30} {insight_results['accuracy']:.1%} ({insight_results['correct']}/{insight_results['total']}){'':<5} {insight_results['avg_response_time']:.3f}s")
        print("-"*60)
        print(f"{'Improvement':<30} {report['comparison']['accuracy_improvement']:<15} {report['comparison']['speed_ratio']}")
        print("="*60)
        
        # Save report
        report_file = self.experiment_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {report_file}")
        
        # Create visualization data
        self.create_visualization_data(rag_results, insight_results)
        
        return report
    
    def create_visualization_data(self, rag_results: Dict, insight_results: Dict):
        """Create data for visualization."""
        viz_data = {
            'methods': ['Standard RAG', 'InsightSpike'],
            'accuracy': [rag_results['accuracy'], insight_results['accuracy']],
            'avg_response_time': [rag_results['avg_response_time'], insight_results['avg_response_time']],
            'correct_answers': [rag_results['correct'], insight_results['correct']],
            'total_questions': rag_results['total']
        }
        
        viz_file = self.experiment_dir / "visualization_data.json"
        with open(viz_file, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        print(f"✓ Visualization data saved to {viz_file}")


def main():
    """Run the performance comparison."""
    print("=== EXPERIMENT 8: RAG PERFORMANCE COMPARISON ===")
    print(f"Start time: {datetime.now()}")
    
    tester = RAGPerformanceTester()
    
    try:
        # Initialize
        if not tester.initialize_for_testing():
            print("Failed to initialize!")
            return
        
        # Run comparison with 100 test questions
        report = tester.compare_methods(test_size=100)
        
        print("\n✅ Experiment complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()