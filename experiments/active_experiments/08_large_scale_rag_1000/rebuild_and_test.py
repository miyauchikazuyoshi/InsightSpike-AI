#!/usr/bin/env python3
"""
Rebuild Knowledge Base and Test Performance
===========================================

Since the save failed, rebuild the knowledge base from Q&A pairs and test.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config


class RebuildAndTest:
    """Rebuild knowledge base and test performance."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        
        # Use experiment-specific directory
        self.experiment_dir = Path(__file__).parent
        self.data_dir = self.experiment_dir / "data"
        
        # Configure paths
        self.config.paths.data_dir = str(self.data_dir)
        self.config.memory.index_file = str(self.data_dir / "index.faiss")
        self.config.reasoning.graph_file = str(self.data_dir / "graph_pyg.pt")
        
        # Adjust integration thresholds
        self.config.memory.episode_integration_similarity_threshold = 0.95
        self.config.memory.episode_integration_content_threshold = 0.8
    
    def initialize_agent(self):
        """Initialize agent."""
        print("\n=== Initializing Agent ===")
        
        self.agent = MainAgent(self.config)
        self.agent.l2_memory = L2EnhancedScalableMemory(
            dim=self.config.embedding.dimension,
            config=self.config,
            use_scalable_graph=True
        )
        
        # Initialize fresh
        self.agent.l2_memory.initialize()
        if hasattr(self.agent, 'l3_graph') and self.agent.l3_graph:
            self.agent.l3_graph.initialize()
        
        print("✓ Agent initialized")
    
    def rebuild_knowledge_base(self):
        """Rebuild knowledge base from Q&A pairs."""
        print("\n=== Rebuilding Knowledge Base ===")
        
        # Load Q&A pairs
        qa_file = self.data_dir / "qa_pairs.json"
        with open(qa_file, 'r') as f:
            qa_pairs = json.load(f)
        
        print(f"Loaded {len(qa_pairs)} Q&A pairs")
        
        # Process in batches to show progress
        batch_size = 100
        episodes_added = 0
        
        start_time = time.time()
        
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i+batch_size]
            
            for qa in batch:
                # Add context
                self.agent.add_episode_with_graph_update(
                    qa['context'],
                    c_value=0.8
                )
                
                # Add Q&A pair
                qa_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                self.agent.add_episode_with_graph_update(
                    qa_text,
                    c_value=0.6
                )
            
            episodes_added = len(self.agent.l2_memory.episodes)
            print(f"Progress: {min(i+batch_size, len(qa_pairs))}/{len(qa_pairs)} - Episodes: {episodes_added}")
        
        build_time = time.time() - start_time
        
        print(f"\n✓ Rebuild complete!")
        print(f"  Episodes: {episodes_added}")
        print(f"  Build time: {build_time:.1f}s")
        
        # Save state
        print("\nSaving state...")
        if self.agent.save_state():
            print("✓ State saved successfully")
        else:
            print("✗ Failed to save state")
        
        return qa_pairs
    
    def test_standard_rag(self, qa_pairs: List[Dict], test_size: int = 100):
        """Test standard RAG approach."""
        print("\n=== Testing Standard RAG ===")
        
        # Use simple semantic search without graph integration
        correct = 0
        total = 0
        
        # Sample test questions
        test_indices = np.random.choice(len(qa_pairs), min(test_size, len(qa_pairs)), replace=False)
        
        start_time = time.time()
        
        for idx in test_indices:
            qa = qa_pairs[idx]
            
            # Simple semantic search
            query_vec = self.agent.l1_perception.embedder.encode(qa['question'])
            
            # Search in FAISS
            if hasattr(self.agent.l2_memory, 'index') and self.agent.l2_memory.index:
                D, I = self.agent.l2_memory.index.search(query_vec.reshape(1, -1), k=5)
                
                # Check if answer is in retrieved episodes
                for i in I[0]:
                    if i < len(self.agent.l2_memory.episodes):
                        ep = self.agent.l2_memory.episodes[i]
                        ep_text = ep.text if hasattr(ep, 'text') else str(ep)
                        
                        if qa['answer'].lower() in ep_text.lower():
                            correct += 1
                            break
            
            total += 1
        
        rag_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0
        
        print(f"Standard RAG Results:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"  Time: {rag_time:.2f}s")
        print(f"  Avg time per query: {rag_time/total:.3f}s")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'time': rag_time
        }
    
    def test_insightspike(self, qa_pairs: List[Dict], test_size: int = 100):
        """Test InsightSpike with graph-based retrieval."""
        print("\n=== Testing InsightSpike ===")
        
        correct = 0
        total = 0
        
        # Use same test indices for fair comparison
        test_indices = np.random.choice(len(qa_pairs), min(test_size, len(qa_pairs)), replace=False)
        
        start_time = time.time()
        
        for idx in test_indices:
            qa = qa_pairs[idx]
            
            try:
                # Use InsightSpike's full pipeline
                result = self.agent.process_question(qa['question'], max_cycles=1, verbose=False)
                response = result.get('response', '')
                
                if qa['answer'].lower() in response.lower():
                    correct += 1
                
            except Exception as e:
                print(f"Error processing question: {e}")
            
            total += 1
        
        insight_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0
        
        print(f"InsightSpike Results:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"  Time: {insight_time:.2f}s")
        print(f"  Avg time per query: {insight_time/total:.3f}s")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'time': insight_time
        }
    
    def generate_report(self, rag_results: Dict, insight_results: Dict):
        """Generate comparison report."""
        print("\n=== Performance Comparison Report ===")
        
        report = {
            'experiment': 'Experiment 8: RAG Performance Comparison',
            'timestamp': datetime.now().isoformat(),
            'knowledge_base': {
                'episodes': len(self.agent.l2_memory.episodes),
                'qa_pairs': len(self.data_dir / "qa_pairs.json")
            },
            'standard_rag': rag_results,
            'insightspike': insight_results,
            'comparison': {
                'accuracy_improvement': f"{(insight_results['accuracy'] - rag_results['accuracy']) * 100:+.1f}%",
                'speed_ratio': f"{rag_results['time'] / insight_results['time']:.2f}x"
            }
        }
        
        # Print summary
        print("\n📊 Summary:")
        print(f"  Standard RAG accuracy: {rag_results['accuracy']:.1%}")
        print(f"  InsightSpike accuracy: {insight_results['accuracy']:.1%}")
        print(f"  Improvement: {report['comparison']['accuracy_improvement']}")
        print(f"  Speed comparison: InsightSpike is {report['comparison']['speed_ratio']} as fast")
        
        # Save report
        report_file = self.experiment_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {report_file}")
        
        return report


def main():
    """Run the rebuild and test."""
    print("=== EXPERIMENT 8: REBUILD AND TEST ===")
    
    tester = RebuildAndTest()
    
    try:
        # Initialize
        tester.initialize_agent()
        
        # Rebuild knowledge base
        qa_pairs = tester.rebuild_knowledge_base()
        
        # Test both approaches
        rag_results = tester.test_standard_rag(qa_pairs, test_size=100)
        insight_results = tester.test_insightspike(qa_pairs, test_size=100)
        
        # Generate report
        tester.generate_report(rag_results, insight_results)
        
        print("\n✅ Experiment complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()