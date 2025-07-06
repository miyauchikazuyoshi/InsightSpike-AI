#!/usr/bin/env python3
"""
Experiment 11 Clean: Graph-Enhanced RAG with Fixed Implementation
================================================================

Tests InsightSpike with properly functioning graph building,
comparing performance with and without graph enhancement.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config
from datasets import Dataset
from sentence_transformers import SentenceTransformer


class GraphEnhancedRAGClean:
    """Test graph-enhanced RAG capabilities with clean initialization."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        self.embedder = None
        
        # Experiment paths
        self.experiment_dir = Path(__file__).parent
        self.data_dir = Path(self.config.paths.data_dir)
        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Experiment configuration
        self.experiment_config = {
            'max_documents': 200,  # Smaller for testing
            'test_checkpoints': [10, 25, 50, 100, 200],
            'test_sample_size': 30
        }
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'graph_stats': [],
            'performance_metrics': [],
            'comparison_results': {}
        }
    
    def load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load SQuAD dataset for testing."""
        print("\n=== Loading Test Dataset ===")
        
        qa_pairs = []
        dataset_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets/squad_300")
        
        if dataset_path.exists():
            dataset = Dataset.load_from_disk(str(dataset_path))
            
            for i, item in enumerate(dataset):
                if i >= self.experiment_config['max_documents']:
                    break
                
                context = item.get('context', '')
                question = item.get('question', '')
                answers = item.get('answers', {})
                
                if isinstance(answers, dict) and 'text' in answers:
                    answer = answers['text'][0] if answers['text'] else ""
                else:
                    answer = str(answers)
                
                if context and question and answer:
                    qa_pairs.append({
                        'context': context,
                        'question': question,
                        'answer': answer,
                        'id': f"squad_{i}"
                    })
            
            print(f"✓ Loaded {len(qa_pairs)} Q&A pairs")
        else:
            print("❌ Dataset not found!")
        
        return qa_pairs
    
    def test_graph_building(self, qa_pairs: List[Dict[str, Any]]):
        """Test progressive graph building with checkpoints."""
        print("\n=== Testing Graph-Enhanced RAG ===")
        
        # Initialize agent
        self.agent = MainAgent(self.config)
        success = self.agent.initialize()
        print(f"✓ Agent initialized: {success}")
        
        # Initial state check
        print(f"  Initial episodes: {len(self.agent.l2_memory.episodes)}")
        print(f"  Initial graph: {self._analyze_graph_state()}")
        
        checkpoints = self.experiment_config['test_checkpoints']
        checkpoint_results = []
        
        for checkpoint in checkpoints:
            if checkpoint > len(qa_pairs):
                continue
            
            print(f"\n📈 Building to {checkpoint} documents...")
            start_time = time.time()
            
            # Calculate how many to add
            current_qa_count = len(self.agent.l2_memory.episodes) // 2
            target_qa_count = checkpoint
            
            print(f"  Current: {current_qa_count} Q&A pairs, Target: {target_qa_count}")
            
            # Add documents up to checkpoint
            added_count = 0
            for i in range(current_qa_count, target_qa_count):
                if i >= len(qa_pairs):
                    break
                
                qa = qa_pairs[i]
                
                # Add context
                result1 = self.agent.add_episode_with_graph_update(
                    qa['context'], 
                    c_value=0.8
                )
                
                # Add Q&A
                qa_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                result2 = self.agent.add_episode_with_graph_update(
                    qa_text,
                    c_value=0.6
                )
                
                added_count += 1
                
                # Progress indicator
                if (added_count % 10 == 0) or (i == target_qa_count - 1):
                    graph_nodes = result2.get('graph_nodes', 0)
                    episodes = result2.get('total_episodes', 0)
                    print(f"    Added {added_count} Q&A → {episodes} episodes, {graph_nodes} graph nodes")
            
            build_time = time.time() - start_time
            
            # Check graph state
            graph_stats = self._analyze_graph_state()
            
            # Test performance
            print(f"  Testing retrieval accuracy...")
            test_accuracy = self._test_retrieval_accuracy(qa_pairs[:checkpoint])
            
            # Test with graph enhancement
            print(f"  Testing graph-enhanced retrieval...")
            graph_enhanced_accuracy = self._test_graph_enhanced_retrieval(qa_pairs[:checkpoint])
            
            checkpoint_result = {
                'checkpoint': checkpoint,
                'episodes': len(self.agent.l2_memory.episodes),
                'graph_nodes': graph_stats['nodes'],
                'graph_edges': graph_stats['edges'],
                'graph_density': graph_stats['density'],
                'build_time': build_time,
                'basic_accuracy': test_accuracy,
                'graph_enhanced_accuracy': graph_enhanced_accuracy,
                'improvement': graph_enhanced_accuracy - test_accuracy
            }
            
            checkpoint_results.append(checkpoint_result)
            
            print(f"\n  📊 Checkpoint {checkpoint} Results:")
            print(f"     Episodes: {checkpoint_result['episodes']}")
            print(f"     Graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges (density: {graph_stats['density']:.3f})")
            print(f"     Basic accuracy: {test_accuracy:.1%}")
            print(f"     Graph-enhanced: {graph_enhanced_accuracy:.1%}")
            print(f"     Improvement: {'+' if checkpoint_result['improvement'] >= 0 else ''}{checkpoint_result['improvement']:.1%}")
            print(f"     Build time: {build_time:.1f}s")
        
        # Save state
        print("\n💾 Saving final state...")
        if self.agent.save_state():
            print("✓ State saved successfully")
        
        self.results['graph_building'] = checkpoint_results
        return checkpoint_results
    
    def _analyze_graph_state(self) -> Dict[str, Any]:
        """Analyze current graph structure."""
        graph_path = self.data_dir / "graph_pyg.pt"
        
        if graph_path.exists():
            try:
                data = torch.load(graph_path)
                nodes = data.x.shape[0] if data.x is not None else 0
                edges = data.edge_index.shape[1] if data.edge_index is not None and data.edge_index.numel() > 0 else 0
                density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'density': density,
                    'file_size': graph_path.stat().st_size
                }
            except Exception as e:
                print(f"Error loading graph: {e}")
        
        return {'nodes': 0, 'edges': 0, 'density': 0, 'file_size': 0}
    
    def _test_retrieval_accuracy(self, qa_pairs: List[Dict], sample_size: int = None) -> float:
        """Test basic retrieval accuracy without graph enhancement."""
        if not qa_pairs:
            return 0.0
        
        sample_size = sample_size or self.experiment_config['test_sample_size']
        sample_size = min(sample_size, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), sample_size)
        
        correct = 0
        for idx in test_indices:
            qa = qa_pairs[idx]
            
            # Basic search
            results = self.agent.l2_memory.search_episodes(
                qa['question'],
                k=5
            )
            
            # Check if answer is found
            for result in results:
                if qa['answer'].lower() in result['text'].lower():
                    correct += 1
                    break
        
        return correct / sample_size
    
    def _test_graph_enhanced_retrieval(self, qa_pairs: List[Dict], sample_size: int = None) -> float:
        """Test retrieval with graph enhancement."""
        if not qa_pairs:
            return 0.0
        
        sample_size = sample_size or self.experiment_config['test_sample_size']
        sample_size = min(sample_size, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), sample_size)
        
        correct = 0
        for idx in test_indices:
            qa = qa_pairs[idx]
            
            # Get initial results
            initial_results = self.agent.l2_memory.search_episodes(
                qa['question'],
                k=3
            )
            
            # Expand search using graph connections
            expanded_results = self._expand_with_graph(initial_results)
            
            # Check all results
            all_results = initial_results + expanded_results
            seen_texts = set()
            
            for result in all_results:
                text = result['text']
                if text not in seen_texts:
                    seen_texts.add(text)
                    if qa['answer'].lower() in text.lower():
                        correct += 1
                        break
        
        return correct / sample_size
    
    def _expand_with_graph(self, initial_results: List[Dict]) -> List[Dict]:
        """Expand search results using graph connections."""
        expanded = []
        
        if not self.agent.l3_graph or not initial_results:
            return expanded
        
        # Load current graph
        graph_path = self.data_dir / "graph_pyg.pt"
        if not graph_path.exists():
            return expanded
        
        try:
            graph_data = torch.load(graph_path)
            edge_index = graph_data.edge_index
            
            if edge_index.numel() == 0:
                return expanded
            
            # Get episode indices from initial results
            initial_indices = set()
            for result in initial_results[:2]:  # Top 2 results
                # Try to find episode index
                for i, ep in enumerate(self.agent.l2_memory.episodes):
                    if ep.text == result['text']:
                        initial_indices.add(i)
                        break
            
            # Find connected nodes
            connected_indices = set()
            for idx in initial_indices:
                # Find edges where this node is source
                mask = edge_index[0] == idx
                if mask.any():
                    neighbors = edge_index[1][mask].tolist()
                    connected_indices.update(neighbors)
            
            # Remove initial indices
            connected_indices -= initial_indices
            
            # Get connected episodes
            for idx in list(connected_indices)[:3]:  # Limit expansion
                if 0 <= idx < len(self.agent.l2_memory.episodes):
                    ep = self.agent.l2_memory.episodes[idx]
                    expanded.append({
                        'text': ep.text,
                        'similarity': 0.7,  # Lower score for expanded results
                        'metadata': {'episode_idx': idx, 'source': 'graph_expansion'}
                    })
        
        except Exception as e:
            # Silent fail for graph expansion
            pass
        
        return expanded
    
    def save_results(self):
        """Save experiment results."""
        print("\n=== Saving Results ===")
        
        # Add final graph analysis
        self.results['final_graph_state'] = self._analyze_graph_state()
        self.results['end_time'] = datetime.now().isoformat()
        
        # Save results
        results_file = self.results_dir / f"experiment_results_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {results_file}")
        
        # Save visualization data
        if self.results.get('graph_building'):
            viz_data = {
                'checkpoints': [r['checkpoint'] for r in self.results['graph_building']],
                'graph_nodes': [r['graph_nodes'] for r in self.results['graph_building']],
                'graph_edges': [r['graph_edges'] for r in self.results['graph_building']],
                'basic_accuracy': [r['basic_accuracy'] for r in self.results['graph_building']],
                'enhanced_accuracy': [r['graph_enhanced_accuracy'] for r in self.results['graph_building']],
                'improvement': [r['improvement'] for r in self.results['graph_building']]
            }
            
            viz_file = self.results_dir / "graph_visualization_data_clean.json"
            with open(viz_file, 'w') as f:
                json.dump(viz_data, f, indent=2)
            print(f"✓ Visualization data saved to {viz_file}")


def main():
    """Run Experiment 11 with clean data."""
    print("=== EXPERIMENT 11 CLEAN: GRAPH-ENHANCED RAG ===")
    print(f"Start time: {datetime.now()}")
    
    experiment = GraphEnhancedRAGClean()
    
    try:
        # Load dataset
        qa_pairs = experiment.load_test_dataset()
        
        if not qa_pairs:
            print("\n❌ No data loaded!")
            return
        
        # Test graph building and performance
        graph_results = experiment.test_graph_building(qa_pairs)
        
        # Save results
        experiment.save_results()
        
        print("\n✅ Experiment 11 Clean complete!")
        
        # Summary
        if graph_results:
            print("\n📊 Summary:")
            print(f"{'Checkpoint':<12} {'Episodes':<10} {'Graph Nodes':<12} {'Graph Edges':<12} {'Improvement':<12}")
            print("-" * 60)
            for r in graph_results:
                print(f"{r['checkpoint']:<12} {r['episodes']:<10} {r['graph_nodes']:<12} {r['graph_edges']:<12} {r['improvement']:+.1%}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()