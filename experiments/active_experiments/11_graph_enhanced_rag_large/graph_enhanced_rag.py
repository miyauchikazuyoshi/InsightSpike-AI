#!/usr/bin/env python3
"""
Experiment 11: Graph-Enhanced RAG with Fixed Implementation
==========================================================

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


class GraphEnhancedRAG:
    """Test graph-enhanced RAG capabilities."""
    
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
            'max_documents': 500,  # Limit for testing
            'test_checkpoints': [10, 50, 100, 200, 500],
            'test_sample_size': 50
        }
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'graph_stats': [],
            'performance_metrics': [],
            'comparison_results': {}
        }
    
    def initialize_clean_data(self):
        """Initialize clean data directory."""
        print("\n=== Initializing Clean Data ===")
        
        # Backup original data
        backup_dir = self.experiment_dir / "data_backup_original"
        if not backup_dir.exists():
            shutil.copytree(self.data_dir, backup_dir)
            print(f"✓ Original data backed up to {backup_dir}")
        
        # Clear data files
        files_to_clear = ['episodes.json', 'index.faiss', 'graph_pyg.pt', 
                         'scalable_index.faiss', 'insight_facts.db', 'unknown_learning.db']
        
        for filename in files_to_clear:
            filepath = self.data_dir / filename
            if filepath.exists():
                filepath.unlink()
                print(f"  Removed: {filename}")
        
        print("✓ Data folder initialized")
    
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
        self.agent.initialize()
        print("✓ Agent initialized")
        
        checkpoints = self.experiment_config['test_checkpoints']
        checkpoint_results = []
        
        for checkpoint in checkpoints:
            if checkpoint > len(qa_pairs):
                continue
            
            print(f"\n📈 Building to {checkpoint} documents...")
            start_time = time.time()
            
            # Add documents up to checkpoint
            current_episodes = len(self.agent.l2_memory.episodes)
            
            for i in range(current_episodes // 2, checkpoint):
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
                
                # Progress indicator
                if (i + 1) % 25 == 0:
                    graph_nodes = result2.get('graph_nodes', 0)
                    episodes = result2.get('total_episodes', 0)
                    print(f"  Progress: {i+1} Q&A → {episodes} episodes, {graph_nodes} graph nodes")
            
            build_time = time.time() - start_time
            
            # Check graph state
            graph_stats = self._analyze_graph_state()
            
            # Test performance
            test_accuracy = self._test_retrieval_accuracy(qa_pairs[:checkpoint])
            
            # Test with graph enhancement
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
            print(f"     Graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
            print(f"     Basic accuracy: {test_accuracy:.1%}")
            print(f"     Graph-enhanced: {graph_enhanced_accuracy:.1%}")
            print(f"     Improvement: +{checkpoint_result['improvement']:.1%}")
            print(f"     Build time: {build_time:.1f}s")
        
        # Save state
        print("\n💾 Saving final state...")
        self.agent.save_state()
        
        self.results['graph_building'] = checkpoint_results
        return checkpoint_results
    
    def _analyze_graph_state(self) -> Dict[str, Any]:
        """Analyze current graph structure."""
        graph_path = self.data_dir / "graph_pyg.pt"
        
        if graph_path.exists():
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
            
            # Check expanded results
            all_results = initial_results + expanded_results
            for result in all_results:
                if qa['answer'].lower() in result['text'].lower():
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
            
            # Get episode indices from initial results
            initial_indices = []
            for result in initial_results[:2]:  # Top 2 results
                if 'metadata' in result and 'episode_idx' in result['metadata']:
                    initial_indices.append(result['metadata']['episode_idx'])
            
            # Find connected nodes
            connected_indices = set()
            for idx in initial_indices:
                # Find edges where this node is source
                mask = edge_index[0] == idx
                neighbors = edge_index[1][mask].tolist()
                connected_indices.update(neighbors)
            
            # Remove initial indices
            connected_indices -= set(initial_indices)
            
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
            print(f"Graph expansion error: {e}")
        
        return expanded
    
    def compare_with_standard_rag(self, qa_pairs: List[Dict]):
        """Compare with standard RAG approach."""
        print("\n=== Comparing with Standard RAG ===")
        
        if not self.embedder:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test standard RAG
        print("\n📊 Testing Standard RAG...")
        standard_correct = 0
        test_size = min(100, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), test_size)
        
        for idx in test_indices:
            qa = qa_pairs[idx]
            query_vec = self.embedder.encode(qa['question'])
            
            if hasattr(self.agent.l2_memory, 'index') and self.agent.l2_memory.index:
                D, I = self.agent.l2_memory.index.search(
                    query_vec.reshape(1, -1).astype(np.float32),
                    k=5
                )
                
                for ep_idx in I[0]:
                    if 0 <= ep_idx < len(self.agent.l2_memory.episodes):
                        ep = self.agent.l2_memory.episodes[ep_idx]
                        if qa['answer'].lower() in ep.text.lower():
                            standard_correct += 1
                            break
        
        standard_accuracy = standard_correct / test_size
        
        # Get InsightSpike results
        insightspike_basic = self.results['graph_building'][-1]['basic_accuracy'] if self.results['graph_building'] else 0
        insightspike_enhanced = self.results['graph_building'][-1]['graph_enhanced_accuracy'] if self.results['graph_building'] else 0
        
        comparison = {
            'standard_rag': standard_accuracy,
            'insightspike_basic': insightspike_basic,
            'insightspike_graph_enhanced': insightspike_enhanced,
            'graph_improvement': insightspike_enhanced - insightspike_basic
        }
        
        print(f"\n📊 Comparison Results:")
        print(f"   Standard RAG: {standard_accuracy:.1%}")
        print(f"   InsightSpike (basic): {insightspike_basic:.1%}")
        print(f"   InsightSpike (graph): {insightspike_enhanced:.1%}")
        print(f"   Graph improvement: +{comparison['graph_improvement']:.1%}")
        
        self.results['comparison'] = comparison
        return comparison
    
    def save_results(self):
        """Save experiment results."""
        print("\n=== Saving Results ===")
        
        # Add final graph analysis
        self.results['final_graph_state'] = self._analyze_graph_state()
        self.results['end_time'] = datetime.now().isoformat()
        
        # Save results
        results_file = self.results_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {results_file}")
        
        # Save visualization data
        if self.results['graph_building']:
            viz_data = {
                'checkpoints': [r['checkpoint'] for r in self.results['graph_building']],
                'graph_nodes': [r['graph_nodes'] for r in self.results['graph_building']],
                'graph_edges': [r['graph_edges'] for r in self.results['graph_building']],
                'basic_accuracy': [r['basic_accuracy'] for r in self.results['graph_building']],
                'enhanced_accuracy': [r['graph_enhanced_accuracy'] for r in self.results['graph_building']],
                'improvement': [r['improvement'] for r in self.results['graph_building']]
            }
            
            viz_file = self.results_dir / "graph_visualization_data.json"
            with open(viz_file, 'w') as f:
                json.dump(viz_data, f, indent=2)
            print(f"✓ Visualization data saved to {viz_file}")


def main():
    """Run Experiment 11."""
    print("=== EXPERIMENT 11: GRAPH-ENHANCED RAG ===")
    print(f"Start time: {datetime.now()}")
    
    experiment = GraphEnhancedRAG()
    
    try:
        # Initialize
        experiment.initialize_clean_data()
        
        # Load dataset
        qa_pairs = experiment.load_test_dataset()
        
        if not qa_pairs:
            print("\n❌ No data loaded!")
            return
        
        # Test graph building and performance
        graph_results = experiment.test_graph_building(qa_pairs)
        
        # Compare with standard RAG
        comparison = experiment.compare_with_standard_rag(qa_pairs)
        
        # Save results
        experiment.save_results()
        
        print("\n✅ Experiment 11 complete!")
        
        # Summary
        if graph_results:
            final = graph_results[-1]
            print(f"\n📊 Final Results:")
            print(f"   Episodes: {final['episodes']}")
            print(f"   Graph: {final['graph_nodes']} nodes, {final['graph_edges']} edges")
            print(f"   Graph enhancement: +{final['improvement']:.1%} accuracy")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()