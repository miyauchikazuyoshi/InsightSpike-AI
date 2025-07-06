#!/usr/bin/env python3
"""
Experiment 11 Large Scale: Graph-Enhanced RAG with Maximum Data
==============================================================

Tests InsightSpike with properly functioning graph building on large dataset,
comparing performance with and without graph enhancement.
No timeouts - let it run as long as needed!
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


class GraphEnhancedRAGLarge:
    """Test graph-enhanced RAG capabilities on large scale data."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        self.embedder = None
        
        # Experiment paths
        self.experiment_dir = Path(__file__).parent
        self.data_dir = Path(self.config.paths.data_dir)
        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Experiment configuration - GO BIG!
        self.experiment_config = {
            'max_documents': 1000,  # Use 1000 documents
            'test_checkpoints': [10, 50, 100, 250, 500, 750, 1000],
            'test_sample_size': 100  # Larger test sample
        }
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'graph_stats': [],
            'performance_metrics': [],
            'comparison_results': {}
        }
    
    def load_all_datasets(self) -> List[Dict[str, Any]]:
        """Load datasets from all available sources."""
        print("\n=== Loading All Available Datasets ===")
        
        all_qa_pairs = []
        
        # Try to load from multiple sources
        dataset_paths = [
            "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets/squad_300",
            "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets/squad_200",
            "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/large_huggingface_datasets/squad_100",
            "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets/hotpot_qa_60",
            "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/huggingface_datasets/squad_30"
        ]
        
        for dataset_path in dataset_paths:
            path = Path(dataset_path)
            if not path.exists():
                print(f"  Skipping {path.name} (not found)")
                continue
                
            try:
                dataset = Dataset.load_from_disk(str(path))
                count = 0
                
                for item in dataset:
                    if len(all_qa_pairs) >= self.experiment_config['max_documents']:
                        break
                        
                    context = item.get('context', '')
                    question = item.get('question', '')
                    answers = item.get('answers', {})
                    
                    if isinstance(answers, dict) and 'text' in answers:
                        answer = answers['text'][0] if answers['text'] else ""
                    else:
                        answer = str(answers)
                    
                    if context and question and answer:
                        all_qa_pairs.append({
                            'context': context,
                            'question': question,
                            'answer': answer,
                            'dataset': path.name,
                            'id': f"{path.name}_{count}"
                        })
                        count += 1
                
                print(f"  ✓ {path.name}: {count} Q&A pairs")
                
            except Exception as e:
                print(f"  ✗ Error loading {path.name}: {e}")
        
        print(f"\n📊 Total Q&A pairs loaded: {len(all_qa_pairs)}")
        return all_qa_pairs
    
    def test_graph_building(self, qa_pairs: List[Dict[str, Any]]):
        """Test progressive graph building with checkpoints."""
        print("\n=== Testing Graph-Enhanced RAG at Scale ===")
        
        # Initialize agent
        self.agent = MainAgent(self.config)
        success = self.agent.initialize()
        print(f"✓ Agent initialized: {success}")
        
        checkpoints = self.experiment_config['test_checkpoints']
        checkpoint_results = []
        
        for checkpoint in checkpoints:
            if checkpoint > len(qa_pairs):
                continue
            
            print(f"\n{'='*60}")
            print(f"📈 Building to {checkpoint} documents...")
            start_time = time.time()
            
            # Calculate how many to add
            current_qa_count = len(self.agent.l2_memory.episodes) // 2
            target_qa_count = checkpoint
            
            print(f"  Current: {current_qa_count} Q&A pairs")
            print(f"  Target: {target_qa_count} Q&A pairs")
            print(f"  To add: {target_qa_count - current_qa_count} Q&A pairs")
            
            # Add documents up to checkpoint
            added_count = 0
            batch_start = time.time()
            
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
                
                # Progress indicator every 25 documents
                if added_count % 25 == 0:
                    batch_time = time.time() - batch_start
                    graph_nodes = result2.get('graph_nodes', 0)
                    episodes = result2.get('total_episodes', 0)
                    print(f"    Progress: {added_count}/{target_qa_count - current_qa_count} " +
                          f"→ {episodes} episodes, {graph_nodes} graph nodes " +
                          f"({batch_time:.1f}s)")
                    batch_start = time.time()
            
            build_time = time.time() - start_time
            
            # Check graph state
            print(f"\n  Analyzing graph state...")
            graph_stats = self._analyze_graph_state()
            
            # Test performance
            print(f"  Testing retrieval accuracy on {self.experiment_config['test_sample_size']} samples...")
            test_accuracy = self._test_retrieval_accuracy(qa_pairs[:checkpoint])
            
            print(f"  Testing graph-enhanced retrieval...")
            graph_enhanced_accuracy = self._test_graph_enhanced_retrieval(qa_pairs[:checkpoint])
            
            # Memory stats
            memory_stats = self.agent.l2_memory.get_memory_stats()
            
            checkpoint_result = {
                'checkpoint': checkpoint,
                'episodes': len(self.agent.l2_memory.episodes),
                'integration_rate': 1 - (len(self.agent.l2_memory.episodes) / (checkpoint * 2)),
                'graph_nodes': graph_stats['nodes'],
                'graph_edges': graph_stats['edges'],
                'graph_density': graph_stats['density'],
                'graph_file_size': graph_stats['file_size'],
                'build_time': build_time,
                'basic_accuracy': test_accuracy,
                'graph_enhanced_accuracy': graph_enhanced_accuracy,
                'improvement': graph_enhanced_accuracy - test_accuracy,
                'memory_stats': memory_stats
            }
            
            checkpoint_results.append(checkpoint_result)
            
            print(f"\n  📊 Checkpoint {checkpoint} Results:")
            print(f"     Episodes: {checkpoint_result['episodes']} (Integration rate: {checkpoint_result['integration_rate']:.1%})")
            print(f"     Graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
            print(f"     Graph density: {graph_stats['density']:.3f}")
            print(f"     Graph file size: {graph_stats['file_size']/1024:.1f} KB")
            print(f"     Basic accuracy: {test_accuracy:.1%}")
            print(f"     Graph-enhanced: {graph_enhanced_accuracy:.1%}")
            print(f"     Improvement: {'+' if checkpoint_result['improvement'] >= 0 else ''}{checkpoint_result['improvement']:.1%}")
            print(f"     Build time: {build_time:.1f}s ({build_time/60:.1f} minutes)")
        
        # Save state
        print("\n💾 Saving final state...")
        if self.agent.save_state():
            print("✓ State saved successfully")
            
        # Save Q&A pairs for reference
        qa_file = self.data_dir / "qa_pairs_large.json"
        with open(qa_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"✓ Q&A pairs saved to {qa_file}")
        
        self.results['graph_building'] = checkpoint_results
        return checkpoint_results
    
    def _analyze_graph_state(self) -> Dict[str, Any]:
        """Analyze current graph structure in detail."""
        graph_path = self.data_dir / "graph_pyg.pt"
        
        if graph_path.exists():
            try:
                data = torch.load(graph_path)
                nodes = data.x.shape[0] if data.x is not None else 0
                edges = data.edge_index.shape[1] if data.edge_index is not None and data.edge_index.numel() > 0 else 0
                density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
                
                # Analyze degree distribution
                if edges > 0:
                    edge_index = data.edge_index
                    degrees = torch.zeros(nodes)
                    for i in range(edges):
                        src = edge_index[0, i].item()
                        degrees[src] += 1
                    
                    avg_degree = degrees.mean().item()
                    max_degree = degrees.max().item()
                else:
                    avg_degree = max_degree = 0
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'density': density,
                    'file_size': graph_path.stat().st_size,
                    'avg_degree': avg_degree,
                    'max_degree': max_degree
                }
            except Exception as e:
                print(f"Error analyzing graph: {e}")
        
        return {'nodes': 0, 'edges': 0, 'density': 0, 'file_size': 0, 'avg_degree': 0, 'max_degree': 0}
    
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
        graph_expansions = 0
        
        for idx in test_indices:
            qa = qa_pairs[idx]
            
            # Get initial results
            initial_results = self.agent.l2_memory.search_episodes(
                qa['question'],
                k=3
            )
            
            # Expand search using graph connections
            expanded_results = self._expand_with_graph(initial_results)
            if expanded_results:
                graph_expansions += 1
            
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
        
        print(f"    Graph expansions used: {graph_expansions}/{sample_size}")
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
                # Try to find episode index by matching text
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
            
            # Get connected episodes (limit to 5 for efficiency)
            for idx in list(connected_indices)[:5]:
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
    
    def compare_with_standard_rag(self, qa_pairs: List[Dict]):
        """Compare with standard RAG approach."""
        print("\n=== Comparing with Standard RAG ===")
        
        if not self.embedder:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test standard RAG
        print("\n📊 Testing Standard RAG on 200 samples...")
        standard_correct = 0
        test_size = min(200, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), test_size)
        
        for i, idx in enumerate(test_indices):
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
            
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{test_size}")
        
        standard_accuracy = standard_correct / test_size
        
        # Get InsightSpike results from last checkpoint
        insightspike_basic = self.results['graph_building'][-1]['basic_accuracy'] if self.results['graph_building'] else 0
        insightspike_enhanced = self.results['graph_building'][-1]['graph_enhanced_accuracy'] if self.results['graph_building'] else 0
        
        comparison = {
            'standard_rag': standard_accuracy,
            'insightspike_basic': insightspike_basic,
            'insightspike_graph_enhanced': insightspike_enhanced,
            'graph_improvement': insightspike_enhanced - insightspike_basic,
            'vs_standard': insightspike_enhanced - standard_accuracy
        }
        
        print(f"\n📊 Final Comparison Results:")
        print(f"   Standard RAG: {standard_accuracy:.1%}")
        print(f"   InsightSpike (basic): {insightspike_basic:.1%}")
        print(f"   InsightSpike (graph): {insightspike_enhanced:.1%}")
        print(f"   Graph improvement: {'+' if comparison['graph_improvement'] >= 0 else ''}{comparison['graph_improvement']:.1%}")
        print(f"   vs Standard RAG: {'+' if comparison['vs_standard'] >= 0 else ''}{comparison['vs_standard']:.1%}")
        
        self.results['comparison'] = comparison
        return comparison
    
    def save_results(self):
        """Save comprehensive experiment results."""
        print("\n=== Saving Results ===")
        
        # Add final graph analysis
        self.results['final_graph_state'] = self._analyze_graph_state()
        self.results['end_time'] = datetime.now().isoformat()
        
        # Calculate total time
        start = datetime.fromisoformat(self.results['start_time'])
        end = datetime.fromisoformat(self.results['end_time'])
        total_minutes = (end - start).total_seconds() / 60
        self.results['total_time_minutes'] = total_minutes
        
        # Save results
        results_file = self.results_dir / f"experiment_results_large_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {results_file}")
        
        # Save visualization data
        if self.results.get('graph_building'):
            viz_data = {
                'checkpoints': [r['checkpoint'] for r in self.results['graph_building']],
                'episodes': [r['episodes'] for r in self.results['graph_building']],
                'integration_rates': [r['integration_rate'] for r in self.results['graph_building']],
                'graph_nodes': [r['graph_nodes'] for r in self.results['graph_building']],
                'graph_edges': [r['graph_edges'] for r in self.results['graph_building']],
                'graph_density': [r['graph_density'] for r in self.results['graph_building']],
                'basic_accuracy': [r['basic_accuracy'] for r in self.results['graph_building']],
                'enhanced_accuracy': [r['graph_enhanced_accuracy'] for r in self.results['graph_building']],
                'improvement': [r['improvement'] for r in self.results['graph_building']],
                'build_times': [r['build_time'] for r in self.results['graph_building']]
            }
            
            viz_file = self.results_dir / "graph_visualization_data_large.json"
            with open(viz_file, 'w') as f:
                json.dump(viz_data, f, indent=2)
            print(f"✓ Visualization data saved to {viz_file}")
        
        # Save detailed summary
        summary_file = self.results_dir / "experiment_summary_large.txt"
        with open(summary_file, 'w') as f:
            f.write("EXPERIMENT 11 LARGE SCALE RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total runtime: {total_minutes:.1f} minutes\n")
            f.write(f"Final graph: {self.results['final_graph_state']['nodes']} nodes, ")
            f.write(f"{self.results['final_graph_state']['edges']} edges\n")
            f.write(f"Graph file size: {self.results['final_graph_state']['file_size']/1024:.1f} KB\n\n")
            
            if self.results.get('comparison'):
                f.write("Performance Comparison:\n")
                f.write(f"  Standard RAG: {self.results['comparison']['standard_rag']:.1%}\n")
                f.write(f"  InsightSpike Basic: {self.results['comparison']['insightspike_basic']:.1%}\n")
                f.write(f"  InsightSpike Graph: {self.results['comparison']['insightspike_graph_enhanced']:.1%}\n")
                f.write(f"  Graph Improvement: {self.results['comparison']['graph_improvement']:.1%}\n")
        
        print(f"✓ Summary saved to {summary_file}")


def main():
    """Run large scale Experiment 11."""
    print("=== EXPERIMENT 11 LARGE SCALE: GRAPH-ENHANCED RAG ===")
    print(f"Start time: {datetime.now()}")
    print("NO TIMEOUTS - This will run until completion!")
    
    experiment = GraphEnhancedRAGLarge()
    
    try:
        # Load all available datasets
        qa_pairs = experiment.load_all_datasets()
        
        if not qa_pairs:
            print("\n❌ No data loaded!")
            return
        
        # Test graph building and performance at scale
        graph_results = experiment.test_graph_building(qa_pairs)
        
        # Compare with standard RAG
        comparison = experiment.compare_with_standard_rag(qa_pairs)
        
        # Save all results
        experiment.save_results()
        
        print("\n✅ Experiment 11 Large Scale complete!")
        
        # Final summary
        if graph_results:
            print("\n" + "="*60)
            print("📊 FINAL SUMMARY")
            print("="*60)
            
            final = graph_results[-1]
            print(f"Scale: {final['checkpoint']} documents processed")
            print(f"Episodes: {final['episodes']} (Integration: {final['integration_rate']:.1%})")
            print(f"Graph: {final['graph_nodes']} nodes, {final['graph_edges']} edges")
            print(f"Graph density: {final['graph_density']:.3f}")
            print(f"Graph file size: {final['graph_file_size']/1024:.1f} KB")
            print(f"\nAccuracy:")
            print(f"  Basic retrieval: {final['basic_accuracy']:.1%}")
            print(f"  Graph-enhanced: {final['graph_enhanced_accuracy']:.1%}")
            print(f"  Improvement: {'+' if final['improvement'] >= 0 else ''}{final['improvement']:.1%}")
            
            if comparison:
                print(f"\nvs Standard RAG: {'+' if comparison['vs_standard'] >= 0 else ''}{comparison['vs_standard']:.1%}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()