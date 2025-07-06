#!/usr/bin/env python3
"""
Experiment 10: Dynamic Growth RAG with All Datasets
===================================================

Tests InsightSpike's dynamic growth capabilities using all HuggingFace datasets,
then compares RAG accuracy with standard approaches.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
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


class DynamicGrowthRAG:
    """Test dynamic growth capabilities of InsightSpike RAG."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        self.embedder = None
        
        # Experiment paths
        self.experiment_dir = Path(__file__).parent
        self.data_dir = Path(self.config.paths.data_dir)
        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Dataset paths
        self.dataset_dirs = {
            'huggingface': Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/huggingface_datasets"),
            'large': Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/large_huggingface_datasets"),
            'mega': Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets")
        }
        
        # Track experiment progress
        self.experiment_log = {
            'start_time': datetime.now().isoformat(),
            'phases': [],
            'growth_metrics': [],
            'final_results': {}
        }
    
    def initialize_clean_data(self):
        """Initialize data folder for clean experiment."""
        print("\n=== Initializing Clean Data Folder ===")
        
        # Clear existing data files
        files_to_clear = ['episodes.json', 'index.faiss', 'graph_pyg.pt', 
                         'scalable_index.faiss', 'insight_facts.db', 'unknown_learning.db']
        
        for filename in files_to_clear:
            filepath = self.data_dir / filename
            if filepath.exists():
                filepath.unlink()
                print(f"  Removed: {filename}")
        
        print("✓ Data folder initialized")
    
    def load_all_datasets(self) -> List[Dict[str, Any]]:
        """Load Q&A pairs from all three dataset directories."""
        print("\n=== Loading All HuggingFace Datasets ===")
        
        all_qa_pairs = []
        dataset_stats = {}
        
        for dataset_type, dataset_dir in self.dataset_dirs.items():
            print(f"\n📁 Loading from {dataset_type} datasets:")
            type_count = 0
            
            if not dataset_dir.exists():
                print(f"  ⚠️  Directory not found: {dataset_dir}")
                continue
            
            # List all subdirectories
            for subdir in sorted(dataset_dir.iterdir()):
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    try:
                        # Load dataset
                        dataset = Dataset.load_from_disk(str(subdir))
                        dataset_name = subdir.name
                        
                        # Extract Q&A pairs
                        qa_count = 0
                        for item in dataset:
                            context = item.get('context', '')
                            question = item.get('question', '')
                            answers = item.get('answers', {})
                            
                            # Extract answer
                            if isinstance(answers, dict) and 'text' in answers:
                                answer = answers['text'][0] if answers['text'] else ""
                            else:
                                answer = str(answers)
                            
                            if context and question and answer:
                                all_qa_pairs.append({
                                    'context': context,
                                    'question': question,
                                    'answer': answer,
                                    'dataset': f"{dataset_type}/{dataset_name}",
                                    'id': f"{dataset_type}_{dataset_name}_{qa_count}"
                                })
                                qa_count += 1
                        
                        type_count += qa_count
                        print(f"  ✓ {dataset_name}: {qa_count} Q&A pairs")
                        
                    except Exception as e:
                        print(f"  ✗ Error loading {subdir.name}: {e}")
            
            dataset_stats[dataset_type] = type_count
            print(f"  Total from {dataset_type}: {type_count} Q&A pairs")
        
        print(f"\n📊 Total Q&A pairs loaded: {len(all_qa_pairs)}")
        
        self.experiment_log['dataset_stats'] = dataset_stats
        self.experiment_log['total_qa_pairs'] = len(all_qa_pairs)
        
        return all_qa_pairs
    
    def test_dynamic_growth(self, qa_pairs: List[Dict[str, Any]]):
        """Test dynamic growth by progressively adding documents."""
        print("\n=== Testing Dynamic Growth RAG ===")
        
        # Initialize agent
        self.agent = MainAgent(self.config)
        print("✓ Agent initialized")
        
        # Growth checkpoints
        checkpoints = [10, 50, 100, 250, 500, 1000, 2000, len(qa_pairs)]
        checkpoints = [cp for cp in checkpoints if cp <= len(qa_pairs)]
        
        growth_results = []
        
        for checkpoint in checkpoints:
            print(f"\n📈 Growing to {checkpoint} documents...")
            
            start_time = time.time()
            
            # Add documents up to checkpoint
            current_size = len(self.agent.l2_memory.episodes)
            documents_to_add = checkpoint - (current_size // 2)  # Approximate, as we add 2 per QA
            
            if documents_to_add > 0:
                # Add new documents
                for i in range(current_size // 2, min(checkpoint, len(qa_pairs))):
                    qa = qa_pairs[i]
                    
                    # Add context
                    self.agent.add_episode_with_graph_update(
                        qa['context'], 
                        c_value=0.8
                    )
                    
                    # Add Q&A
                    qa_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                    self.agent.add_episode_with_graph_update(
                        qa_text,
                        c_value=0.6
                    )
                    
                    # Progress indicator
                    if (i + 1) % 100 == 0:
                        episodes = len(self.agent.l2_memory.episodes)
                        print(f"  Progress: {i+1} Q&A pairs → {episodes} episodes")
            
            growth_time = time.time() - start_time
            
            # Get current stats
            memory_stats = self.agent.l2_memory.get_memory_stats()
            episodes_count = memory_stats.get('total_episodes', 0)
            
            # Test performance at this size
            test_accuracy = self._test_accuracy_sample(qa_pairs[:checkpoint], sample_size=50)
            
            growth_result = {
                'checkpoint': checkpoint,
                'episodes': episodes_count,
                'integration_rate': 1 - episodes_count / (checkpoint * 2),
                'growth_time': growth_time,
                'test_accuracy': test_accuracy,
                'memory_stats': memory_stats
            }
            
            growth_results.append(growth_result)
            
            print(f"  ✓ Episodes: {episodes_count}")
            print(f"  ✓ Integration rate: {growth_result['integration_rate']:.1%}")
            print(f"  ✓ Test accuracy: {test_accuracy:.1%}")
            print(f"  ✓ Growth time: {growth_time:.1f}s")
        
        # Save state
        print("\n💾 Saving final state...")
        if self.agent.save_state():
            print("✓ State saved successfully")
        else:
            print("✗ Failed to save state")
        
        # Save Q&A pairs
        qa_file = self.data_dir / "qa_pairs.json"
        with open(qa_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        self.experiment_log['phases'].append({
            'phase': 'dynamic_growth',
            'results': growth_results
        })
        
        return growth_results
    
    def _test_accuracy_sample(self, qa_pairs: List[Dict], sample_size: int = 50) -> float:
        """Test accuracy on a sample of questions."""
        if not qa_pairs:
            return 0.0
        
        sample_size = min(sample_size, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), sample_size)
        
        correct = 0
        for idx in test_indices:
            qa = qa_pairs[idx]
            
            # Search for answer
            results = self.agent.l2_memory.search_episodes(
                qa['question'],
                k=5
            )
            
            # Check if answer is in any result
            for result in results:
                if qa['answer'].lower() in result['text'].lower():
                    correct += 1
                    break
        
        return correct / sample_size
    
    def compare_rag_approaches(self, qa_pairs: List[Dict]):
        """Compare InsightSpike with standard RAG approaches."""
        print("\n=== Comparing RAG Approaches ===")
        
        # Initialize embedder for standard RAG
        if not self.embedder:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test parameters
        test_size = min(200, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), test_size)
        
        results = {}
        
        # 1. Standard Semantic Search RAG
        print("\n📊 Testing Standard RAG...")
        results['standard_rag'] = self._test_standard_rag(qa_pairs, test_indices)
        
        # 2. InsightSpike with Graph Enhancement
        print("\n📊 Testing InsightSpike...")
        results['insightspike'] = self._test_insightspike(qa_pairs, test_indices)
        
        # 3. InsightSpike with Dynamic Growth (already tested above)
        print("\n📊 Dynamic Growth Results:")
        if self.experiment_log['phases']:
            growth_results = self.experiment_log['phases'][0]['results']
            final_growth = growth_results[-1] if growth_results else {}
            results['dynamic_growth'] = {
                'final_episodes': final_growth.get('episodes', 0),
                'integration_rate': final_growth.get('integration_rate', 0),
                'accuracy': final_growth.get('test_accuracy', 0),
                'growth_stages': len(growth_results)
            }
        
        self.experiment_log['comparison_results'] = results
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _test_standard_rag(self, qa_pairs: List[Dict], test_indices: List[int]) -> Dict:
        """Test standard semantic search RAG."""
        correct = 0
        response_times = []
        
        for idx in test_indices:
            qa = qa_pairs[idx]
            start_time = time.time()
            
            # Encode question
            query_vec = self.embedder.encode(qa['question'])
            
            # Search in FAISS
            if hasattr(self.agent.l2_memory, 'index') and self.agent.l2_memory.index:
                D, I = self.agent.l2_memory.index.search(
                    query_vec.reshape(1, -1).astype(np.float32),
                    k=5
                )
                
                # Check retrieved episodes
                for ep_idx in I[0]:
                    if 0 <= ep_idx < len(self.agent.l2_memory.episodes):
                        ep = self.agent.l2_memory.episodes[ep_idx]
                        ep_text = ep.text if hasattr(ep, 'text') else str(ep)
                        
                        if qa['answer'].lower() in ep_text.lower():
                            correct += 1
                            break
            
            response_times.append(time.time() - start_time)
        
        return {
            'accuracy': correct / len(test_indices),
            'correct': correct,
            'total': len(test_indices),
            'avg_response_time': np.mean(response_times),
            'method': 'Standard Semantic Search'
        }
    
    def _test_insightspike(self, qa_pairs: List[Dict], test_indices: List[int]) -> Dict:
        """Test InsightSpike with full pipeline."""
        correct = 0
        response_times = []
        spike_count = 0
        
        # Use a simple response checker since LLM might have issues
        for idx in test_indices:
            qa = qa_pairs[idx]
            start_time = time.time()
            
            try:
                # Use direct search with graph context
                results = self.agent.l2_memory.search_episodes(
                    qa['question'],
                    k=5
                )
                
                # Check if answer is found
                found = False
                for result in results:
                    if qa['answer'].lower() in result['text'].lower():
                        correct += 1
                        found = True
                        break
                
                # Check for spike detection (if using full pipeline)
                if hasattr(self.agent, 'l3_graph') and self.agent.l3_graph:
                    # Simulate graph analysis
                    docs = [{'text': r['text'], 'embedding': None} for r in results[:3]]
                    graph_analysis = self.agent.l3_graph.analyze_documents(docs)
                    if graph_analysis.get('spike_detected', False):
                        spike_count += 1
                
            except Exception as e:
                pass  # Count as incorrect
            
            response_times.append(time.time() - start_time)
        
        return {
            'accuracy': correct / len(test_indices),
            'correct': correct,
            'total': len(test_indices),
            'avg_response_time': np.mean(response_times),
            'spike_detections': spike_count,
            'method': 'InsightSpike Graph-Enhanced'
        }
    
    def _print_comparison(self, results: Dict):
        """Print comparison results."""
        print("\n" + "="*70)
        print("📊 RAG COMPARISON RESULTS")
        print("="*70)
        
        # Standard RAG
        if 'standard_rag' in results:
            r = results['standard_rag']
            print(f"\n🔹 Standard RAG:")
            print(f"   Accuracy: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            print(f"   Avg time: {r['avg_response_time']:.3f}s")
        
        # InsightSpike
        if 'insightspike' in results:
            r = results['insightspike']
            print(f"\n🔹 InsightSpike:")
            print(f"   Accuracy: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            print(f"   Avg time: {r['avg_response_time']:.3f}s")
            print(f"   Spikes detected: {r['spike_detections']}")
        
        # Dynamic Growth
        if 'dynamic_growth' in results:
            r = results['dynamic_growth']
            print(f"\n🔹 Dynamic Growth:")
            print(f"   Final episodes: {r['final_episodes']}")
            print(f"   Integration rate: {r['integration_rate']:.1%}")
            print(f"   Final accuracy: {r['accuracy']:.1%}")
            print(f"   Growth stages: {r['growth_stages']}")
        
        print("="*70)
    
    def save_results(self):
        """Save experiment results."""
        print("\n=== Saving Results ===")
        
        # Save experiment log
        log_file = self.results_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        print(f"✓ Log saved to {log_file}")
        
        # Save growth visualization data
        if self.experiment_log['phases']:
            growth_data = self.experiment_log['phases'][0]['results']
            viz_file = self.results_dir / "growth_visualization.json"
            with open(viz_file, 'w') as f:
                json.dump({
                    'checkpoints': [g['checkpoint'] for g in growth_data],
                    'episodes': [g['episodes'] for g in growth_data],
                    'accuracies': [g['test_accuracy'] for g in growth_data],
                    'integration_rates': [g['integration_rate'] for g in growth_data]
                }, f, indent=2)
            print(f"✓ Visualization data saved to {viz_file}")
    
    def backup_and_restore(self):
        """Backup experiment data and restore original."""
        print("\n=== Backing Up and Restoring ===")
        
        # Backup experiment data
        backup_dir = self.experiment_dir / "data_experiment"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(self.data_dir, backup_dir)
        print(f"✓ Experiment data backed up to {backup_dir}")
        
        # List backed up files
        for f in backup_dir.iterdir():
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  - {f.name}: {size_mb:.2f} MB")
        
        # Restore original data
        original_backup = self.experiment_dir / "data_backup_original"
        if original_backup.exists():
            # Clear current data
            shutil.rmtree(self.data_dir)
            self.data_dir.mkdir()
            
            # Copy back original
            for item in original_backup.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.data_dir / item.name)
                else:
                    shutil.copytree(item, self.data_dir / item.name)
            
            print("✓ Original data restored")


def main():
    """Run Experiment 10."""
    print("=== EXPERIMENT 10: DYNAMIC GROWTH RAG ===")
    print(f"Start time: {datetime.now()}")
    
    experiment = DynamicGrowthRAG()
    
    try:
        # Initialize clean data
        experiment.initialize_clean_data()
        
        # Load all datasets
        qa_pairs = experiment.load_all_datasets()
        
        if not qa_pairs:
            print("\n❌ No Q&A pairs loaded!")
            return
        
        # Test dynamic growth
        growth_results = experiment.test_dynamic_growth(qa_pairs)
        
        # Compare RAG approaches
        comparison_results = experiment.compare_rag_approaches(qa_pairs)
        
        # Save results
        experiment.save_results()
        
        print("\n✅ Experiment 10 complete!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always backup and restore
        experiment.backup_and_restore()
        
        print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()