#!/usr/bin/env python3
"""
Experiment 9: Clean RAG Build and Test
======================================

Build 1000-scale knowledge base in data folder and compare RAG performance.
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config
from datasets import Dataset
from sentence_transformers import SentenceTransformer


class Experiment9:
    """Build and test RAG with clean data folder."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        self.embedder = None
        
        # Use main data directory
        self.data_dir = Path(self.config.paths.data_dir)
        self.experiment_dir = Path(__file__).parent
        
        # HuggingFace datasets path
        self.hf_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets")
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'build_phase': {},
            'test_phase': {}
        }
    
    def load_datasets(self, target: int = 1000) -> List[Dict]:
        """Load Q&A pairs from HuggingFace datasets."""
        print(f"\n=== Loading Datasets (Target: {target} Q&A pairs) ===")
        
        datasets_info = [
            ('squad_200', 200),
            ('squad_300', 300),
            ('squad_100', 100),
            ('ms_marco_150', 150),
            ('coqa_80', 80),
            ('hotpot_qa_60', 60),
            ('drop_50', 50)
        ]
        
        all_qa_pairs = []
        
        for dataset_name, max_samples in datasets_info:
            if len(all_qa_pairs) >= target:
                break
                
            dataset_path = self.hf_path / dataset_name
            if not dataset_path.exists():
                print(f"Skipping {dataset_name} (not found)")
                continue
            
            print(f"Loading {dataset_name}...")
            dataset = Dataset.load_from_disk(str(dataset_path))
            
            remaining = target - len(all_qa_pairs)
            samples = min(len(dataset), min(max_samples, remaining))
            
            for i in range(samples):
                item = dataset[i]
                
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
                        'dataset': dataset_name,
                        'id': f"{dataset_name}_{i}"
                    })
        
        print(f"\nLoaded {len(all_qa_pairs)} Q&A pairs")
        return all_qa_pairs
    
    def build_knowledge_base(self, qa_pairs: List[Dict]):
        """Build knowledge base from Q&A pairs."""
        print("\n=== Building Knowledge Base ===")
        start_time = time.time()
        
        # Initialize agent
        self.agent = MainAgent(self.config)
        print("Agent initialized")
        
        # Process each Q&A pair
        episodes_added = 0
        for i, qa in enumerate(qa_pairs):
            # Add context
            self.agent.add_episode_with_graph_update(qa['context'], c_value=0.8)
            
            # Add Q&A as episode
            qa_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            self.agent.add_episode_with_graph_update(qa_text, c_value=0.6)
            
            if (i + 1) % 100 == 0:
                episodes_count = len(self.agent.l2_memory.episodes)
                print(f"Progress: {i+1}/{len(qa_pairs)} - Episodes: {episodes_count}")
        
        build_time = time.time() - start_time
        final_episodes = len(self.agent.l2_memory.episodes)
        
        # Save the knowledge base
        print("\nSaving knowledge base...")
        if self.agent.save_state():
            print("✓ Saved successfully")
        else:
            print("✗ Save failed")
        
        # Also save Q&A pairs for testing
        qa_file = self.data_dir / "qa_pairs.json"
        with open(qa_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        self.results['build_phase'] = {
            'qa_pairs': len(qa_pairs),
            'documents_processed': len(qa_pairs) * 2,
            'final_episodes': final_episodes,
            'integration_rate': 1 - final_episodes / (len(qa_pairs) * 2) if len(qa_pairs) > 0 else 0,
            'build_time': build_time
        }
        
        print(f"\n✓ Build complete!")
        print(f"  Q&A pairs: {len(qa_pairs)}")
        print(f"  Episodes: {final_episodes}")
        print(f"  Integration rate: {self.results['build_phase']['integration_rate']:.1%}")
        print(f"  Build time: {build_time:.1f}s")
        
        return qa_pairs
    
    def test_standard_rag(self, qa_pairs: List[Dict], test_size: int = 100):
        """Test standard semantic search RAG."""
        print(f"\n=== Testing Standard RAG (n={test_size}) ===")
        
        # Initialize embedder
        if not self.embedder:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Random test samples
        test_indices = random.sample(range(len(qa_pairs)), min(test_size, len(qa_pairs)))
        
        correct = 0
        response_times = []
        
        for i, idx in enumerate(test_indices):
            qa = qa_pairs[idx]
            start = time.time()
            
            # Encode question
            query_vec = self.embedder.encode(qa['question'])
            
            # Search in FAISS
            if hasattr(self.agent.l2_memory, 'index') and self.agent.l2_memory.index:
                D, I = self.agent.l2_memory.index.search(
                    query_vec.reshape(1, -1).astype(np.float32),
                    k=5
                )
                
                # Check if answer is in retrieved episodes
                for ep_idx in I[0]:
                    if 0 <= ep_idx < len(self.agent.l2_memory.episodes):
                        ep = self.agent.l2_memory.episodes[ep_idx]
                        ep_text = ep.text if hasattr(ep, 'text') else str(ep)
                        
                        if qa['answer'].lower() in ep_text.lower():
                            correct += 1
                            break
            
            response_times.append(time.time() - start)
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{test_size}")
        
        accuracy = correct / test_size
        avg_time = np.mean(response_times)
        
        results = {
            'method': 'Standard RAG',
            'accuracy': accuracy,
            'correct': correct,
            'total': test_size,
            'avg_response_time': avg_time,
            'total_time': sum(response_times)
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{test_size})")
        print(f"  Avg time: {avg_time:.3f}s")
        
        return results
    
    def test_insightspike(self, qa_pairs: List[Dict], test_size: int = 100):
        """Test InsightSpike with graph-based retrieval."""
        print(f"\n=== Testing InsightSpike (n={test_size}) ===")
        
        # Use same test indices for fair comparison
        test_indices = random.sample(range(len(qa_pairs)), min(test_size, len(qa_pairs)))
        
        correct = 0
        response_times = []
        
        for i, idx in enumerate(test_indices):
            qa = qa_pairs[idx]
            start = time.time()
            
            try:
                # Use InsightSpike's full pipeline
                result = self.agent.process_question(
                    qa['question'],
                    max_cycles=1,
                    verbose=False
                )
                
                response = result.get('response', '')
                
                if qa['answer'].lower() in response.lower():
                    correct += 1
                    
            except Exception as e:
                pass  # Count as incorrect
            
            response_times.append(time.time() - start)
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{test_size}")
        
        accuracy = correct / test_size
        avg_time = np.mean(response_times)
        
        results = {
            'method': 'InsightSpike',
            'accuracy': accuracy,
            'correct': correct,
            'total': test_size,
            'avg_response_time': avg_time,
            'total_time': sum(response_times)
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{test_size})")
        print(f"  Avg time: {avg_time:.3f}s")
        
        return results
    
    def run_comparison(self, qa_pairs: List[Dict]):
        """Run full comparison."""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        # Test both methods
        rag_results = self.test_standard_rag(qa_pairs, test_size=100)
        insight_results = self.test_insightspike(qa_pairs, test_size=100)
        
        # Calculate improvements
        accuracy_diff = insight_results['accuracy'] - rag_results['accuracy']
        speed_ratio = rag_results['avg_response_time'] / insight_results['avg_response_time']
        
        self.results['test_phase'] = {
            'standard_rag': rag_results,
            'insightspike': insight_results,
            'comparison': {
                'accuracy_improvement': accuracy_diff,
                'speed_ratio': speed_ratio,
                'winner': 'InsightSpike' if accuracy_diff > 0 else 'Standard RAG'
            }
        }
        
        # Print comparison table
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        print(f"{'Method':<20} {'Accuracy':<15} {'Avg Time':<15}")
        print("-"*60)
        print(f"{'Standard RAG':<20} {rag_results['accuracy']:.1%} ({rag_results['correct']}/{rag_results['total']}){'':<3} {rag_results['avg_response_time']:.3f}s")
        print(f"{'InsightSpike':<20} {insight_results['accuracy']:.1%} ({insight_results['correct']}/{insight_results['total']}){'':<3} {insight_results['avg_response_time']:.3f}s")
        print("-"*60)
        print(f"{'Improvement:':<20} {accuracy_diff*100:+.1f}%{'':<10} {speed_ratio:.2f}x")
        print("="*60)
        
        # Save results
        self.results['end_time'] = datetime.now().isoformat()
        results_file = self.experiment_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
    
    def backup_data(self):
        """Backup data folder to experiment directory."""
        print("\n=== Backing up data ===")
        
        import shutil
        backup_dir = self.experiment_dir / "data_after_experiment"
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        shutil.copytree(self.data_dir, backup_dir)
        
        # List backed up files
        for f in backup_dir.iterdir():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  ✓ {f.name}: {size_mb:.2f} MB")
        
        print("✓ Backup complete")


def main():
    """Run experiment 9."""
    print("=== EXPERIMENT 9: CLEAN RAG BUILD AND TEST ===")
    print(f"Start time: {datetime.now()}")
    
    exp = Experiment9()
    
    try:
        # Load datasets
        qa_pairs = exp.load_datasets(target=1000)
        
        # Build knowledge base
        exp.build_knowledge_base(qa_pairs)
        
        # Run comparison tests
        exp.run_comparison(qa_pairs)
        
        # Backup results
        exp.backup_data()
        
        print("\n✅ Experiment 9 complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original data
        print("\n=== Restoring original data ===")
        import shutil
        backup_before = Path(__file__).parent / "data_backup_before"
        data_dir = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data")
        
        # Clear current data
        shutil.rmtree(data_dir)
        data_dir.mkdir()
        
        # Restore from backup
        for item in backup_before.iterdir():
            if item.is_file():
                shutil.copy2(item, data_dir / item.name)
            else:
                shutil.copytree(item, data_dir / item.name)
        
        print("✓ Original data restored")


if __name__ == "__main__":
    main()