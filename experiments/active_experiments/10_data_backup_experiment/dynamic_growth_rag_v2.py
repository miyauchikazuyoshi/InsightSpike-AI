#!/usr/bin/env python3
"""
Experiment 10 V2: Dynamic Growth RAG with Better Error Handling
===============================================================
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
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config
from datasets import Dataset
from sentence_transformers import SentenceTransformer


class DynamicGrowthRAGV2:
    """Test dynamic growth capabilities with better error handling."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        self.embedder = None
        
        # Experiment paths
        self.experiment_dir = Path(__file__).parent
        self.data_dir = Path(self.config.paths.data_dir)
        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Track experiment progress
        self.experiment_log = {
            'start_time': datetime.now().isoformat(),
            'phases': [],
            'growth_metrics': [],
            'final_results': {}
        }
    
    def load_datasets_safely(self) -> List[Dict[str, Any]]:
        """Load datasets with better error handling."""
        print("\n=== Loading HuggingFace Datasets ===")
        
        all_qa_pairs = []
        
        # Define specific datasets to load
        datasets_to_load = [
            ('huggingface_datasets/squad_30', 30),
            ('large_huggingface_datasets/squad_100', 100),
            ('mega_huggingface_datasets/squad_200', 200),
            ('mega_huggingface_datasets/squad_300', 300),
            ('mega_huggingface_datasets/hotpot_qa_60', 60),
        ]
        
        for dataset_path, expected_size in datasets_to_load:
            full_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data") / dataset_path
            
            if not full_path.exists():
                print(f"⚠️  Skipping {dataset_path} (not found)")
                continue
            
            try:
                dataset = Dataset.load_from_disk(str(full_path))
                qa_count = 0
                
                for i, item in enumerate(dataset):
                    if i >= expected_size:
                        break
                    
                    context = item.get('context', '')
                    question = item.get('question', '')
                    answers = item.get('answers', {})
                    
                    # Handle different answer formats
                    answer = ""
                    if isinstance(answers, dict) and 'text' in answers:
                        texts = answers.get('text', [])
                        if texts and isinstance(texts, list):
                            answer = texts[0]
                    elif isinstance(answers, str):
                        answer = answers
                    
                    if context and question and answer:
                        all_qa_pairs.append({
                            'context': context,
                            'question': question,
                            'answer': answer,
                            'dataset': dataset_path,
                            'id': f"{dataset_path}_{qa_count}"
                        })
                        qa_count += 1
                
                print(f"✓ {dataset_path}: {qa_count} Q&A pairs")
                
            except Exception as e:
                print(f"✗ Error loading {dataset_path}: {e}")
        
        print(f"\n📊 Total Q&A pairs loaded: {len(all_qa_pairs)}")
        return all_qa_pairs
    
    def test_dynamic_growth_robust(self, qa_pairs: List[Dict[str, Any]]):
        """Test dynamic growth with robust error handling."""
        print("\n=== Testing Dynamic Growth RAG ===")
        
        # Initialize agent
        self.agent = MainAgent(self.config)
        print("✓ Agent initialized")
        
        # Define checkpoints
        checkpoints = [10, 50, 100, 250, 500, min(690, len(qa_pairs))]
        growth_results = []
        
        for checkpoint in checkpoints:
            if checkpoint > len(qa_pairs):
                continue
                
            print(f"\n📈 Growing to {checkpoint} documents...")
            
            try:
                start_time = time.time()
                
                # Get current episode count
                current_episodes = len(self.agent.l2_memory.episodes)
                target_docs = checkpoint * 2  # 2 docs per Q&A pair
                
                # Add documents
                added = 0
                for i in range(current_episodes // 2, checkpoint):
                    if i >= len(qa_pairs):
                        break
                    
                    qa = qa_pairs[i]
                    
                    try:
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
                        
                        added += 2
                        
                    except Exception as e:
                        print(f"  Warning: Failed to add document {i}: {e}")
                        continue
                    
                    # Progress
                    if (i + 1) % 50 == 0:
                        episodes = len(self.agent.l2_memory.episodes)
                        print(f"  Progress: {i+1} Q&A pairs → {episodes} episodes")
                
                growth_time = time.time() - start_time
                
                # Get stats
                memory_stats = self.agent.l2_memory.get_memory_stats()
                final_episodes = memory_stats.get('total_episodes', 0)
                
                # Test accuracy
                test_accuracy = self._test_accuracy_safely(qa_pairs[:checkpoint])
                
                result = {
                    'checkpoint': checkpoint,
                    'episodes': final_episodes,
                    'integration_rate': 1 - final_episodes / (checkpoint * 2) if checkpoint > 0 else 0,
                    'growth_time': growth_time,
                    'test_accuracy': test_accuracy,
                    'docs_added': added
                }
                
                growth_results.append(result)
                
                print(f"  ✓ Episodes: {final_episodes}")
                print(f"  ✓ Integration rate: {result['integration_rate']:.1%}")
                print(f"  ✓ Test accuracy: {test_accuracy:.1%}")
                print(f"  ✓ Growth time: {growth_time:.1f}s")
                
            except Exception as e:
                print(f"  ❌ Error at checkpoint {checkpoint}: {e}")
                traceback.print_exc()
        
        # Save state
        print("\n💾 Saving state...")
        try:
            if self.agent.save_state():
                print("✓ State saved")
                
            # Save Q&A pairs
            qa_file = self.data_dir / "qa_pairs.json"
            with open(qa_file, 'w') as f:
                json.dump(qa_pairs, f, indent=2)
            print("✓ Q&A pairs saved")
            
        except Exception as e:
            print(f"✗ Save failed: {e}")
        
        return growth_results
    
    def _test_accuracy_safely(self, qa_pairs: List[Dict], sample_size: int = 30) -> float:
        """Test accuracy with error handling."""
        if not qa_pairs:
            return 0.0
        
        sample_size = min(sample_size, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), sample_size)
        
        correct = 0
        tested = 0
        
        for idx in test_indices:
            try:
                qa = qa_pairs[idx]
                
                # Search for answer
                results = self.agent.l2_memory.search_episodes(
                    qa['question'],
                    k=5
                )
                
                # Check if answer is in any result
                for result in results:
                    text = result.get('text', '')
                    # Handle both string and other types
                    if isinstance(text, str) and isinstance(qa['answer'], str):
                        if qa['answer'].lower() in text.lower():
                            correct += 1
                            break
                
                tested += 1
                
            except Exception as e:
                continue
        
        return correct / tested if tested > 0 else 0.0
    
    def run_rag_comparison(self, qa_pairs: List[Dict]):
        """Run simplified RAG comparison."""
        print("\n=== RAG Comparison ===")
        
        if not self.embedder:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test on smaller sample
        test_size = min(100, len(qa_pairs))
        test_indices = random.sample(range(len(qa_pairs)), test_size)
        
        # Standard RAG test
        print("\n📊 Testing Standard RAG...")
        standard_correct = 0
        
        for idx in test_indices[:50]:  # Test 50 for speed
            try:
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
                            ep_text = ep.text if hasattr(ep, 'text') else ""
                            
                            if isinstance(ep_text, str) and qa['answer'].lower() in ep_text.lower():
                                standard_correct += 1
                                break
            except:
                continue
        
        standard_accuracy = standard_correct / 50
        print(f"  Standard RAG Accuracy: {standard_accuracy:.1%} ({standard_correct}/50)")
        
        return {
            'standard_rag': {
                'accuracy': standard_accuracy,
                'tested': 50
            }
        }
    
    def save_and_cleanup(self, growth_results, comparison_results):
        """Save results and cleanup."""
        print("\n=== Saving Results ===")
        
        # Prepare final results
        final_results = {
            'experiment': 'Dynamic Growth RAG',
            'timestamp': datetime.now().isoformat(),
            'growth_results': growth_results,
            'comparison': comparison_results,
            'summary': {
                'total_qa_pairs': self.experiment_log.get('total_qa_pairs', 0),
                'final_episodes': growth_results[-1]['episodes'] if growth_results else 0,
                'final_accuracy': growth_results[-1]['test_accuracy'] if growth_results else 0,
                'final_integration_rate': growth_results[-1]['integration_rate'] if growth_results else 0
            }
        }
        
        # Save results
        results_file = self.results_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"✓ Results saved to {results_file}")
        
        # Backup data
        print("\n=== Backing up data ===")
        backup_dir = self.experiment_dir / "data_final"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(self.data_dir, backup_dir)
        print(f"✓ Data backed up to {backup_dir}")
        
        # Restore original
        original_backup = self.experiment_dir / "data_backup_original"
        if original_backup.exists():
            shutil.rmtree(self.data_dir)
            self.data_dir.mkdir()
            
            for item in original_backup.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.data_dir / item.name)
                else:
                    shutil.copytree(item, self.data_dir / item.name)
            
            print("✓ Original data restored")


def main():
    """Run experiment with better error handling."""
    print("=== EXPERIMENT 10 V2: DYNAMIC GROWTH RAG ===")
    print(f"Start: {datetime.now()}")
    
    experiment = DynamicGrowthRAGV2()
    
    # Initialize clean data
    print("\n=== Initializing Clean Data ===")
    files_to_clear = ['episodes.json', 'index.faiss', 'graph_pyg.pt', 
                     'scalable_index.faiss', 'insight_facts.db', 'unknown_learning.db']
    
    for filename in files_to_clear:
        filepath = experiment.data_dir / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  Removed: {filename}")
    
    # Run experiment
    try:
        # Load datasets
        qa_pairs = experiment.load_datasets_safely()
        experiment.experiment_log['total_qa_pairs'] = len(qa_pairs)
        
        if not qa_pairs:
            print("❌ No data loaded!")
            return
        
        # Test dynamic growth
        growth_results = experiment.test_dynamic_growth_robust(qa_pairs)
        
        # Run comparison
        comparison_results = experiment.run_rag_comparison(qa_pairs)
        
        # Save and cleanup
        experiment.save_and_cleanup(growth_results, comparison_results)
        
        print("\n✅ Experiment complete!")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    
    finally:
        print(f"\nEnd: {datetime.now()}")


if __name__ == "__main__":
    main()