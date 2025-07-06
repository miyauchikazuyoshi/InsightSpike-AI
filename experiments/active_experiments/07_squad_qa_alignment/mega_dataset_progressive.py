#!/usr/bin/env python3
"""
Progressive Mega Dataset Experiment
===================================

Build knowledge graph progressively with checkpoints and resume capability.
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config


class ProgressiveMegaExperiment:
    """Progressive experiment with checkpoint saving."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        
        # Paths
        self.experiment_dir = Path(__file__).parent
        self.data_dir = self.experiment_dir / "data"
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Update config
        self.config.paths.data_dir = str(self.data_dir)
        self.config.memory.index_file = str(self.data_dir / "index.faiss")
        self.config.reasoning.graph_file = str(self.data_dir / "graph_pyg.pt")
        
        # Dataset path
        self.mega_dataset_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets")
        
        # Available datasets (smaller batches for stability)
        self.datasets_config = [
            {'name': 'squad_200', 'batch_size': 50, 'max_docs': 400},
            {'name': 'squad_300', 'batch_size': 50, 'max_docs': 600},
            {'name': 'ms_marco_150', 'batch_size': 30, 'max_docs': 450},
            {'name': 'coqa_80', 'batch_size': 20, 'max_docs': 240},
            {'name': 'hotpot_qa_60', 'batch_size': 20, 'max_docs': 180},
            {'name': 'drop_50', 'batch_size': 20, 'max_docs': 100},
            {'name': 'boolq_50', 'batch_size': 20, 'max_docs': 100},
            {'name': 'commonsense_qa_20', 'batch_size': 10, 'max_docs': 40}
        ]
        
        # Progress tracking
        self.progress_file = self.experiment_dir / "progress.json"
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict[str, Any]:
        """Load progress from checkpoint."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'start_time': datetime.now().isoformat(),
                'datasets_completed': [],
                'current_dataset': None,
                'current_batch': 0,
                'total_documents': 0,
                'total_episodes': 0,
                'checkpoints': []
            }
    
    def save_progress(self):
        """Save progress checkpoint."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def initialize_or_load_agent(self):
        """Initialize agent or load from checkpoint."""
        print("\n=== Initializing Agent ===")
        
        self.agent = MainAgent(self.config)
        
        # Replace with enhanced memory
        self.agent.l2_memory = L2EnhancedScalableMemory(
            dim=self.config.embedding.dimension,
            config=self.config,
            use_scalable_graph=True
        )
        
        # Try to load existing state
        if (self.data_dir / "episodes.json").exists():
            print("Loading existing agent state...")
            if self.agent.load_state():
                stats = self.agent.get_stats()
                memory_stats = stats.get('memory_stats', {})
                print(f"Loaded {memory_stats.get('total_episodes', 0)} episodes")
                self.progress['total_episodes'] = memory_stats.get('total_episodes', 0)
            else:
                print("Failed to load state, starting fresh")
                self.agent.initialize()
        else:
            print("Starting with fresh agent")
            self.agent.initialize()
    
    def load_dataset_batch(self, dataset_name: str, batch_start: int, batch_size: int) -> List[str]:
        """Load a batch of documents from dataset."""
        dataset_dir = self.mega_dataset_path / dataset_name
        
        if not dataset_dir.exists():
            return []
        
        try:
            dataset = Dataset.load_from_disk(str(dataset_dir))
            documents = []
            
            # Get batch of samples
            end_idx = min(batch_start + batch_size, len(dataset))
            batch_items = dataset[batch_start:end_idx]
            
            # Extract based on dataset type
            if 'squad' in dataset_name:
                for i in range(len(batch_items['id'])):
                    if batch_items.get('context') and batch_items['context'][i]:
                        documents.append(batch_items['context'][i])
                    if batch_items.get('question') and batch_items['question'][i]:
                        documents.append(batch_items['question'][i])
            
            elif 'ms_marco' in dataset_name:
                for i in range(len(batch_items['id'])):
                    if batch_items.get('query') and batch_items['query'][i]:
                        documents.append(batch_items['query'][i])
                    # Handle nested passages structure
                    if batch_items.get('passages') and batch_items['passages'][i]:
                        passages = batch_items['passages'][i]
                        if 'passage_text' in passages:
                            for p in passages['passage_text'][:2]:  # Limit passages
                                if p:
                                    documents.append(p)
            
            elif 'coqa' in dataset_name:
                for i in range(len(batch_items['id'])):
                    if batch_items.get('story') and batch_items['story'][i]:
                        documents.append(batch_items['story'][i])
                    if batch_items.get('questions') and batch_items['questions'][i]:
                        documents.append(batch_items['questions'][i])
            
            else:
                # Generic extraction for other datasets
                for key in ['passage', 'context', 'question', 'query', 'text']:
                    if key in batch_items:
                        for i in range(len(batch_items[key])):
                            if batch_items[key][i]:
                                documents.append(batch_items[key][i])
            
            return documents
            
        except Exception as e:
            print(f"Error loading batch from {dataset_name}: {e}")
            return []
    
    def process_batch(self, documents: List[str], dataset_name: str) -> Dict[str, Any]:
        """Process a batch of documents."""
        results = {
            'successful': 0,
            'failed': 0,
            'start_episodes': len(self.agent.l2_memory.episodes)
        }
        
        for doc in documents:
            if not doc or len(doc.strip()) < 20:
                continue
            
            if len(doc) > 2000:
                doc = doc[:2000] + "..."
            
            try:
                result = self.agent.add_episode_with_graph_update(doc)
                if result.get('success', False):
                    results['successful'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                print(f"Error processing document: {e}")
                results['failed'] += 1
        
        results['end_episodes'] = len(self.agent.l2_memory.episodes)
        results['new_episodes'] = results['end_episodes'] - results['start_episodes']
        
        return results
    
    def run_progressive_experiment(self):
        """Run experiment with progressive loading and checkpoints."""
        print("\n=== PROGRESSIVE MEGA DATASET EXPERIMENT ===")
        print(f"Progress: {len(self.progress['datasets_completed'])} datasets completed")
        print(f"Current episodes: {self.progress['total_episodes']}")
        
        # Initialize agent
        self.initialize_or_load_agent()
        
        # Process datasets
        for dataset_config in self.datasets_config:
            dataset_name = dataset_config['name']
            
            # Skip completed datasets
            if dataset_name in self.progress['datasets_completed']:
                print(f"\nSkipping completed dataset: {dataset_name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")
            
            # Resume from last batch if needed
            start_batch = 0
            if self.progress['current_dataset'] == dataset_name:
                start_batch = self.progress['current_batch']
                print(f"Resuming from batch {start_batch}")
            
            # Process in batches
            batch_size = dataset_config['batch_size']
            max_docs = dataset_config['max_docs']
            dataset_start_time = time.time()
            dataset_docs = 0
            
            for batch_idx in range(start_batch, max_docs, batch_size):
                # Load batch
                print(f"\nBatch {batch_idx//batch_size + 1}: Loading {batch_size} samples...")
                documents = self.load_dataset_batch(dataset_name, batch_idx, batch_size)
                
                if not documents:
                    break
                
                # Process batch
                batch_results = self.process_batch(documents, dataset_name)
                dataset_docs += batch_results['successful']
                
                # Update progress
                self.progress['current_dataset'] = dataset_name
                self.progress['current_batch'] = batch_idx + batch_size
                self.progress['total_documents'] += batch_results['successful']
                self.progress['total_episodes'] = batch_results['end_episodes']
                
                # Report
                print(f"  Processed: {batch_results['successful']} docs")
                print(f"  New episodes: {batch_results['new_episodes']}")
                print(f"  Total episodes: {self.progress['total_episodes']}")
                
                # Save checkpoint every batch
                self.save_checkpoint()
                
                # Check if we reached 1000 episodes
                if self.progress['total_episodes'] >= 1000:
                    print(f"\n🎯 Reached {self.progress['total_episodes']} episodes!")
                    self.progress['datasets_completed'].append(dataset_name)
                    self.save_progress()
                    return
            
            # Dataset completed
            dataset_time = time.time() - dataset_start_time
            print(f"\n{dataset_name} completed in {dataset_time:.1f}s")
            print(f"  Documents: {dataset_docs}")
            
            self.progress['datasets_completed'].append(dataset_name)
            self.progress['current_dataset'] = None
            self.progress['current_batch'] = 0
            self.save_progress()
        
        print("\n=== EXPERIMENT COMPLETE ===")
        print(f"Total documents: {self.progress['total_documents']}")
        print(f"Total episodes: {self.progress['total_episodes']}")
    
    def save_checkpoint(self):
        """Save agent state as checkpoint."""
        checkpoint_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save agent state
        if self.agent.save_state():
            # Record checkpoint
            checkpoint = {
                'timestamp': checkpoint_time,
                'episodes': self.progress['total_episodes'],
                'documents': self.progress['total_documents'],
                'dataset': self.progress['current_dataset'],
                'batch': self.progress['current_batch']
            }
            self.progress['checkpoints'].append(checkpoint)
            self.save_progress()
            
            # Get graph stats
            if hasattr(self.agent.l2_memory, 'get_graph_stats'):
                stats = self.agent.l2_memory.get_graph_stats()
                print(f"  Checkpoint saved: {stats.get('nodes', 0)} nodes, {stats.get('edges', 0)} edges")
    
    def generate_summary(self):
        """Generate summary of the experiment."""
        print("\n=== GENERATING SUMMARY ===")
        
        # Load final state
        self.initialize_or_load_agent()
        
        # Get statistics
        stats = self.agent.get_stats()
        memory_stats = stats.get('memory_stats', {})
        
        if hasattr(self.agent.l2_memory, 'get_graph_stats'):
            graph_stats = self.agent.l2_memory.get_graph_stats()
        else:
            graph_stats = {}
        
        summary = {
            'experiment': 'Progressive Mega Dataset',
            'completed': datetime.now().isoformat(),
            'datasets_processed': len(self.progress['datasets_completed']),
            'total_documents': self.progress['total_documents'],
            'total_episodes': memory_stats.get('total_episodes', 0),
            'graph_nodes': graph_stats.get('nodes', 0),
            'graph_edges': graph_stats.get('edges', 0),
            'integration_rate': 1 - (memory_stats.get('total_episodes', 0) / max(1, self.progress['total_documents'])),
            'checkpoints': len(self.progress['checkpoints']),
            'datasets': self.progress['datasets_completed']
        }
        
        # Save summary
        summary_file = self.experiment_dir / f'mega_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary:")
        print(f"  Documents processed: {summary['total_documents']}")
        print(f"  Episodes created: {summary['total_episodes']}")
        print(f"  Integration rate: {summary['integration_rate']*100:.1f}%")
        print(f"  Graph size: {summary['graph_nodes']} nodes, {summary['graph_edges']} edges")
        print(f"  Summary saved to: {summary_file}")


def main():
    """Run progressive mega experiment."""
    experiment = ProgressiveMegaExperiment()
    
    try:
        # Run experiment
        experiment.run_progressive_experiment()
        
        # Generate summary
        experiment.generate_summary()
        
        print("\n✅ Progressive experiment completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted - progress saved")
        experiment.save_checkpoint()
        experiment.save_progress()
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save progress
        try:
            experiment.save_checkpoint()
            experiment.save_progress()
        except:
            pass


if __name__ == "__main__":
    main()