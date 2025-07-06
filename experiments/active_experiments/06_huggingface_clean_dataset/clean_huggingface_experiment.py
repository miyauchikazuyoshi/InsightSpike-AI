#!/usr/bin/env python3
"""
Clean HuggingFace Dataset Experiment
====================================

Test with real datasets starting from a completely clean state.
"""

import os
import sys
import json
import time
import shutil
import pandas as pd
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
from insightspike.monitoring import create_default_monitor
from insightspike.core.config import get_config


class CleanHuggingFaceExperiment:
    """Run clean experiments with real HuggingFace datasets."""
    
    def __init__(self, use_temp_dir: bool = True):
        self.config = get_config()
        self.monitor = create_default_monitor()
        self.agent = None
        self.dataset_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/huggingface_datasets")
        
        # Use temporary data directory to ensure clean state
        if use_temp_dir:
            self.original_data_dir = Path(self.config.paths.data_dir)
            self.temp_data_dir = Path(__file__).parent / "temp_data"
            self.temp_data_dir.mkdir(exist_ok=True)
            # Update config to use temp directory
            self.config.paths.data_dir = str(self.temp_data_dir)
            self.config.memory.index_file = str(self.temp_data_dir / "index.faiss")
            print(f"Using temporary data directory: {self.temp_data_dir}")
        
        self.results = {
            'datasets': {},
            'metrics': {
                'episodes': [],
                'nodes': [],
                'edges': [],
                'processing_times': [],
                'integration_rates': [],
                'documents_processed': []
            },
            'clean_start': True,
            'initial_state': None
        }
    
    def load_dataset(self, dataset_name: str) -> List[str]:
        """Load a HuggingFace dataset from Arrow format."""
        dataset_dir = self.dataset_path / dataset_name
        
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return []
        
        try:
            # Load dataset from Arrow format
            dataset = Dataset.load_from_disk(str(dataset_dir))
            
            print(f"Loading {dataset_name} with {len(dataset)} samples...")
            
            documents = []
            
            # Extract text based on dataset structure
            if dataset_name == "squad_30":
                # SQuAD format: questions and contexts
                for item in dataset:
                    if 'context' in item:
                        documents.append(item['context'])
                    if 'question' in item:
                        documents.append(item['question'])
            
            elif dataset_name == "ms_marco_20":
                # MS MARCO format: queries and passages
                for item in dataset:
                    if 'query' in item:
                        documents.append(item['query'])
                    if 'passages' in item and 'passage_text' in item['passages']:
                        # MS MARCO has passages as a dict with passage_text list
                        for passage_text in item['passages']['passage_text']:
                            if passage_text:
                                documents.append(passage_text)
            
            print(f"Loaded {len(documents)} documents from {dataset_name}")
            return documents
            
        except ImportError:
            print("Error: 'datasets' library not installed. Run: pip install datasets")
            return []
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def initialize_agent(self):
        """Initialize agent with enhanced scalable memory in clean state."""
        print("\nInitializing agent with scalable features...")
        
        self.agent = MainAgent()
        
        # Replace with enhanced memory
        old_memory = self.agent.l2_memory
        self.agent.l2_memory = L2EnhancedScalableMemory(
            dim=self.config.embedding.dimension,
            config=self.config,
            use_scalable_graph=True
        )
        
        # Do NOT call agent.initialize() to avoid loading existing state
        # Instead, initialize components directly
        self.agent.l2_memory.initialize()
        if hasattr(self.agent, 'l3_graph') and self.agent.l3_graph:
            self.agent.l3_graph.initialize()
        
        # Verify clean state
        initial_stats = self.agent.get_stats()
        self.results['initial_state'] = initial_stats
        
        memory_stats = initial_stats.get('memory_stats', {})
        print(f"Agent initialized in clean state:")
        print(f"  Initial episodes: {memory_stats.get('total_episodes', 0)}")
        print(f"  Initial graph nodes: {memory_stats.get('graph_nodes', 0)}")
        
        if memory_stats.get('total_episodes', 0) > 0:
            print("WARNING: Agent has pre-existing episodes! Not a clean start.")
            self.results['clean_start'] = False
    
    def process_dataset(self, dataset_name: str, max_docs: int = None):
        """Process a dataset and track metrics."""
        print(f"\n=== Processing {dataset_name} ===" )
        
        # Load dataset
        documents = self.load_dataset(dataset_name)
        if not documents:
            return
        
        if max_docs:
            documents = documents[:max_docs]
        
        # Initialize metrics for this dataset
        self.results['datasets'][dataset_name] = {
            'total_documents': len(documents),
            'processing_times': [],
            'episodes_created': 0,
            'episodes_before': len(self.agent.l2_memory.episodes),
            'integrations': 0,
            'conflicts': 0,
            'graph_stats': {}
        }
        
        # Process documents
        start_time = time.time()
        successful = 0
        failed = 0
        
        for i, doc in enumerate(documents):
            doc_start = time.time()
            
            # Skip empty documents
            if not doc or len(doc.strip()) < 10:
                continue
            
            # Truncate very long documents
            if len(doc) > 1000:
                doc = doc[:1000] + "..."
            
            try:
                result = self.agent.add_episode_with_graph_update(doc)
                doc_time = time.time() - doc_start
                
                if result.get('success', False):
                    successful += 1
                    self.results['datasets'][dataset_name]['processing_times'].append(doc_time)
                    
                    # Track integration
                    if result.get('episode_idx', -1) < len(self.agent.l2_memory.episodes) - 1:
                        self.results['datasets'][dataset_name]['integrations'] += 1
                else:
                    failed += 1
                
                # Progress report
                if (i + 1) % 50 == 0:
                    self._report_progress(dataset_name, i + 1, len(documents))
                    
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                failed += 1
        
        # Final statistics
        total_time = time.time() - start_time
        dataset_results = self.results['datasets'][dataset_name]
        dataset_results['successful'] = successful
        dataset_results['failed'] = failed
        dataset_results['total_time'] = total_time
        dataset_results['episodes_after'] = len(self.agent.l2_memory.episodes)
        dataset_results['episodes_created'] = dataset_results['episodes_after'] - dataset_results['episodes_before']
        
        # Get graph stats
        if hasattr(self.agent.l2_memory, 'get_graph_stats'):
            dataset_results['graph_stats'] = self.agent.l2_memory.get_graph_stats()
        
        print(f"\nCompleted {dataset_name}:")
        print(f"  Documents processed: {successful}/{len(documents)}")
        print(f"  Episodes before: {dataset_results['episodes_before']}")
        print(f"  Episodes after: {dataset_results['episodes_after']}")
        print(f"  New episodes created: {dataset_results['episodes_created']}")
        print(f"  Integrations: {dataset_results['integrations']}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per doc: {total_time/max(1, successful):.3f}s")
    
    def _report_progress(self, dataset_name: str, processed: int, total: int):
        """Report processing progress."""
        # Get current state
        stats = self.agent.get_stats()
        memory_stats = stats.get('memory_stats', {})
        
        if hasattr(self.agent.l2_memory, 'get_graph_stats'):
            graph_stats = self.agent.l2_memory.get_graph_stats()
        else:
            graph_stats = {}
        
        print(f"\n[{dataset_name}] Progress: {processed}/{total}")
        print(f"  Episodes: {memory_stats.get('total_episodes', 0)}")
        print(f"  Graph: {graph_stats.get('nodes', 0)} nodes, {graph_stats.get('edges', 0)} edges")
        
        # Record metrics
        self.results['metrics']['documents_processed'].append(processed)
        self.results['metrics']['episodes'].append(memory_stats.get('total_episodes', 0))
        self.results['metrics']['nodes'].append(graph_stats.get('nodes', 0))
        self.results['metrics']['edges'].append(graph_stats.get('edges', 0))
        
        # Calculate integration rate
        if processed > 0:
            integration_rate = 1 - (memory_stats.get('total_episodes', 0) / processed)
            self.results['metrics']['integration_rates'].append(integration_rate)
    
    def compare_datasets(self):
        """Compare results across datasets."""
        print("\n=== Dataset Comparison ===")
        
        comparison_data = []
        for dataset_name, results in self.results['datasets'].items():
            if results['successful'] > 0:
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Documents': results['successful'],
                    'Episodes Before': results['episodes_before'],
                    'Episodes After': results['episodes_after'],
                    'New Episodes': results['episodes_created'],
                    'Integration Rate': f"{(1 - results['episodes_created']/results['successful'])*100:.1f}%",
                    'Avg Time (s)': f"{sum(results['processing_times'])/len(results['processing_times']):.3f}",
                    'Graph Nodes': results['graph_stats'].get('nodes', 0),
                    'Graph Edges': results['graph_stats'].get('edges', 0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            
            # Save comparison
            csv_file = Path(__file__).parent / 'clean_dataset_comparison.csv'
            df.to_csv(csv_file, index=False)
            print(f"\nSaved comparison to: {csv_file}")
    
    def save_results(self):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = Path(__file__).parent / f'clean_huggingface_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nSaved results to: {results_file}")
        
        # Save final state
        if self.agent:
            print("\nSaving final agent state...")
            if self.agent.save_state():
                print("✓ Agent state saved successfully")
            else:
                print("✗ Failed to save agent state")
    
    def cleanup(self):
        """Clean up temporary directory if used."""
        if hasattr(self, 'temp_data_dir') and self.temp_data_dir.exists():
            print(f"\nCleaning up temporary directory: {self.temp_data_dir}")
            shutil.rmtree(self.temp_data_dir)


def main():
    """Run the clean HuggingFace dataset experiment."""
    print("=== Clean HuggingFace Dataset Experiment ===")
    print(f"Start time: {datetime.now()}")
    
    experiment = CleanHuggingFaceExperiment(use_temp_dir=True)
    
    try:
        # Initialize agent in clean state
        experiment.initialize_agent()
        
        # Process each dataset
        datasets = ['squad_30', 'ms_marco_20']
        for dataset in datasets:
            experiment.process_dataset(dataset, max_docs=200)  # Limit for testing
        
        # Compare results
        experiment.compare_datasets()
        
        # Save results
        experiment.save_results()
        
        print(f"\n✅ Experiment completed at: {datetime.now()}")
        print(f"Clean start verified: {experiment.results['clean_start']}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        experiment.cleanup()


if __name__ == "__main__":
    main()