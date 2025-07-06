#!/usr/bin/env python3
"""
Experiment 7: Mega Dataset Experiment with 1000+ Episodes
=========================================================

Build a comprehensive knowledge graph using multiple large HuggingFace datasets.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

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


class MegaDatasetExperiment:
    """Build large-scale knowledge graph from multiple datasets."""
    
    def __init__(self):
        self.config = get_config()
        self.monitor = create_default_monitor()
        self.agent = None
        
        # Dataset paths
        self.mega_dataset_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets")
        
        # Use experiment-specific data directory
        self.experiment_dir = Path(__file__).parent
        self.data_dir = self.experiment_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Update config to use experiment data directory
        self.config.paths.data_dir = str(self.data_dir)
        self.config.memory.index_file = str(self.data_dir / "index.faiss")
        self.config.reasoning.graph_file = str(self.data_dir / "graph_pyg.pt")
        
        print(f"Using data directory: {self.data_dir}")
        
        # Available datasets with expected document counts
        self.available_datasets = {
            'squad_200': {'samples': 200, 'priority': 1, 'type': 'qa'},
            'squad_300': {'samples': 300, 'priority': 2, 'type': 'qa'},
            'ms_marco_150': {'samples': 150, 'priority': 3, 'type': 'retrieval'},
            'coqa_80': {'samples': 80, 'priority': 4, 'type': 'conversational'},
            'hotpot_qa_60': {'samples': 60, 'priority': 5, 'type': 'multi-hop'},
            'drop_50': {'samples': 50, 'priority': 6, 'type': 'numerical'},
            'boolq_50': {'samples': 50, 'priority': 7, 'type': 'boolean'},
            'commonsense_qa_20': {'samples': 20, 'priority': 8, 'type': 'commonsense'}
        }
        
        # Results tracking
        self.results = {
            'start_time': datetime.now().isoformat(),
            'datasets_processed': {},
            'global_metrics': {
                'total_documents': 0,
                'total_episodes': 0,
                'total_nodes': 0,
                'total_edges': 0,
                'processing_time': 0,
                'integration_rate': 0
            },
            'timeline': {
                'documents': [],
                'episodes': [],
                'nodes': [],
                'edges': [],
                'timestamps': [],
                'datasets': []
            },
            'performance': {
                'doc_per_second': [],
                'memory_usage': [],
                'graph_density': []
            }
        }
    
    def initialize_agent(self):
        """Initialize agent with enhanced scalable memory."""
        print("\n=== Initializing Agent with Scalable Features ===")
        
        self.agent = MainAgent(self.config)
        
        # Replace with enhanced memory
        self.agent.l2_memory = L2EnhancedScalableMemory(
            dim=self.config.embedding.dimension,
            config=self.config,
            use_scalable_graph=True
        )
        
        # Initialize components
        self.agent.l2_memory.initialize()
        if hasattr(self.agent, 'l3_graph') and self.agent.l3_graph:
            self.agent.l3_graph.initialize()
        
        # Verify clean state
        initial_stats = self.agent.get_stats()
        memory_stats = initial_stats.get('memory_stats', {})
        
        print(f"Agent initialized:")
        print(f"  Initial episodes: {memory_stats.get('total_episodes', 0)}")
        print(f"  Initial nodes: 0")
        print(f"  Scalable graph: Enabled")
        print(f"  FAISS index: Ready")
        print(f"  Conflict detection: Enabled")
    
    def load_dataset_documents(self, dataset_name: str, max_docs: int = None) -> List[str]:
        """Load documents from a specific dataset."""
        dataset_dir = self.mega_dataset_path / dataset_name
        
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return []
        
        try:
            # Load dataset from Arrow format
            dataset = Dataset.load_from_disk(str(dataset_dir))
            
            documents = []
            dataset_info = self.available_datasets.get(dataset_name, {})
            dataset_type = dataset_info.get('type', 'unknown')
            
            print(f"\nLoading {dataset_name} ({dataset_type} type) with {len(dataset)} samples...")
            
            # Extract text based on dataset type and structure
            for i, item in enumerate(dataset):
                if max_docs and len(documents) >= max_docs:
                    break
                
                # SQuAD format
                if 'squad' in dataset_name:
                    if 'context' in item and item['context']:
                        documents.append(item['context'])
                    if 'question' in item and item['question']:
                        documents.append(item['question'])
                
                # MS MARCO format
                elif 'ms_marco' in dataset_name:
                    if 'query' in item and item['query']:
                        documents.append(item['query'])
                    if 'passages' in item and 'passage_text' in item['passages']:
                        for passage in item['passages']['passage_text'][:3]:  # Limit passages
                            if passage:
                                documents.append(passage)
                
                # CoQA conversational format
                elif 'coqa' in dataset_name:
                    if 'story' in item and item['story']:
                        documents.append(item['story'])
                    if 'questions' in item and item['questions']:
                        documents.append(item['questions'])
                    if 'answers' in item and 'input_text' in item['answers']:
                        documents.append(item['answers']['input_text'])
                
                # HotpotQA multi-hop format
                elif 'hotpot' in dataset_name:
                    if 'context' in item and 'sentences' in item['context']:
                        # Combine context sentences
                        for sentences in item['context']['sentences'][:2]:  # Limit contexts
                            if sentences:
                                context = ' '.join(sentences)
                                documents.append(context)
                    if 'question' in item and item['question']:
                        documents.append(item['question'])
                
                # DROP numerical reasoning format
                elif 'drop' in dataset_name:
                    if 'passage' in item and item['passage']:
                        documents.append(item['passage'])
                    if 'question' in item and item['question']:
                        documents.append(item['question'])
                
                # BoolQ boolean format
                elif 'boolq' in dataset_name:
                    if 'passage' in item and item['passage']:
                        documents.append(item['passage'])
                    if 'question' in item and item['question']:
                        documents.append(item['question'])
                
                # CommonsenseQA format
                elif 'commonsense' in dataset_name:
                    if 'question' in item and item['question']:
                        # Combine question with choices for context
                        question = item['question']
                        if 'choices' in item and 'text' in item['choices']:
                            choices = ' '.join([f"({i+1}) {c}" for i, c in enumerate(item['choices']['text'])])
                            documents.append(f"{question} Choices: {choices}")
                        else:
                            documents.append(question)
            
            print(f"Extracted {len(documents)} documents from {dataset_name}")
            return documents
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def process_dataset(self, dataset_name: str, max_docs: int = None) -> Dict[str, Any]:
        """Process a single dataset and return metrics."""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")
        
        # Load documents
        documents = self.load_dataset_documents(dataset_name, max_docs)
        if not documents:
            return {'success': False, 'error': 'No documents loaded'}
        
        # Track dataset-specific metrics
        dataset_results = {
            'dataset': dataset_name,
            'type': self.available_datasets[dataset_name]['type'],
            'total_documents': len(documents),
            'episodes_before': len(self.agent.l2_memory.episodes),
            'start_time': time.time(),
            'successful': 0,
            'failed': 0,
            'integrations': 0,
            'conflicts': 0,
            'processing_times': []
        }
        
        # Process documents
        for i, doc in enumerate(documents):
            # Skip very short documents
            if not doc or len(doc.strip()) < 20:
                continue
            
            # Truncate very long documents
            if len(doc) > 2000:
                doc = doc[:2000] + "..."
            
            doc_start = time.time()
            
            try:
                # Add episode with graph update
                result = self.agent.add_episode_with_graph_update(doc)
                
                if result.get('success', False):
                    dataset_results['successful'] += 1
                    doc_time = time.time() - doc_start
                    dataset_results['processing_times'].append(doc_time)
                    
                    # Check if it was integrated
                    if result.get('episode_idx', -1) < len(self.agent.l2_memory.episodes) - 1:
                        dataset_results['integrations'] += 1
                    
                    # Check for conflicts
                    if hasattr(self.agent.l2_memory, 'recent_conflicts'):
                        dataset_results['conflicts'] = len(self.agent.l2_memory.recent_conflicts)
                else:
                    dataset_results['failed'] += 1
                
                # Progress report every 50 documents
                if (i + 1) % 50 == 0:
                    self._report_progress(dataset_name, i + 1, len(documents))
                    
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                dataset_results['failed'] += 1
        
        # Calculate final metrics
        dataset_results['end_time'] = time.time()
        dataset_results['total_time'] = dataset_results['end_time'] - dataset_results['start_time']
        dataset_results['episodes_after'] = len(self.agent.l2_memory.episodes)
        dataset_results['new_episodes'] = dataset_results['episodes_after'] - dataset_results['episodes_before']
        
        if dataset_results['successful'] > 0:
            dataset_results['avg_time_per_doc'] = np.mean(dataset_results['processing_times'])
            dataset_results['integration_rate'] = dataset_results['integrations'] / dataset_results['successful']
        
        # Get graph stats
        if hasattr(self.agent.l2_memory, 'get_graph_stats'):
            dataset_results['graph_stats'] = self.agent.l2_memory.get_graph_stats()
        
        # Print summary
        print(f"\n{dataset_name} Summary:")
        print(f"  Documents processed: {dataset_results['successful']}/{len(documents)}")
        print(f"  New episodes: {dataset_results['new_episodes']}")
        print(f"  Integration rate: {dataset_results.get('integration_rate', 0)*100:.1f}%")
        print(f"  Processing time: {dataset_results['total_time']:.2f}s")
        
        return dataset_results
    
    def _report_progress(self, dataset_name: str, processed: int, total: int):
        """Report processing progress and update timeline."""
        # Get current stats
        stats = self.agent.get_stats()
        memory_stats = stats.get('memory_stats', {})
        
        if hasattr(self.agent.l2_memory, 'get_graph_stats'):
            graph_stats = self.agent.l2_memory.get_graph_stats()
        else:
            graph_stats = {}
        
        print(f"\n[{dataset_name}] Progress: {processed}/{total}")
        print(f"  Episodes: {memory_stats.get('total_episodes', 0)}")
        print(f"  Graph: {graph_stats.get('nodes', 0)} nodes, {graph_stats.get('edges', 0)} edges")
        
        # Update timeline
        self.results['timeline']['documents'].append(self.results['global_metrics']['total_documents'] + processed)
        self.results['timeline']['episodes'].append(memory_stats.get('total_episodes', 0))
        self.results['timeline']['nodes'].append(graph_stats.get('nodes', 0))
        self.results['timeline']['edges'].append(graph_stats.get('edges', 0))
        self.results['timeline']['timestamps'].append(time.time())
        self.results['timeline']['datasets'].append(dataset_name)
        
        # Calculate performance metrics
        if len(self.results['timeline']['timestamps']) > 1:
            time_diff = self.results['timeline']['timestamps'][-1] - self.results['timeline']['timestamps'][-2]
            doc_diff = self.results['timeline']['documents'][-1] - self.results['timeline']['documents'][-2]
            if time_diff > 0:
                self.results['performance']['doc_per_second'].append(doc_diff / time_diff)
        
        # Graph density
        nodes = graph_stats.get('nodes', 0)
        edges = graph_stats.get('edges', 0)
        if nodes > 1:
            max_edges = nodes * (nodes - 1) / 2
            density = edges / max_edges if max_edges > 0 else 0
            self.results['performance']['graph_density'].append(density)
    
    def run_mega_experiment(self):
        """Run the complete mega dataset experiment."""
        print("\n=== MEGA DATASET EXPERIMENT ===")
        print(f"Start time: {datetime.now()}")
        print(f"Target: 1000+ episodes from {len(self.available_datasets)} datasets")
        
        # Initialize agent
        self.initialize_agent()
        
        # Process datasets in priority order
        sorted_datasets = sorted(
            self.available_datasets.items(),
            key=lambda x: x[1]['priority']
        )
        
        experiment_start = time.time()
        
        for dataset_name, info in sorted_datasets:
            # Process dataset
            dataset_results = self.process_dataset(dataset_name)
            
            if dataset_results.get('successful', 0) > 0:
                # Store results
                self.results['datasets_processed'][dataset_name] = dataset_results
                
                # Update global metrics
                self.results['global_metrics']['total_documents'] += dataset_results['successful']
                self.results['global_metrics']['total_episodes'] = dataset_results['episodes_after']
                
                # Check if we've reached 1000+ episodes
                if self.results['global_metrics']['total_episodes'] >= 1000:
                    print(f"\n🎯 Reached {self.results['global_metrics']['total_episodes']} episodes!")
                    break
        
        # Final statistics
        experiment_end = time.time()
        self.results['global_metrics']['processing_time'] = experiment_end - experiment_start
        
        # Get final graph stats
        if hasattr(self.agent.l2_memory, 'get_graph_stats'):
            final_graph = self.agent.l2_memory.get_graph_stats()
            self.results['global_metrics']['total_nodes'] = final_graph.get('nodes', 0)
            self.results['global_metrics']['total_edges'] = final_graph.get('edges', 0)
        
        # Calculate overall integration rate
        if self.results['global_metrics']['total_documents'] > 0:
            self.results['global_metrics']['integration_rate'] = (
                1 - self.results['global_metrics']['total_episodes'] / 
                self.results['global_metrics']['total_documents']
            )
        
        print("\n=== EXPERIMENT COMPLETE ===")
        print(f"Total time: {self.results['global_metrics']['processing_time']:.2f}s")
        print(f"Documents processed: {self.results['global_metrics']['total_documents']}")
        print(f"Episodes created: {self.results['global_metrics']['total_episodes']}")
        print(f"Integration rate: {self.results['global_metrics']['integration_rate']*100:.1f}%")
        print(f"Final graph: {self.results['global_metrics']['total_nodes']} nodes, "
              f"{self.results['global_metrics']['total_edges']} edges")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations of the experiment."""
        print("\n=== Generating Visualizations ===")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Growth over time plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episodes growth
        ax1 = axes[0, 0]
        ax1.plot(self.results['timeline']['documents'], 
                self.results['timeline']['episodes'], 
                'b-', linewidth=2)
        ax1.set_xlabel('Documents Processed')
        ax1.set_ylabel('Total Episodes')
        ax1.set_title('Episode Growth')
        
        # Add dataset markers
        dataset_changes = []
        current_dataset = None
        for i, dataset in enumerate(self.results['timeline']['datasets']):
            if dataset != current_dataset:
                dataset_changes.append(i)
                current_dataset = dataset
                ax1.axvline(x=self.results['timeline']['documents'][i], 
                          color='gray', linestyle='--', alpha=0.5)
                ax1.text(self.results['timeline']['documents'][i], 
                        ax1.get_ylim()[1] * 0.95, 
                        dataset.split('_')[0], 
                        rotation=90, verticalalignment='top')
        
        # Graph growth
        ax2 = axes[0, 1]
        ax2.plot(self.results['timeline']['documents'], 
                self.results['timeline']['nodes'], 
                'g-', linewidth=2, label='Nodes')
        ax2.plot(self.results['timeline']['documents'], 
                self.results['timeline']['edges'], 
                'r-', linewidth=2, label='Edges')
        ax2.set_xlabel('Documents Processed')
        ax2.set_ylabel('Count')
        ax2.set_title('Graph Growth')
        ax2.legend()
        ax2.set_yscale('log')
        
        # Processing speed
        ax3 = axes[1, 0]
        if self.results['performance']['doc_per_second']:
            ax3.plot(self.results['performance']['doc_per_second'], 
                    'o-', linewidth=1, markersize=4)
            ax3.set_xlabel('Checkpoint')
            ax3.set_ylabel('Documents per Second')
            ax3.set_title('Processing Speed')
        
        # Graph density
        ax4 = axes[1, 1]
        if self.results['performance']['graph_density']:
            ax4.plot(self.results['timeline']['documents'][:len(self.results['performance']['graph_density'])],
                    self.results['performance']['graph_density'], 
                    'm-', linewidth=2)
            ax4.set_xlabel('Documents Processed')
            ax4.set_ylabel('Graph Density')
            ax4.set_title('Graph Density Evolution')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'mega_experiment_growth.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Dataset comparison
        if len(self.results['datasets_processed']) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            datasets = []
            doc_counts = []
            episode_counts = []
            integration_rates = []
            
            for name, results in self.results['datasets_processed'].items():
                datasets.append(name.replace('_', '\n'))
                doc_counts.append(results['successful'])
                episode_counts.append(results['new_episodes'])
                integration_rates.append(results.get('integration_rate', 0) * 100)
            
            x = np.arange(len(datasets))
            width = 0.25
            
            ax.bar(x - width, doc_counts, width, label='Documents', alpha=0.8)
            ax.bar(x, episode_counts, width, label='New Episodes', alpha=0.8)
            ax.bar(x + width, integration_rates, width, label='Integration %', alpha=0.8)
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Count / Percentage')
            ax.set_title('Dataset Processing Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.experiment_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Visualizations saved!")
    
    def save_results(self):
        """Save all experiment results and state."""
        print("\n=== Saving Results ===")
        
        # Save detailed results
        results_file = self.experiment_dir / f'mega_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        # Save summary CSV
        summary_data = []
        for name, results in self.results['datasets_processed'].items():
            summary_data.append({
                'Dataset': name,
                'Type': results['type'],
                'Documents': results['successful'],
                'New Episodes': results['new_episodes'],
                'Integration Rate': f"{results.get('integration_rate', 0)*100:.1f}%",
                'Avg Time (s)': f"{results.get('avg_time_per_doc', 0):.3f}",
                'Total Time (s)': f"{results['total_time']:.1f}"
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.experiment_dir / 'mega_experiment_summary.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary saved to: {csv_file}")
        
        # Save agent state
        print("\nSaving agent state...")
        if self.agent.save_state():
            print("✓ Agent state saved successfully")
            print(f"  Episodes: {self.data_dir}/episodes.json")
            print(f"  Index: {self.data_dir}/index.faiss")
            print(f"  Graph: {self.data_dir}/graph_pyg.pt")
        else:
            print("✗ Failed to save agent state")
        
        # Save monitoring data
        if self.monitor:
            monitor_file = self.experiment_dir / f'monitor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            self.monitor.export_metrics(monitor_file)
            print(f"Monitoring data saved to: {monitor_file}")


def main():
    """Run the mega dataset experiment."""
    experiment = MegaDatasetExperiment()
    
    try:
        # Run the experiment
        experiment.run_mega_experiment()
        
        # Generate visualizations
        experiment.generate_visualizations()
        
        # Save results
        experiment.save_results()
        
        print(f"\n✅ Mega experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial results
        try:
            experiment.save_results()
        except:
            print("Failed to save partial results")


if __name__ == "__main__":
    main()