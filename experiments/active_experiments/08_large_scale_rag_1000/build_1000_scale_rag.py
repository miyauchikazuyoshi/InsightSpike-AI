#!/usr/bin/env python3
"""
Experiment 8: Build 1000-Scale Dynamic RAG
==========================================

Build a comprehensive knowledge base with 1000+ documents and compare with standard RAG.
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

from datasets import Dataset
from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config


class Experiment8Builder:
    """Build and test 1000-scale dynamic RAG."""
    
    def __init__(self):
        self.config = get_config()
        self.agent = None
        
        # Use experiment-specific directory
        self.experiment_dir = Path(__file__).parent
        self.data_dir = self.experiment_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Configure paths
        self.config.paths.data_dir = str(self.data_dir)
        self.config.memory.index_file = str(self.data_dir / "index.faiss")
        self.config.reasoning.graph_file = str(self.data_dir / "graph_pyg.pt")
        
        # Adjust integration thresholds to prevent over-integration
        self.config.memory.episode_integration_similarity_threshold = 0.95  # Higher threshold
        self.config.memory.episode_integration_content_threshold = 0.8     # Higher threshold
        
        # Dataset paths
        self.huggingface_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data")
        
        # Track progress
        self.build_log = {
            'start_time': datetime.now().isoformat(),
            'datasets': [],
            'total_documents': 0,
            'total_episodes': 0,
            'integration_stats': {},
            'build_time': 0,
            'checkpoints': []
        }
    
    def initialize_agent(self):
        """Initialize agent with proper configuration."""
        print("\n=== Initializing Agent ===")
        print(f"Data directory: {self.data_dir}")
        print(f"Integration thresholds: similarity={self.config.memory.episode_integration_similarity_threshold}, "
              f"content={self.config.memory.episode_integration_content_threshold}")
        
        self.agent = MainAgent(self.config)
        
        # Use enhanced scalable memory
        self.agent.l2_memory = L2EnhancedScalableMemory(
            dim=self.config.embedding.dimension,
            config=self.config,
            use_scalable_graph=True
        )
        
        # Initialize fresh (no loading of existing state)
        self.agent.l2_memory.initialize()
        if hasattr(self.agent, 'l3_graph') and self.agent.l3_graph:
            self.agent.l3_graph.initialize()
        
        print("✓ Agent initialized with clean state")
    
    def load_squad_dataset(self, dataset_name: str, max_samples: int = None) -> List[Dict[str, str]]:
        """Load SQuAD dataset and return Q&A pairs."""
        dataset_path = self.huggingface_path / "mega_huggingface_datasets" / dataset_name
        if not dataset_path.exists():
            dataset_path = self.huggingface_path / "large_huggingface_datasets" / dataset_name
        
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            return []
        
        print(f"\nLoading {dataset_name}...")
        dataset = Dataset.load_from_disk(str(dataset_path))
        
        qa_pairs = []
        samples = min(len(dataset), max_samples) if max_samples else len(dataset)
        
        for i in range(samples):
            item = dataset[i]
            
            # Extract context and questions
            context = item.get('context', '')
            question = item.get('question', '')
            
            # For SQuAD, answers are in a specific format
            answers = item.get('answers', {})
            if isinstance(answers, dict) and 'text' in answers:
                answer_texts = answers['text']
                answer = answer_texts[0] if answer_texts else ""
            else:
                answer = str(answers) if answers else ""
            
            if context and question:
                qa_pairs.append({
                    'context': context,
                    'question': question,
                    'answer': answer,
                    'dataset': dataset_name,
                    'index': i
                })
        
        print(f"Loaded {len(qa_pairs)} Q&A pairs from {dataset_name}")
        return qa_pairs
    
    def build_knowledge_base(self, target_documents: int = 1000):
        """Build knowledge base with target number of documents."""
        print(f"\n=== Building Knowledge Base (Target: {target_documents} documents) ===")
        
        start_time = time.time()
        
        # Available datasets
        datasets = [
            ('squad_200', 200),
            ('squad_300', 300),
            ('squad_100', 100),
            ('ms_marco_150', 150),
            ('ms_marco_50', 50),
            ('coqa_80', 80),
            ('hotpot_qa_60', 60),
            ('drop_50', 50)
        ]
        
        all_qa_pairs = []
        documents_processed = 0
        
        # Load datasets until we reach target
        for dataset_name, max_samples in datasets:
            if documents_processed >= target_documents:
                break
            
            remaining = target_documents - documents_processed
            samples_to_load = min(max_samples, remaining)
            
            qa_pairs = self.load_squad_dataset(dataset_name, samples_to_load)
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                documents_processed += len(qa_pairs)
                
                self.build_log['datasets'].append({
                    'name': dataset_name,
                    'samples': len(qa_pairs)
                })
        
        print(f"\nTotal Q&A pairs loaded: {len(all_qa_pairs)}")
        
        # Process documents into episodes
        print("\n=== Processing Documents ===")
        
        # We'll add both contexts and Q&A pairs as episodes
        episodes_before = 0
        checkpoint_interval = 100
        
        for i, qa_pair in enumerate(all_qa_pairs):
            # Add context as episode
            context_result = self.agent.add_episode_with_graph_update(
                qa_pair['context'],
                c_value=0.8  # Higher importance for contexts
            )
            
            # Add question as episode (linked to context)
            question_text = f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
            question_result = self.agent.add_episode_with_graph_update(
                question_text,
                c_value=0.6  # Lower importance for questions
            )
            
            # Progress report
            if (i + 1) % checkpoint_interval == 0:
                current_episodes = len(self.agent.l2_memory.episodes)
                integration_rate = 1 - (current_episodes - episodes_before) / (2 * checkpoint_interval)
                
                print(f"\nProgress: {i+1}/{len(all_qa_pairs)} documents")
                print(f"  Episodes: {current_episodes}")
                print(f"  Integration rate: {integration_rate:.1%}")
                
                # Save checkpoint
                self.save_checkpoint(i + 1, current_episodes)
                episodes_before = current_episodes
        
        # Final statistics
        build_time = time.time() - start_time
        final_episodes = len(self.agent.l2_memory.episodes)
        
        self.build_log['total_documents'] = len(all_qa_pairs) * 2  # contexts + questions
        self.build_log['total_episodes'] = final_episodes
        self.build_log['build_time'] = build_time
        self.build_log['integration_rate'] = 1 - final_episodes / (len(all_qa_pairs) * 2)
        
        print(f"\n=== Build Complete ===")
        print(f"Documents processed: {self.build_log['total_documents']}")
        print(f"Episodes created: {final_episodes}")
        print(f"Integration rate: {self.build_log['integration_rate']:.1%}")
        print(f"Build time: {build_time:.1f}s")
        
        # Save the knowledge base
        print("\nSaving knowledge base...")
        if self.agent.save_state():
            print("✓ Knowledge base saved successfully")
        else:
            print("✗ Failed to save knowledge base")
        
        # Save Q&A pairs for testing
        qa_file = self.data_dir / "qa_pairs.json"
        with open(qa_file, 'w') as f:
            json.dump(all_qa_pairs, f, indent=2)
        print(f"✓ Q&A pairs saved to {qa_file}")
        
        return all_qa_pairs
    
    def save_checkpoint(self, docs_processed: int, episodes: int):
        """Save checkpoint during build."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'documents_processed': docs_processed,
            'episodes': episodes,
            'integration_rate': 1 - episodes / (docs_processed * 2)
        }
        self.build_log['checkpoints'].append(checkpoint)
        
        # Save build log
        log_file = self.experiment_dir / "build_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.build_log, f, indent=2)
    
    def verify_build(self):
        """Verify the built knowledge base."""
        print("\n=== Verifying Build ===")
        
        # Check saved files
        files_to_check = [
            (self.data_dir / "episodes.json", "Episodes"),
            (self.data_dir / "index.faiss", "FAISS index"),
            (self.data_dir / "graph_pyg.pt", "Graph"),
            (self.data_dir / "qa_pairs.json", "Q&A pairs")
        ]
        
        all_good = True
        for file_path, name in files_to_check:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"✓ {name}: {size_mb:.2f} MB")
            else:
                print(f"✗ {name}: NOT FOUND")
                all_good = False
        
        # Load and check episodes
        episodes_file = self.data_dir / "episodes.json"
        if episodes_file.exists():
            with open(episodes_file, 'r') as f:
                episodes = json.load(f)
            
            print(f"\nEpisode Statistics:")
            print(f"  Total episodes: {len(episodes)}")
            
            # Check for over-integration
            integrated_count = 0
            max_integration = 0
            
            for ep in episodes[:100]:  # Check first 100
                text = ep.get('text', '')
                if '|' in text:
                    parts = text.split('|')
                    integrated_count += 1
                    max_integration = max(max_integration, len(parts))
            
            print(f"  Integrated episodes (first 100): {integrated_count}")
            print(f"  Max integration depth: {max_integration}")
            
            if integrated_count > 50:
                print("  ⚠️  Warning: High integration rate detected")
        
        return all_good


def main():
    """Run the experiment."""
    print("=== EXPERIMENT 8: 1000-SCALE DYNAMIC RAG ===")
    print(f"Start time: {datetime.now()}")
    
    builder = Experiment8Builder()
    
    try:
        # Initialize agent
        builder.initialize_agent()
        
        # Build knowledge base
        qa_pairs = builder.build_knowledge_base(target_documents=1000)
        
        # Verify build
        if builder.verify_build():
            print("\n✅ Build verification passed!")
        else:
            print("\n⚠️  Build verification failed!")
        
        # Save final report
        report = {
            'experiment': 'Experiment 8: 1000-Scale Dynamic RAG',
            'completed': datetime.now().isoformat(),
            'build_log': builder.build_log,
            'verification': 'passed' if builder.verify_build() else 'failed'
        }
        
        report_file = builder.experiment_dir / f"build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Build complete! Report saved to {report_file}")
        print("\nNext step: Run test_rag_performance.py to compare with standard RAG")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()