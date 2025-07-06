#!/usr/bin/env python3
"""
Continue to 1000 Episodes
=========================

Continue processing datasets until we reach 1000+ episodes.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config


def main():
    """Continue the experiment to reach 1000 episodes."""
    print("\n=== CONTINUING TO 1000 EPISODES ===")
    
    # Setup
    config = get_config()
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "data"
    
    # Update config
    config.paths.data_dir = str(data_dir)
    config.memory.index_file = str(data_dir / "index.faiss")
    config.reasoning.graph_file = str(data_dir / "graph_pyg.pt")
    
    # Load progress
    progress_file = experiment_dir / "progress.json"
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    print(f"Current status:")
    print(f"  Episodes: {progress['total_episodes']}")
    print(f"  Documents: {progress['total_documents']}")
    print(f"  Needed: {1000 - progress['total_episodes']} more episodes")
    
    # Initialize agent and load state
    agent = MainAgent(config)
    agent.l2_memory = L2EnhancedScalableMemory(
        dim=config.embedding.dimension,
        config=config,
        use_scalable_graph=True
    )
    
    print("\nLoading agent state...")
    if agent.load_state():
        stats = agent.get_stats()
        memory_stats = stats.get('memory_stats', {})
        print(f"✓ Loaded {memory_stats.get('total_episodes', 0)} episodes")
    else:
        print("✗ Failed to load state")
        return
    
    # Available datasets
    mega_path = Path("/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/data/mega_huggingface_datasets")
    remaining_datasets = [
        {'name': 'hotpot_qa_60', 'batch_size': 30, 'samples': 60},
        {'name': 'drop_50', 'batch_size': 30, 'samples': 50},
        {'name': 'boolq_50', 'batch_size': 30, 'samples': 50},
        {'name': 'commonsense_qa_20', 'batch_size': 20, 'samples': 20}
    ]
    
    # Process remaining datasets
    for dataset_config in remaining_datasets:
        if progress['total_episodes'] >= 1000:
            break
            
        dataset_name = dataset_config['name']
        if dataset_name in progress['datasets_completed']:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")
        
        dataset_dir = mega_path / dataset_name
        if not dataset_dir.exists():
            print(f"Dataset not found: {dataset_dir}")
            continue
        
        try:
            # Load dataset
            dataset = Dataset.load_from_disk(str(dataset_dir))
            print(f"Loaded {len(dataset)} samples")
            
            # Process all at once for simpler datasets
            documents = []
            
            # Extract based on common keys
            for item in dataset:
                # Try different field names
                for field in ['passage', 'context', 'question', 'query', 'text', 'statement']:
                    if field in item and item[field] and isinstance(item[field], str):
                        documents.append(item[field])
                        break
            
            print(f"Extracted {len(documents)} documents")
            
            # Process documents
            successful = 0
            start_time = time.time()
            
            for i, doc in enumerate(documents):
                if not doc or len(doc.strip()) < 20:
                    continue
                
                if len(doc) > 2000:
                    doc = doc[:2000] + "..."
                
                try:
                    result = agent.add_episode_with_graph_update(doc)
                    if result.get('success', False):
                        successful += 1
                    
                    # Progress report
                    if (i + 1) % 20 == 0:
                        current_episodes = len(agent.l2_memory.episodes)
                        print(f"  Progress: {i+1}/{len(documents)} docs, {current_episodes} total episodes")
                        
                        if current_episodes >= 1000:
                            print(f"\n🎯 Reached {current_episodes} episodes!")
                            progress['total_episodes'] = current_episodes
                            progress['total_documents'] += successful
                            progress['datasets_completed'].append(dataset_name)
                            
                            # Save progress
                            with open(progress_file, 'w') as f:
                                json.dump(progress, f, indent=2)
                            
                            # Save agent state
                            print("\nSaving final state...")
                            if agent.save_state():
                                print("✓ State saved")
                                
                                # Get final stats
                                if hasattr(agent.l2_memory, 'get_graph_stats'):
                                    graph_stats = agent.l2_memory.get_graph_stats()
                                    print(f"\nFinal graph: {graph_stats.get('nodes', 0)} nodes, {graph_stats.get('edges', 0)} edges")
                            
                            return
                        
                except Exception as e:
                    print(f"Error processing document: {e}")
            
            # Dataset completed
            elapsed = time.time() - start_time
            current_episodes = len(agent.l2_memory.episodes)
            
            print(f"\n{dataset_name} completed:")
            print(f"  Documents: {successful}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Total episodes: {current_episodes}")
            
            progress['total_documents'] += successful
            progress['total_episodes'] = current_episodes
            progress['datasets_completed'].append(dataset_name)
            
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            # Save checkpoint
            if agent.save_state():
                if hasattr(agent.l2_memory, 'get_graph_stats'):
                    stats = agent.l2_memory.get_graph_stats()
                    print(f"  Checkpoint: {stats.get('nodes', 0)} nodes, {stats.get('edges', 0)} edges")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Total episodes: {progress['total_episodes']}")
    print(f"Total documents: {progress['total_documents']}")
    print(f"Datasets processed: {len(progress['datasets_completed'])}")
    
    # Generate final report
    if hasattr(agent.l2_memory, 'get_graph_stats'):
        graph_stats = agent.l2_memory.get_graph_stats()
        
        summary = {
            'experiment': 'Mega Dataset 1000+ Episodes',
            'completed': datetime.now().isoformat(),
            'total_episodes': progress['total_episodes'],
            'total_documents': progress['total_documents'],
            'graph_nodes': graph_stats.get('nodes', 0),
            'graph_edges': graph_stats.get('edges', 0),
            'integration_rate': 1 - (progress['total_episodes'] / max(1, progress['total_documents'])),
            'datasets': progress['datasets_completed']
        }
        
        summary_file = experiment_dir / f'final_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFinal statistics:")
        print(f"  Integration rate: {summary['integration_rate']*100:.1f}%")
        print(f"  Graph: {summary['graph_nodes']} nodes, {summary['graph_edges']} edges")
        print(f"  Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()