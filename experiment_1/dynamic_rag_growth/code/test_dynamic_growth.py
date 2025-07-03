#!/usr/bin/env python3
"""
Test Dynamic Growth of InsightSpike-AI RAG System
Tests the ability to dynamically grow graph and JSON files using CLI and data folder
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import torch
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.insightspike.core.agents.agent_builder import AgentBuilder

class DynamicGrowthTester:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create test data directory
        self.test_data_dir = self.data_dir / "test_growth"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy initial clean data
        self.setup_clean_data()
        
    def setup_clean_data(self):
        """Copy clean backup data to test directory"""
        clean_backup = Path("data/clean_backup")
        if clean_backup.exists():
            # Copy essential files
            for file in ["episodes_clean.json", "graph_pyg_clean.pt"]:
                src = clean_backup / file
                if src.exists():
                    dst = self.test_data_dir / file.replace("_clean", "")
                    shutil.copy2(src, dst)
                    print(f"Copied {src} to {dst}")
    
    def get_file_sizes(self):
        """Get current file sizes"""
        sizes = {}
        episodes_path = self.test_data_dir / "episodes.json"
        graph_path = self.test_data_dir / "graph_pyg.pt"
        
        if episodes_path.exists():
            sizes['episodes_size'] = episodes_path.stat().st_size
            with open(episodes_path, 'r') as f:
                data = json.load(f)
                sizes['episodes_count'] = len(data)
        
        if graph_path.exists():
            sizes['graph_size'] = graph_path.stat().st_size
            graph_data = torch.load(graph_path)
            if hasattr(graph_data, 'num_nodes'):
                sizes['graph_nodes'] = graph_data.num_nodes
            elif 'x' in graph_data:
                sizes['graph_nodes'] = graph_data['x'].shape[0]
        
        return sizes
    
    def test_cli_growth(self, num_additions: int = 10):
        """Test growing data through CLI commands"""
        print(f"\n=== Testing CLI-based growth with {num_additions} additions ===")
        
        initial_sizes = self.get_file_sizes()
        print(f"Initial state: {initial_sizes}")
        
        growth_results = []
        
        # Change to test data directory
        original_dir = os.getcwd()
        os.chdir(self.test_data_dir)
        
        try:
            # Initialize InsightSpike with test data
            agent_builder = AgentBuilder()
            agent = agent_builder.build_main_agent()
            
            # Add data progressively
            for i in range(num_additions):
                # Create test query
                test_query = f"Test knowledge item {i}: The capital of TestCountry{i} is TestCity{i}."
                
                # Process through InsightSpike
                start_time = time.time()
                response = agent.process_input(test_query)
                process_time = time.time() - start_time
                
                # Get updated sizes
                current_sizes = self.get_file_sizes()
                
                growth_results.append({
                    'iteration': i + 1,
                    'query': test_query,
                    'process_time': process_time,
                    'sizes': current_sizes,
                    'growth': {
                        'episodes': current_sizes.get('episodes_count', 0) - initial_sizes.get('episodes_count', 0),
                        'graph_nodes': current_sizes.get('graph_nodes', 0) - initial_sizes.get('graph_nodes', 0)
                    }
                })
                
                print(f"Iteration {i+1}: Episodes={current_sizes.get('episodes_count', 0)}, "
                      f"Nodes={current_sizes.get('graph_nodes', 0)}, Time={process_time:.3f}s")
        
        finally:
            os.chdir(original_dir)
        
        return growth_results
    
    def test_batch_growth(self, dataset_path: str = None):
        """Test growth with batch data loading"""
        print(f"\n=== Testing batch data growth ===")
        
        if not dataset_path:
            # Use existing HuggingFace datasets
            dataset_path = Path("experiments/gedig_embedding_evaluation/data/huggingface_datasets")
        
        initial_sizes = self.get_file_sizes()
        batch_results = {
            'initial': initial_sizes,
            'datasets_processed': [],
            'final': None,
            'total_growth': {}
        }
        
        # Process available datasets
        if Path(dataset_path).exists():
            for dataset_dir in Path(dataset_path).iterdir():
                if dataset_dir.is_dir() and dataset_dir.name != "__pycache__":
                    print(f"\nProcessing dataset: {dataset_dir.name}")
                    
                    # Load dataset (simplified for now)
                    arrow_files = list(dataset_dir.glob("*.arrow"))
                    if arrow_files:
                        # Record processing
                        batch_results['datasets_processed'].append({
                            'name': dataset_dir.name,
                            'files': len(arrow_files)
                        })
        
        # Get final sizes
        final_sizes = self.get_file_sizes()
        batch_results['final'] = final_sizes
        batch_results['total_growth'] = {
            'episodes': final_sizes.get('episodes_count', 0) - initial_sizes.get('episodes_count', 0),
            'graph_nodes': final_sizes.get('graph_nodes', 0) - initial_sizes.get('graph_nodes', 0),
            'size_increase_bytes': {
                'episodes': final_sizes.get('episodes_size', 0) - initial_sizes.get('episodes_size', 0),
                'graph': final_sizes.get('graph_size', 0) - initial_sizes.get('graph_size', 0)
            }
        }
        
        return batch_results
    
    def calculate_compression_ratio(self, results):
        """Calculate compression ratio compared to raw text storage"""
        # Estimate raw text size (assuming average 100 chars per entry)
        raw_text_size = results.get('episodes_count', 0) * 100
        
        # Actual storage size
        actual_size = results.get('episodes_size', 0) + results.get('graph_size', 0)
        
        if actual_size > 0:
            compression_ratio = raw_text_size / actual_size
        else:
            compression_ratio = 0
        
        return {
            'estimated_raw_size': raw_text_size,
            'actual_size': actual_size,
            'compression_ratio': compression_ratio,
            'space_saved_percent': (1 - actual_size/raw_text_size) * 100 if raw_text_size > 0 else 0
        }
    
    def save_results(self, cli_results, batch_results):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'cli_growth_test': cli_results,
            'batch_growth_test': batch_results,
            'compression_analysis': self.calculate_compression_ratio(
                batch_results.get('final', cli_results[-1]['sizes'] if cli_results else {})
            )
        }
        
        # Save JSON results
        results_file = self.results_dir / f"dynamic_growth_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return results


def main():
    """Run dynamic growth experiments"""
    print("Starting InsightSpike-AI Dynamic Growth Experiment")
    
    # Initialize tester
    tester = DynamicGrowthTester("experiment_1/dynamic_rag_growth")
    
    # Run CLI growth test
    cli_results = tester.test_cli_growth(num_additions=20)
    
    # Run batch growth test
    batch_results = tester.test_batch_growth()
    
    # Save and analyze results
    final_results = tester.save_results(cli_results, batch_results)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"CLI Growth: {len(cli_results)} iterations")
    if cli_results:
        print(f"  - Episodes grew by: {cli_results[-1]['growth']['episodes']}")
        print(f"  - Graph nodes grew by: {cli_results[-1]['growth']['graph_nodes']}")
    
    print(f"\nBatch Processing: {len(batch_results.get('datasets_processed', []))} datasets")
    print(f"  - Total growth: {batch_results.get('total_growth', {})}")
    
    compression = final_results.get('compression_analysis', {})
    print(f"\nCompression Analysis:")
    print(f"  - Compression ratio: {compression.get('compression_ratio', 0):.2f}x")
    print(f"  - Space saved: {compression.get('space_saved_percent', 0):.1f}%")


if __name__ == "__main__":
    main()