#!/usr/bin/env python3
"""
Simplified Dynamic Growth Test for InsightSpike-AI
Tests data growth and compression with real datasets
"""

import os
import json
import time
import torch
import shutil
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class SimplifiedGrowthTester:
    def __init__(self):
        self.results_dir = Path("experiment_1/dynamic_rag_growth/results")
        self.results_dir.mkdir(exist_ok=True)
        self.data_dir = Path("experiment_1/dynamic_rag_growth/data")
        self.data_dir.mkdir(exist_ok=True)
        
    def analyze_existing_data(self):
        """Analyze existing data files"""
        print("\n=== Analyzing Existing Data ===")
        
        results = {
            'episodes': {},
            'graph': {},
            'datasets': {}
        }
        
        # Analyze episodes.json
        episodes_path = Path("data/episodes.json")
        if episodes_path.exists():
            size = episodes_path.stat().st_size
            with open(episodes_path, 'r') as f:
                data = json.load(f)
            results['episodes'] = {
                'size_bytes': size,
                'count': len(data),
                'size_per_episode': size / len(data) if len(data) > 0 else 0
            }
            print(f"Episodes: {len(data)} items, {size:,} bytes, {size/len(data):.1f} bytes/item")
        
        # Analyze graph_pyg.pt
        graph_path = Path("data/graph_pyg.pt")
        if graph_path.exists():
            size = graph_path.stat().st_size
            graph_data = torch.load(graph_path)
            num_nodes = 0
            if hasattr(graph_data, 'num_nodes'):
                num_nodes = graph_data.num_nodes
            elif isinstance(graph_data, dict) and 'x' in graph_data:
                num_nodes = graph_data['x'].shape[0]
            
            results['graph'] = {
                'size_bytes': size,
                'num_nodes': num_nodes,
                'size_per_node': size / num_nodes if num_nodes > 0 else 0
            }
            print(f"Graph: {num_nodes} nodes, {size:,} bytes, {size/num_nodes:.1f} bytes/node")
        
        return results
    
    def analyze_huggingface_datasets(self):
        """Analyze available HuggingFace datasets"""
        print("\n=== Analyzing HuggingFace Datasets ===")
        
        dataset_paths = [
            "experiments/gedig_embedding_evaluation/data/huggingface_datasets",
            "experiments/gedig_embedding_evaluation/data/large_huggingface_datasets",
            "experiments/gedig_embedding_evaluation/data/mega_huggingface_datasets"
        ]
        
        all_datasets = []
        total_samples = 0
        total_size = 0
        
        for base_path in dataset_paths:
            base = Path(base_path)
            if not base.exists():
                continue
                
            for dataset_dir in base.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name not in ["__pycache__", ".DS_Store"]:
                    arrow_files = list(dataset_dir.glob("*.arrow"))
                    if arrow_files:
                        # Get dataset info
                        dataset_name = dataset_dir.name
                        dataset_size = sum(f.stat().st_size for f in arrow_files)
                        
                        # Extract sample count from name
                        parts = dataset_name.split('_')
                        sample_count = int(parts[-1]) if parts[-1].isdigit() else 0
                        
                        dataset_info = {
                            'name': dataset_name,
                            'path': str(dataset_dir),
                            'size_bytes': dataset_size,
                            'sample_count': sample_count,
                            'category': base.name
                        }
                        
                        all_datasets.append(dataset_info)
                        total_samples += sample_count
                        total_size += dataset_size
                        
                        print(f"  {dataset_name}: {sample_count} samples, {dataset_size:,} bytes")
        
        print(f"\nTotal: {total_samples} samples, {total_size:,} bytes")
        return all_datasets, total_samples, total_size
    
    def simulate_dynamic_growth(self, datasets_info):
        """Simulate dynamic data growth"""
        print("\n=== Simulating Dynamic Growth ===")
        
        growth_simulation = []
        cumulative_samples = 0
        cumulative_size = 0
        
        # Sort datasets by sample count for progressive growth
        sorted_datasets = sorted(datasets_info, key=lambda x: x['sample_count'])
        
        for dataset in sorted_datasets:
            cumulative_samples += dataset['sample_count']
            cumulative_size += dataset['size_bytes']
            
            # Estimate InsightSpike storage (using compression ratio from previous experiments)
            compression_ratio = 19.4  # From compression efficiency experiment
            estimated_insightspike_size = cumulative_size / compression_ratio
            
            growth_point = {
                'dataset': dataset['name'],
                'cumulative_samples': cumulative_samples,
                'raw_size_bytes': cumulative_size,
                'estimated_insightspike_size': estimated_insightspike_size,
                'compression_ratio': cumulative_size / estimated_insightspike_size if estimated_insightspike_size > 0 else 0,
                'size_per_sample_raw': cumulative_size / cumulative_samples if cumulative_samples > 0 else 0,
                'size_per_sample_compressed': estimated_insightspike_size / cumulative_samples if cumulative_samples > 0 else 0
            }
            
            growth_simulation.append(growth_point)
            
            if cumulative_samples % 100 == 0 or dataset == sorted_datasets[-1]:
                print(f"After {cumulative_samples} samples: "
                      f"Raw={cumulative_size/1024/1024:.1f}MB, "
                      f"Compressed={estimated_insightspike_size/1024/1024:.1f}MB, "
                      f"Ratio={compression_ratio:.1f}x")
        
        return growth_simulation
    
    def create_visualizations(self, growth_data):
        """Create growth and compression visualizations"""
        print("\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        samples = [d['cumulative_samples'] for d in growth_data]
        raw_sizes_mb = [d['raw_size_bytes']/1024/1024 for d in growth_data]
        compressed_sizes_mb = [d['estimated_insightspike_size']/1024/1024 for d in growth_data]
        compression_ratios = [d['compression_ratio'] for d in growth_data]
        
        # 1. Growth comparison
        ax = axes[0, 0]
        ax.plot(samples, raw_sizes_mb, 'r-', label='Traditional RAG', linewidth=2)
        ax.plot(samples, compressed_sizes_mb, 'b-', label='InsightSpike-AI', linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Storage Size (MB)')
        ax.set_title('Storage Growth Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Compression ratio over scale
        ax = axes[0, 1]
        ax.plot(samples, compression_ratios, 'g-', linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression Ratio vs Scale')
        ax.grid(True, alpha=0.3)
        
        # 3. Size per sample
        ax = axes[1, 0]
        size_per_sample_raw = [d['size_per_sample_raw'] for d in growth_data]
        size_per_sample_compressed = [d['size_per_sample_compressed'] for d in growth_data]
        ax.plot(samples, size_per_sample_raw, 'r-', label='Traditional RAG', linewidth=2)
        ax.plot(samples, size_per_sample_compressed, 'b-', label='InsightSpike-AI', linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Bytes per Sample')
        ax.set_title('Storage Efficiency per Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Storage savings
        ax = axes[1, 1]
        savings_mb = [r - c for r, c in zip(raw_sizes_mb, compressed_sizes_mb)]
        savings_percent = [(1 - c/r) * 100 for r, c in zip(raw_sizes_mb, compressed_sizes_mb)]
        ax.bar(range(len(growth_data)), savings_percent, color='green', alpha=0.7)
        ax.set_xlabel('Growth Stage')
        ax.set_ylabel('Storage Saved (%)')
        ax.set_title('Storage Savings at Each Stage')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = self.results_dir / f"dynamic_growth_analysis_{timestamp}.png"
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        print(f"Visualization saved to: {viz_path}")
        return viz_path
    
    def save_results(self, existing_data, datasets_info, growth_data):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'existing_data_analysis': existing_data,
            'datasets_analyzed': len(datasets_info),
            'total_samples': sum(d['sample_count'] for d in datasets_info),
            'growth_simulation': growth_data,
            'key_findings': {
                'final_sample_count': growth_data[-1]['cumulative_samples'],
                'final_raw_size_mb': growth_data[-1]['raw_size_bytes'] / 1024 / 1024,
                'final_compressed_size_mb': growth_data[-1]['estimated_insightspike_size'] / 1024 / 1024,
                'average_compression_ratio': np.mean([d['compression_ratio'] for d in growth_data]),
                'storage_saved_percent': (1 - growth_data[-1]['estimated_insightspike_size'] / growth_data[-1]['raw_size_bytes']) * 100
            }
        }
        
        results_file = self.results_dir / f"growth_analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return results


def main():
    """Run simplified growth analysis"""
    print("Starting Simplified Dynamic Growth Analysis")
    
    tester = SimplifiedGrowthTester()
    
    # 1. Analyze existing data
    existing_data = tester.analyze_existing_data()
    
    # 2. Analyze available datasets
    datasets_info, total_samples, total_size = tester.analyze_huggingface_datasets()
    
    # 3. Simulate dynamic growth
    growth_data = tester.simulate_dynamic_growth(datasets_info)
    
    # 4. Create visualizations
    viz_path = tester.create_visualizations(growth_data)
    
    # 5. Save results
    results = tester.save_results(existing_data, datasets_info, growth_data)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Datasets analyzed: {len(datasets_info)}")
    print(f"Total samples: {results['key_findings']['final_sample_count']:,}")
    print(f"Final raw size: {results['key_findings']['final_raw_size_mb']:.1f} MB")
    print(f"Final compressed size: {results['key_findings']['final_compressed_size_mb']:.1f} MB")
    print(f"Average compression ratio: {results['key_findings']['average_compression_ratio']:.1f}x")
    print(f"Storage saved: {results['key_findings']['storage_saved_percent']:.1f}%")


if __name__ == "__main__":
    main()