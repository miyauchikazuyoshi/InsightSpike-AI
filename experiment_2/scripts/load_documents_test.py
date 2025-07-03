#!/usr/bin/env python3
"""
Test loading documents to see actual data growth
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def get_data_metrics():
    """Get comprehensive data metrics"""
    metrics = {}
    
    # Episodes
    episodes_path = Path("data/episodes.json")
    if episodes_path.exists():
        with open(episodes_path, 'r') as f:
            episodes = json.load(f)
        metrics['episodes'] = {
            'count': len(episodes),
            'size': episodes_path.stat().st_size,
            'size_per_item': episodes_path.stat().st_size / len(episodes) if episodes else 0
        }
    
    # Graph
    graph_path = Path("data/graph_pyg.pt")
    if graph_path.exists():
        metrics['graph'] = {
            'size': graph_path.stat().st_size
        }
    
    # Index
    index_path = Path("data/index.faiss")
    if index_path.exists():
        metrics['index'] = {
            'size': index_path.stat().st_size
        }
    
    # Total size
    total_size = 0
    for key in ['episodes', 'graph', 'index']:
        if key in metrics and 'size' in metrics[key]:
            total_size += metrics[key]['size']
    metrics['total_size'] = total_size
    
    return metrics


def create_test_batches():
    """Create batch files for testing"""
    # Load test knowledge
    test_data_path = Path("experiment_2/dynamic_growth/test_knowledge.json")
    with open(test_data_path, 'r') as f:
        knowledge_items = json.load(f)
    
    # Create batch files
    batch_dir = Path("experiment_2/dynamic_growth/batches")
    batch_dir.mkdir(exist_ok=True)
    
    batch_size = 5
    for i in range(0, len(knowledge_items), batch_size):
        batch = knowledge_items[i:i+batch_size]
        batch_file = batch_dir / f"batch_{i//batch_size + 1}.txt"
        
        with open(batch_file, 'w') as f:
            for item in batch:
                f.write(f"{item['text']}\\n\\n")
        
        print(f"Created {batch_file}")
    
    return list(batch_dir.glob("batch_*.txt"))


def load_batch(batch_file):
    """Load a batch file using InsightSpike CLI"""
    try:
        cmd = ['poetry', 'run', 'insightspike', 'load-documents', str(batch_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✓ Loaded: {batch_file.name}")
            return True
        else:
            print(f"✗ Failed to load {batch_file.name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error loading {batch_file.name}: {e}")
        return False


def main():
    """Run document loading test"""
    print("Testing InsightSpike Document Loading")
    print("=" * 50)
    
    # Get initial metrics
    initial_metrics = get_data_metrics()
    print(f"\\nInitial metrics:")
    print(f"  Episodes: {initial_metrics.get('episodes', {}).get('count', 0)}")
    print(f"  Total size: {initial_metrics['total_size']:,} bytes")
    
    # Create test batches
    print("\\nCreating test batches...")
    batch_files = create_test_batches()
    print(f"Created {len(batch_files)} batch files")
    
    # Track growth
    growth_data = [{
        'batch': 0,
        'metrics': initial_metrics
    }]
    
    # Load batches one by one
    print("\\nLoading batches...")
    for i, batch_file in enumerate(sorted(batch_files)[:6]):  # Limit to 6 batches
        print(f"\\n[Batch {i+1}]")
        
        if load_batch(batch_file):
            time.sleep(2)  # Wait for processing
            
            # Get updated metrics
            current_metrics = get_data_metrics()
            growth_data.append({
                'batch': i + 1,
                'metrics': current_metrics
            })
            
            # Show progress
            episode_growth = current_metrics.get('episodes', {}).get('count', 0) - initial_metrics.get('episodes', {}).get('count', 0)
            size_growth = current_metrics['total_size'] - initial_metrics['total_size']
            
            print(f"  Episodes: {current_metrics.get('episodes', {}).get('count', 0)} (+{episode_growth})")
            print(f"  Total size: {current_metrics['total_size']:,} bytes (+{size_growth:,})")
    
    # Analyze results
    print("\\n" + "=" * 50)
    print("RESULTS ANALYSIS")
    
    final_metrics = growth_data[-1]['metrics']
    total_episode_growth = final_metrics.get('episodes', {}).get('count', 0) - initial_metrics.get('episodes', {}).get('count', 0)
    total_size_growth = final_metrics['total_size'] - initial_metrics['total_size']
    
    print(f"\\nTotal growth:")
    print(f"  Episodes: +{total_episode_growth}")
    print(f"  Size: +{total_size_growth:,} bytes")
    
    if total_episode_growth > 0:
        bytes_per_episode = total_size_growth / total_episode_growth
        print(f"  Bytes per episode: {bytes_per_episode:.1f}")
        
        # Estimate compression ratio (assuming 150 chars per input)
        estimated_raw_size = total_episode_growth * 150
        compression_ratio = estimated_raw_size / total_size_growth if total_size_growth > 0 else 0
        print(f"  Estimated compression ratio: {compression_ratio:.1f}x")
    
    # Create visualization
    if len(growth_data) > 1:
        create_growth_chart(growth_data)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'growth_data': growth_data,
        'summary': {
            'episode_growth': total_episode_growth,
            'size_growth': total_size_growth,
            'batches_loaded': len(growth_data) - 1
        }
    }
    
    results_path = Path("experiment_2/results/load_documents_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to: {results_path}")


def create_growth_chart(growth_data):
    """Create visualization of growth"""
    batches = [d['batch'] for d in growth_data]
    episodes = [d['metrics'].get('episodes', {}).get('count', 0) for d in growth_data]
    sizes_kb = [d['metrics']['total_size'] / 1024 for d in growth_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Episode growth
    ax1.plot(batches, episodes, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Episode Count')
    ax1.set_title('Episode Growth')
    ax1.grid(True, alpha=0.3)
    
    # Size growth
    ax2.plot(batches, sizes_kb, 'g-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Total Size (KB)')
    ax2.set_title('Storage Growth')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = Path(f"experiment_2/results/document_loading_growth_{timestamp}.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"\\nChart saved to: {chart_path}")


if __name__ == "__main__":
    main()