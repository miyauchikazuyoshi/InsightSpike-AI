#!/usr/bin/env python3
"""
Test actual dynamic growth of InsightSpike-AI using CLI
"""

import os
import sys
import json
import time
import subprocess
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class ActualGrowthTester:
    def __init__(self):
        self.results_dir = Path("experiment_2/results")
        self.results_dir.mkdir(exist_ok=True)
        self.growth_dir = Path("experiment_2/dynamic_growth")
        self.growth_dir.mkdir(exist_ok=True)
        
        # Track growth metrics
        self.growth_metrics = []
        
    def get_current_metrics(self):
        """Get current data metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'episodes': {},
            'graph': {},
            'index': {}
        }
        
        # Episodes metrics
        episodes_path = Path("data/episodes.json")
        if episodes_path.exists():
            size = episodes_path.stat().st_size
            with open(episodes_path, 'r') as f:
                data = json.load(f)
            metrics['episodes'] = {
                'size_bytes': size,
                'count': len(data),
                'size_per_item': size / len(data) if len(data) > 0 else 0
            }
        
        # Graph metrics
        graph_path = Path("data/graph_pyg.pt")
        if graph_path.exists():
            size = graph_path.stat().st_size
            try:
                graph_data = torch.load(graph_path)
                if hasattr(graph_data, 'num_nodes'):
                    num_nodes = graph_data.num_nodes
                elif isinstance(graph_data, dict) and 'x' in graph_data:
                    num_nodes = graph_data['x'].shape[0]
                else:
                    num_nodes = 0
            except:
                num_nodes = 0
                
            metrics['graph'] = {
                'size_bytes': size,
                'num_nodes': num_nodes,
                'size_per_node': size / num_nodes if num_nodes > 0 else 0
            }
        
        # Index metrics
        index_path = Path("data/index.faiss")
        if index_path.exists():
            metrics['index']['size_bytes'] = index_path.stat().st_size
        
        return metrics
    
    def add_data_via_cli(self, text: str) -> bool:
        """Add data using InsightSpike CLI"""
        try:
            # Use the insight command
            cmd = ['poetry', 'run', 'insight', '-m', text]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Added: {text[:50]}...")
                return True
            else:
                print(f"✗ Failed to add data: {result.stderr}")
                return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def add_data_programmatically(self, text: str) -> bool:
        """Add data using Python API"""
        try:
            from src.insightspike.core.agents.agent_builder import AgentBuilder
            
            # Build agent
            builder = AgentBuilder()
            agent = builder.build_main_agent()
            
            # Process input
            response = agent.process_input(text)
            print(f"✓ Processed: {text[:50]}...")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def run_growth_experiment(self, data_file: str):
        """Run the growth experiment with prepared data"""
        print("\\n=== Starting Dynamic Growth Experiment ===")
        
        # Record initial state
        initial_metrics = self.get_current_metrics()
        self.growth_metrics.append({
            'step': 0,
            'added': 0,
            'metrics': initial_metrics
        })
        print(f"Initial state: {initial_metrics['episodes']['count']} episodes, "
              f"{initial_metrics['graph']['num_nodes']} nodes")
        
        # Load prepared data
        data_path = Path(data_file)
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return
        
        with open(data_path, 'r') as f:
            knowledge_items = json.load(f)
        
        print(f"\\nLoaded {len(knowledge_items)} knowledge items")
        
        # Add data in batches
        batch_size = 5
        total_added = 0
        
        for i in range(0, min(len(knowledge_items), 30), batch_size):  # Limit to 30 for testing
            batch = knowledge_items[i:i+batch_size]
            print(f"\\nProcessing batch {i//batch_size + 1} ({len(batch)} items)...")
            
            batch_added = 0
            for item in batch:
                # Use the knowledge text
                knowledge = item['text']
                
                # Try programmatic addition (faster)
                if self.add_data_programmatically(knowledge):
                    batch_added += 1
                    total_added += 1
                
                time.sleep(0.1)  # Small delay to avoid overwhelming
            
            # Record metrics after batch
            current_metrics = self.get_current_metrics()
            self.growth_metrics.append({
                'step': i//batch_size + 1,
                'added': total_added,
                'metrics': current_metrics
            })
            
            print(f"Batch complete. Total added: {total_added}")
            print(f"Current: {current_metrics['episodes']['count']} episodes, "
                  f"{current_metrics['graph']['num_nodes']} nodes")
        
        return self.growth_metrics
    
    def analyze_growth(self):
        """Analyze growth patterns"""
        if not self.growth_metrics:
            print("No growth data to analyze")
            return
        
        print("\\n=== Growth Analysis ===")
        
        # Extract data series
        steps = [m['step'] for m in self.growth_metrics]
        added = [m['added'] for m in self.growth_metrics]
        episodes = [m['metrics']['episodes']['count'] for m in self.growth_metrics]
        nodes = [m['metrics']['graph']['num_nodes'] for m in self.growth_metrics]
        episode_sizes = [m['metrics']['episodes']['size_bytes'] for m in self.growth_metrics]
        graph_sizes = [m['metrics']['graph']['size_bytes'] for m in self.growth_metrics]
        
        # Calculate growth rates
        if len(episodes) > 1:
            episode_growth = episodes[-1] - episodes[0]
            node_growth = nodes[-1] - nodes[0]
            size_growth = (episode_sizes[-1] + graph_sizes[-1]) - (episode_sizes[0] + graph_sizes[0])
            
            print(f"Episode growth: {episode_growth} ({episode_growth/added[-1]:.2f} per input)")
            print(f"Node growth: {node_growth} ({node_growth/added[-1]:.2f} per input)")
            print(f"Size growth: {size_growth:,} bytes ({size_growth/added[-1]:.1f} bytes per input)")
            
            # Compression ratio (compared to raw text)
            avg_input_size = 150  # Estimated average input size
            raw_size = added[-1] * avg_input_size
            actual_size = size_growth
            compression = raw_size / actual_size if actual_size > 0 else 0
            print(f"Compression ratio: {compression:.1f}x")
        
        # Create visualizations
        self.create_growth_visualizations(steps, added, episodes, nodes, episode_sizes, graph_sizes)
        
        return {
            'total_added': added[-1] if added else 0,
            'episode_growth': episodes[-1] - episodes[0] if len(episodes) > 1 else 0,
            'node_growth': nodes[-1] - nodes[0] if len(nodes) > 1 else 0,
            'compression_ratio': compression if 'compression' in locals() else 0
        }
    
    def create_growth_visualizations(self, steps, added, episodes, nodes, episode_sizes, graph_sizes):
        """Create growth visualization charts"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Data count growth
        ax = axes[0, 0]
        ax.plot(steps, episodes, 'b-o', label='Episodes', linewidth=2)
        ax.plot(steps, nodes, 'r-s', label='Graph Nodes', linewidth=2)
        ax.set_xlabel('Batch Step')
        ax.set_ylabel('Count')
        ax.set_title('Data Growth Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Size growth
        ax = axes[0, 1]
        total_sizes_kb = [(e + g) / 1024 for e, g in zip(episode_sizes, graph_sizes)]
        ax.plot(steps, total_sizes_kb, 'g-^', linewidth=2)
        ax.set_xlabel('Batch Step')
        ax.set_ylabel('Total Size (KB)')
        ax.set_title('Storage Growth')
        ax.grid(True, alpha=0.3)
        
        # 3. Growth rate
        ax = axes[1, 0]
        if len(episodes) > 1:
            episode_rates = [0] + [(episodes[i] - episodes[i-1]) / (added[i] - added[i-1]) 
                                   if added[i] > added[i-1] else 0 
                                   for i in range(1, len(episodes))]
            ax.plot(steps, episode_rates, 'c-d', linewidth=2)
            ax.set_xlabel('Batch Step')
            ax.set_ylabel('Episodes per Input')
            ax.set_title('Episode Growth Rate')
            ax.grid(True, alpha=0.3)
        
        # 4. Efficiency metrics
        ax = axes[1, 1]
        if len(added) > 1 and added[-1] > 0:
            bytes_per_input = [(episode_sizes[i] + graph_sizes[i]) / added[i] 
                               if added[i] > 0 else 0 
                               for i in range(len(added))]
            ax.plot(steps, bytes_per_input, 'm-*', linewidth=2)
            ax.set_xlabel('Batch Step')
            ax.set_ylabel('Bytes per Input')
            ax.set_title('Storage Efficiency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = self.results_dir / f"actual_growth_analysis_{timestamp}.png"
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        print(f"\\nVisualization saved to: {viz_path}")
    
    def save_results(self):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'growth_metrics': self.growth_metrics,
            'summary': self.analyze_growth() if self.growth_metrics else {}
        }
        
        results_file = self.results_dir / f"actual_growth_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\nResults saved to: {results_file}")


def main():
    """Run the actual growth experiment"""
    print("Starting Actual Dynamic Growth Experiment")
    
    # Run growth experiment
    print("\\nRunning growth experiment...")
    tester = ActualGrowthTester()
    
    data_file = "experiment_2/dynamic_growth/test_knowledge.json"
    if Path(data_file).exists():
        growth_metrics = tester.run_growth_experiment(data_file)
        
        # Analyze and save results
        if growth_metrics:
            summary = tester.analyze_growth()
            tester.save_results()
            
            print("\\n=== Experiment Complete ===")
            print(f"Total inputs processed: {summary.get('total_added', 0)}")
            print(f"Episode growth: {summary.get('episode_growth', 0)}")
            print(f"Graph node growth: {summary.get('node_growth', 0)}")
            print(f"Compression ratio: {summary.get('compression_ratio', 0):.1f}x")
    else:
        print(f"Data file not found: {data_file}")


if __name__ == "__main__":
    main()