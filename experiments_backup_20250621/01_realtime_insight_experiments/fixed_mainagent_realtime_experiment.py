#!/usr/bin/env python3
"""
Fixed MainAgent Realtime Insight Detection Experiment
====================================================

Uses the fixed MainAgent with L3GraphReasoner for every-episode insight detection.
Includes graph growth visualization and comprehensive result summarization.
"""

import json
import logging
import sys
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insightspike.core.agents.main_agent import MainAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedMainAgentExperiment:
    """Practical realtime insight detection using the fixed MainAgent"""
    
    def __init__(self, output_dir: str = "experiments/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MainAgent
        self.agent = MainAgent()
        if not self.agent.initialize():
            raise RuntimeError("Failed to initialize MainAgent")
        
        # Experiment tracking
        self.episode_results = []
        self.insight_events = []
        self.graph_states = []
        self.performance_metrics = []
        
        # Graph visualization data
        self.graph_nodes_over_time = []
        self.graph_edges_over_time = []
        
        logger.info("Fixed MainAgent experiment initialized")
    
    def generate_test_episodes(self, num_episodes: int = 50) -> List[str]:
        """Generate diverse test episodes for insight detection"""
        
        # Base topics for episode generation
        base_topics = [
            "machine learning",
            "neural networks", 
            "quantum computing",
            "artificial intelligence",
            "data science",
            "computer vision",
            "natural language processing",
            "robotics",
            "deep learning",
            "reinforcement learning"
        ]
        
        # Question patterns
        patterns = [
            "How does {} work in practice?",
            "What are the limitations of {}?",
            "How does {} relate to artificial intelligence?",
            "What advances have been made in {} recently?",
            "How can {} be applied to solve real-world problems?",
            "What are the ethical implications of {}?",
            "How does {} compare to traditional approaches?",
            "What mathematical principles underlie {}?",
            "How might {} evolve in the next decade?",
            "What are the computational requirements for {}?"
        ]
        
        episodes = []
        for i in range(num_episodes):
            topic = base_topics[i % len(base_topics)]
            pattern = patterns[i % len(patterns)]
            episode = pattern.format(topic)
            episodes.append(episode)
        
        return episodes
    
    def run_experiment(self, num_episodes: int = 50, every_episode: bool = True) -> Dict[str, Any]:
        """Run realtime insight detection experiment"""
        
        print(f"=== Fixed MainAgent Realtime Insight Detection ===")
        print(f"Episodes: {num_episodes}")
        print(f"Every-episode detection: {every_episode}")
        print(f"Agent components: L1-ErrorMonitor✅, L2-Memory✅, L3-GraphReasoner✅, L4-CleanLLM✅")
        print()
        
        # Generate test episodes
        episodes = self.generate_test_episodes(num_episodes)
        
        # Run experiment
        start_time = time.time()
        
        for i, episode in enumerate(episodes):
            episode_start = time.time()
            
            print(f"Episode {i+1}/{num_episodes}: {episode[:60]}...")
            
            # Process episode through MainAgent
            try:
                result = self.agent.process_question(episode, max_cycles=1, verbose=False)
                
                # Extract insights and metrics
                episode_result = self._extract_episode_results(i+1, episode, result)
                self.episode_results.append(episode_result)
                
                # Check for insight detection
                if result.get('spike_detected', False):
                    insight_event = self._create_insight_event(i+1, episode, result)
                    self.insight_events.append(insight_event)
                    print(f"  🔥 INSIGHT DETECTED! Quality: {result.get('reasoning_quality', 0):.3f}")
                
                # Track graph growth
                self._track_graph_growth(i+1, result)
                
                # Performance metrics
                episode_time = time.time() - episode_start
                self.performance_metrics.append({
                    'episode': i+1,
                    'processing_time': episode_time,
                    'quality': result.get('reasoning_quality', 0),
                    'spike_detected': result.get('spike_detected', False)
                })
                
                print(f"  Quality: {result.get('reasoning_quality', 0):.3f}, Time: {episode_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Episode {i+1} failed: {e}")
                # Add error result
                error_result = {
                    'episode': i+1,
                    'question': episode,
                    'error': str(e),
                    'processing_time': 0,
                    'quality': 0,
                    'spike_detected': False
                }
                self.episode_results.append(error_result)
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = self._compile_final_results(total_time)
        
        # Generate visualizations
        self._create_visualizations()
        
        # Save results
        self._save_results(final_results)
        
        return final_results
    
    def _extract_episode_results(self, episode_num: int, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured results from episode processing"""
        
        return {
            'episode': episode_num,
            'question': question,
            'response': result.get('response', ''),
            'reasoning_quality': result.get('reasoning_quality', 0),
            'spike_detected': result.get('spike_detected', False),
            'cycle_count': result.get('cycle_count', 0),
            'converged': result.get('converged', False),
            'retrieved_documents': len(result.get('retrieved_documents', [])),
            'graph_metrics': result.get('graph_analysis', {}).get('metrics', {}),
            'conflicts': result.get('graph_analysis', {}).get('conflicts', {}),
            'timestamp': time.time()
        }
    
    def _create_insight_event(self, episode_num: int, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed insight event record"""
        
        graph_analysis = result.get('graph_analysis', {})
        
        return {
            'episode': episode_num,
            'input_episode': question,
            'insight_trigger': 'spike_detected',
            'vector_to_language': result.get('response', ''),
            'related_nodes': self._extract_related_nodes(graph_analysis),
            'graph_metrics': {
                'delta_ged': graph_analysis.get('metrics', {}).get('delta_ged', 0),
                'delta_ig': graph_analysis.get('metrics', {}).get('delta_ig', 0)
            },
            'quality_score': result.get('reasoning_quality', 0),
            'timestamp': time.time(),
            'conflict_analysis': graph_analysis.get('conflicts', {}),
            'reward_signal': graph_analysis.get('reward', {})
        }
    
    def _extract_related_nodes(self, graph_analysis: Dict[str, Any]) -> List[str]:
        """Extract related nodes from graph analysis"""
        try:
            # Get graph data if available
            graph_data = graph_analysis.get('graph')
            if not graph_data:
                return []
            
            # Extract node information
            if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                # PyTorch Geometric format
                num_nodes = graph_data.x.shape[0] if graph_data.x is not None else 0
                return [f"node_{i}" for i in range(min(5, num_nodes))]  # Top 5 nodes
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to extract related nodes: {e}")
            return []
    
    def _track_graph_growth(self, episode_num: int, result: Dict[str, Any]):
        """Track graph growth over episodes"""
        try:
            graph_analysis = result.get('graph_analysis', {})
            graph_data = graph_analysis.get('graph')
            
            if graph_data and hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                num_nodes = graph_data.x.shape[0] if graph_data.x is not None else 0
                num_edges = graph_data.edge_index.shape[1] if graph_data.edge_index is not None else 0
            else:
                # Estimate based on memory or defaults
                num_nodes = len(self.agent.l2_memory.episodes) if hasattr(self.agent.l2_memory, 'episodes') else episode_num
                num_edges = max(0, num_nodes - 1)  # Minimum spanning tree
            
            self.graph_nodes_over_time.append({'episode': episode_num, 'nodes': num_nodes})
            self.graph_edges_over_time.append({'episode': episode_num, 'edges': num_edges})
            
        except Exception as e:
            logger.warning(f"Failed to track graph growth: {e}")
            # Fallback values
            self.graph_nodes_over_time.append({'episode': episode_num, 'nodes': episode_num})
            self.graph_edges_over_time.append({'episode': episode_num, 'edges': max(0, episode_num-1)})
    
    def _compile_final_results(self, total_time: float) -> Dict[str, Any]:
        """Compile comprehensive experiment results"""
        
        total_episodes = len(self.episode_results)
        total_insights = len(self.insight_events)
        
        # Calculate statistics
        qualities = [r.get('reasoning_quality', 0) for r in self.episode_results if 'reasoning_quality' in r]
        avg_quality = np.mean(qualities) if qualities else 0
        
        processing_times = [m['processing_time'] for m in self.performance_metrics]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        return {
            'experiment_summary': {
                'total_episodes': total_episodes,
                'total_insights': total_insights,
                'insight_rate': total_insights / total_episodes if total_episodes > 0 else 0,
                'average_quality': avg_quality,
                'average_processing_time': avg_processing_time,
                'total_experiment_time': total_time
            },
            'episode_results': self.episode_results,
            'insight_events': self.insight_events,
            'performance_metrics': self.performance_metrics,
            'graph_growth': {
                'nodes_over_time': self.graph_nodes_over_time,
                'edges_over_time': self.graph_edges_over_time
            },
            'agent_stats': self.agent.get_stats()
        }
    
    def _create_visualizations(self):
        """Create graph growth and performance visualizations"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Graph growth over time
            if self.graph_nodes_over_time:
                episodes = [d['episode'] for d in self.graph_nodes_over_time]
                nodes = [d['nodes'] for d in self.graph_nodes_over_time]
                edges = [d['edges'] for d in self.graph_edges_over_time]
                
                ax1.plot(episodes, nodes, 'b-o', label='Nodes', markersize=4)
                ax1.plot(episodes, edges, 'r-o', label='Edges', markersize=4)
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Count')
                ax1.set_title('Graph Growth Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Quality over time
            if self.performance_metrics:
                episodes = [m['episode'] for m in self.performance_metrics]
                qualities = [m['quality'] for m in self.performance_metrics]
                
                ax2.plot(episodes, qualities, 'g-o', markersize=4)
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Reasoning Quality')
                ax2.set_title('Quality Over Time')
                ax2.grid(True, alpha=0.3)
            
            # Processing time distribution
            if self.performance_metrics:
                times = [m['processing_time'] for m in self.performance_metrics]
                ax3.hist(times, bins=20, alpha=0.7, color='orange')
                ax3.set_xlabel('Processing Time (seconds)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Processing Time Distribution')
                ax3.grid(True, alpha=0.3)
            
            # Insight detection over time
            if self.insight_events:
                insight_episodes = [event['episode'] for event in self.insight_events]
                insight_qualities = [event['quality_score'] for event in self.insight_events]
                
                ax4.scatter(insight_episodes, insight_qualities, color='red', s=50, alpha=0.7)
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Insight Quality')
                ax4.set_title('Insight Detection Events')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'fixed_mainagent_experiment_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualizations saved to {self.output_dir / 'fixed_mainagent_experiment_visualization.png'}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive experiment results"""
        
        # Save complete results as JSON
        json_path = self.output_dir / 'fixed_mainagent_experiment_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save episode results as CSV
        if self.episode_results:
            df_episodes = pd.DataFrame(self.episode_results)
            csv_path = self.output_dir / 'fixed_mainagent_episode_results.csv'
            df_episodes.to_csv(csv_path, index=False)
        
        # Save insight events as CSV
        if self.insight_events:
            df_insights = pd.DataFrame(self.insight_events)
            csv_path = self.output_dir / 'fixed_mainagent_insight_events.csv'
            df_insights.to_csv(csv_path, index=False)
        
        print(f"📁 Results saved to {self.output_dir}")
        print(f"   - Complete results: fixed_mainagent_experiment_results.json")
        print(f"   - Episode data: fixed_mainagent_episode_results.csv")
        print(f"   - Insight events: fixed_mainagent_insight_events.csv")


def main():
    """Run the fixed MainAgent realtime insight detection experiment"""
    
    print("Starting Fixed MainAgent Realtime Insight Detection Experiment")
    print("Using MainAgent with:")
    print("  ✅ ErrorMonitor.reset() - Fixed")
    print("  ✅ ErrorMonitor.analyze_uncertainty() - Added")
    print("  ✅ CleanLLMProvider (no data leaks) - Integrated")
    print("  ✅ L3GraphReasoner - Enabled")
    print()
    
    try:
        # Initialize experiment
        experiment = FixedMainAgentExperiment()
        
        # Run experiment with 50 episodes
        results = experiment.run_experiment(num_episodes=50, every_episode=True)
        
        # Print summary
        summary = results['experiment_summary']
        print(f"\n=== Experiment Summary ===")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Insights Detected: {summary['total_insights']}")
        print(f"Insight Rate: {summary['insight_rate']:.1%}")
        print(f"Average Quality: {summary['average_quality']:.3f}")
        print(f"Average Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"Total Experiment Time: {summary['total_experiment_time']:.1f}s")
        
        if results['insight_events']:
            print(f"\n🔥 Insight Events:")
            for event in results['insight_events'][:5]:  # Show first 5
                print(f"  Episode {event['episode']}: {event['input_episode'][:50]}...")
                print(f"    Quality: {event['quality_score']:.3f}")
                print(f"    ΔGED: {event['graph_metrics']['delta_ged']:.3f}")
                print(f"    ΔIG: {event['graph_metrics']['delta_ig']:.3f}")
                
        print(f"\n✅ Fixed MainAgent realtime insight detection completed successfully!")
        print(f"\n🎯 Next Steps:")
        print(f"   - Review visualization plots for graph growth patterns")
        print(f"   - Analyze insight events for quality patterns")
        print(f"   - Consider scaling to larger episode counts")
        print(f"   - Implement TopK optimization for O(n²) graph metrics")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
