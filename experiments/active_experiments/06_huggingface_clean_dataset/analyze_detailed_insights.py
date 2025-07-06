#!/usr/bin/env python3
"""
Analyze detailed insights from experiment results
================================================

Extract and analyze actual episodes that triggered insights.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import get_config


def analyze_episode_insights():
    """Analyze actual episodes and their insight connections."""
    print("=== Detailed Insight Episode Analysis ===")
    print(f"Start time: {datetime.now()}\n")
    
    # Initialize agent to load current state
    agent = MainAgent()
    agent.initialize()
    agent.load_state()
    
    # Get episodes and analyze connections
    if hasattr(agent, 'l2_memory') and hasattr(agent.l2_memory, 'episodes'):
        episodes = agent.l2_memory.episodes
        print(f"Total episodes in memory: {len(episodes)}")
        
        # Analyze episodes with high C-values or many connections
        insight_episodes = []
        
        # Get graph state
        graph_state = agent.get_memory_graph_state()
        
        # For each episode, analyze its properties
        for i, episode in enumerate(episodes[:50]):  # Analyze first 50 episodes
            episode_data = {
                'episode_id': i,
                'text': episode.text[:100] + '...' if len(episode.text) > 100 else episode.text,
                'c_value': episode.c,
                'metadata': episode.metadata
            }
            
            # Check if this is an integrated episode
            if episode.metadata and 'integration_history' in episode.metadata:
                integration_count = len(episode.metadata['integration_history'])
                episode_data['integration_count'] = integration_count
                episode_data['integrated_texts'] = [
                    hist['integrated_text'][:50] + '...' 
                    for hist in episode.metadata['integration_history'][:3]
                ]
                episode_data['insight_type'] = 'Integration'
            else:
                episode_data['integration_count'] = 0
                episode_data['integrated_texts'] = []
                episode_data['insight_type'] = 'New'
            
            # Check for splits
            if episode.metadata and 'split_from' in episode.metadata:
                episode_data['split_info'] = {
                    'from_episode': episode.metadata['split_from'],
                    'part': episode.metadata.get('split_part', 0),
                    'total_parts': episode.metadata.get('split_total', 0),
                    'reason': episode.metadata.get('split_reason', 'unknown')
                }
                episode_data['insight_type'] = 'Split'
            
            # Calculate insight score
            insight_score = episode.c * (1 + episode_data['integration_count'] * 0.5)
            episode_data['insight_score'] = insight_score
            
            if insight_score > 0.5 or episode_data['integration_count'] > 0:
                insight_episodes.append(episode_data)
        
        # Create detailed CSV
        if insight_episodes:
            # Sort by insight score
            insight_episodes.sort(key=lambda x: x['insight_score'], reverse=True)
            
            # Create DataFrame for CSV
            csv_data = []
            for ep in insight_episodes[:20]:  # Top 20 insights
                csv_data.append({
                    'episode_id': ep['episode_id'],
                    'input_text': ep['text'],
                    'c_value': ep['c_value'],
                    'integration_count': ep['integration_count'],
                    'insight_type': ep['insight_type'],
                    'insight_score': round(ep['insight_score'], 3),
                    'integrated_episodes': ' | '.join(ep['integrated_texts']) if ep['integrated_texts'] else 'None',
                    'split_info': json.dumps(ep.get('split_info', {})) if 'split_info' in ep else 'None'
                })
            
            df = pd.DataFrame(csv_data)
            
            # Save detailed CSV
            csv_file = Path(__file__).parent / 'detailed_insight_episodes.csv'
            df.to_csv(csv_file, index=False)
            print(f"\nSaved detailed insight episodes to: {csv_file}")
            
            # Display summary
            print("\nInsight Type Distribution:")
            type_counts = df['insight_type'].value_counts()
            for insight_type, count in type_counts.items():
                print(f"  {insight_type}: {count}")
            
            print("\nTop 5 Insight Episodes by Score:")
            for _, row in df.head(5).iterrows():
                print(f"\n  Episode {row['episode_id']} (Score: {row['insight_score']}):")
                print(f"    Text: {row['input_text']}")
                print(f"    Type: {row['insight_type']}")
                if row['integration_count'] > 0:
                    print(f"    Integrations: {row['integration_count']}")
                    print(f"    Integrated with: {row['integrated_episodes']}")
        
        # Analyze graph connections for high-insight episodes
        if hasattr(agent, 'l3_graph') and agent.l3_graph and hasattr(agent.l3_graph, 'previous_graph'):
            graph = agent.l3_graph.previous_graph
            if graph and graph.num_nodes > 0:
                print("\nAnalyzing Graph Connections for Insights...")
                
                # Find highly connected nodes
                edge_index = graph.edge_index
                node_connections = {}
                
                for i in range(graph.num_nodes):
                    # Count connections
                    connections = ((edge_index[0] == i) | (edge_index[1] == i)).sum().item()
                    node_connections[i] = connections
                
                # Get top connected nodes
                top_connected = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]
                
                print("\nTop 10 Most Connected Episodes:")
                for node_id, connections in top_connected:
                    if node_id < len(episodes):
                        episode = episodes[node_id]
                        print(f"  Node {node_id}: {connections} connections")
                        print(f"    Text: {episode.text[:80]}...")
                        print(f"    C-value: {episode.c}")
    
    else:
        print("No episodes found in memory")
    
    print(f"\n✅ Analysis completed at: {datetime.now()}")


def create_insight_network_data():
    """Create data for network visualization of insights."""
    print("\nCreating insight network data...")
    
    # Initialize agent
    agent = MainAgent()
    agent.initialize()
    agent.load_state()
    
    network_data = {
        'nodes': [],
        'edges': []
    }
    
    if hasattr(agent, 'l3_graph') and agent.l3_graph and hasattr(agent.l3_graph, 'previous_graph'):
        graph = agent.l3_graph.previous_graph
        episodes = agent.l2_memory.episodes if hasattr(agent.l2_memory, 'episodes') else []
        
        if graph and graph.num_nodes > 0 and episodes:
            # Create node data
            for i in range(min(50, graph.num_nodes)):  # First 50 nodes
                if i < len(episodes):
                    episode = episodes[i]
                    network_data['nodes'].append({
                        'id': i,
                        'label': episode.text[:30] + '...',
                        'c_value': episode.c,
                        'type': 'integration' if episode.metadata and 'integration_history' in episode.metadata else 'normal'
                    })
            
            # Create edge data
            edge_index = graph.edge_index
            edges_added = set()
            
            for idx in range(edge_index.shape[1]):
                src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()
                if src < 50 and dst < 50 and (src, dst) not in edges_added:
                    network_data['edges'].append({
                        'source': src,
                        'target': dst
                    })
                    edges_added.add((src, dst))
            
            # Save network data
            network_file = Path(__file__).parent / 'insight_network_data.json'
            with open(network_file, 'w') as f:
                json.dump(network_data, f, indent=2)
            print(f"Saved network data to: {network_file}")
            print(f"  Nodes: {len(network_data['nodes'])}")
            print(f"  Edges: {len(network_data['edges'])}")


if __name__ == "__main__":
    analyze_episode_insights()
    create_insight_network_data()