#!/usr/bin/env python3
"""
Check episode integration/splitting in experiment
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.agents.main_agent import MainAgent


def test_episode_integration():
    """Test episode integration behavior"""
    print("=== Episode Integration/Split Analysis ===")
    print(f"Time: {datetime.now()}\n")
    
    # Initialize agent
    agent = MainAgent()
    agent.initialize()
    
    # Get initial state
    initial_episodes = len(agent.l2_memory.episodes)
    print(f"Initial episodes: {initial_episodes}")
    
    # Track integration statistics
    stats = {
        'total_additions': 0,
        'integrations': 0,
        'new_nodes': 0,
        'splits': 0,
        'episodes_over_time': []
    }
    
    # Generate test documents with varying similarity
    documents = []
    
    # Create clusters of similar documents
    print("\nGenerating test documents...")
    topics = ["AI research", "quantum computing", "biotechnology", "space exploration", "climate science"]
    
    for topic_id, topic in enumerate(topics):
        # Base document for each topic
        base_text = f"Fundamental research in {topic} provides new insights"
        documents.append((base_text, topic_id))
        
        # Similar variations (should integrate)
        for i in range(3):
            variation = f"Advanced {topic} research reveals novel findings and insights"
            documents.append((variation, topic_id))
        
        # Somewhat different (edge case)
        edge_case = f"Applications of {topic} in industry"
        documents.append((edge_case, topic_id))
    
    # Process documents and track integration
    print("\nProcessing documents and tracking integration...")
    for i, (text, topic_id) in enumerate(documents):
        print(f"\n--- Document {i+1}/{len(documents)} ---")
        print(f"Text: {text[:50]}...")
        print(f"Topic: {topics[topic_id]}")
        
        # Add episode
        result = agent.add_episode_with_graph_update(text)
        stats['total_additions'] += 1
        
        if result['success']:
            # Check if it was integrated or added as new
            current_episodes = len(agent.l2_memory.episodes)
            episodes_added = current_episodes - len(stats['episodes_over_time']) if stats['episodes_over_time'] else current_episodes - initial_episodes
            
            if episodes_added == 0:
                stats['integrations'] += 1
                print("Result: INTEGRATED with existing episode")
            else:
                stats['new_nodes'] += 1
                print("Result: Added as NEW episode node")
            
            stats['episodes_over_time'].append(current_episodes)
            
            # Print current state
            print(f"Current episodes: {current_episodes}")
            print(f"C-value: {result.get('c_value', 'N/A')}")
    
    # Check for any splits (would require checking episode history)
    # Note: Current implementation doesn't seem to have automatic splitting
    
    # Final analysis
    print("\n\n=== Final Analysis ===")
    print(f"Total documents processed: {stats['total_additions']}")
    print(f"Episodes integrated: {stats['integrations']} ({stats['integrations']/stats['total_additions']*100:.1f}%)")
    print(f"New episode nodes: {stats['new_nodes']} ({stats['new_nodes']/stats['total_additions']*100:.1f}%)")
    print(f"Episode splits: {stats['splits']}")
    print(f"\nFinal episode count: {len(agent.l2_memory.episodes)}")
    print(f"Net episodes added: {len(agent.l2_memory.episodes) - initial_episodes}")
    
    # Check integration thresholds
    print("\n=== Integration Thresholds ===")
    if hasattr(agent.l2_memory.config, 'reasoning'):
        print(f"Similarity threshold: {getattr(agent.l2_memory.config.reasoning, 'episode_integration_similarity_threshold', 0.85)}")
        print(f"Content overlap threshold: {getattr(agent.l2_memory.config.reasoning, 'episode_integration_content_threshold', 0.7)}")
        print(f"C-value diff threshold: {getattr(agent.l2_memory.config.reasoning, 'episode_integration_c_threshold', 0.3)}")
    else:
        print("Using default thresholds: similarity=0.85, content=0.7, c_diff=0.3")
    
    # Analyze episode growth pattern
    if stats['episodes_over_time']:
        print("\n=== Episode Growth Pattern ===")
        growth_points = []
        for i in range(1, len(stats['episodes_over_time'])):
            if stats['episodes_over_time'][i] > stats['episodes_over_time'][i-1]:
                growth_points.append(i)
        
        print(f"Growth occurred at document indices: {growth_points}")
        
    return stats


def check_existing_data():
    """Check if there's existing data with integration/split info"""
    print("\n=== Checking Existing Data ===")
    
    episodes_file = "data/episodes.json"
    if os.path.exists(episodes_file):
        with open(episodes_file, 'r') as f:
            episodes = json.load(f)
        
        print(f"Found {len(episodes)} episodes in data/episodes.json")
        
        # Look for integration markers
        integrated_count = 0
        for ep in episodes:
            # Check if episode has integration markers
            if 'integrated_from' in ep or 'integration_count' in ep:
                integrated_count += 1
        
        if integrated_count > 0:
            print(f"Episodes with integration markers: {integrated_count}")
        else:
            print("No explicit integration markers found in episodes")
    else:
        print("No existing episodes.json found")


if __name__ == "__main__":
    # Check existing data first
    check_existing_data()
    
    # Run integration test
    print("\n" + "="*50 + "\n")
    test_episode_integration()