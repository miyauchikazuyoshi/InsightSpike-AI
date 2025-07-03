
#!/usr/bin/env python3
"""
Analyze insights from the InsightSpike-AI knowledge graph
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent

def load_data():
    """Load episodes and graph data"""
    # Load episodes
    with open('data/episodes.json', 'r') as f:
        episodes = json.load(f)
    
    # Load graph
    graph = torch.load('data/graph_pyg.pt')
    
    return episodes, graph

def analyze_topics(episodes):
    """Analyze topic distribution in episodes"""
    print("=== Topic Analysis ===\n")
    
    # Extract topics and fields
    topics = []
    fields = []
    topic_field_pairs = []
    
    for ep in episodes:
        text = ep['text'].lower()
        
        # Common topics in our dataset
        topic_keywords = {
            'artificial intelligence': 'AI',
            'machine learning': 'ML',
            'deep learning': 'DL',
            'natural language processing': 'NLP',
            'computer vision': 'CV',
            'robotics': 'robotics',
            'quantum computing': 'quantum',
            'blockchain': 'blockchain',
            'cybersecurity': 'security',
            'data science': 'data science',
            'neural networks': 'neural nets',
            'reinforcement learning': 'RL'
        }
        
        field_keywords = {
            'healthcare': 'healthcare',
            'finance': 'finance',
            'education': 'education',
            'manufacturing': 'manufacturing',
            'transportation': 'transport',
            'energy': 'energy',
            'agriculture': 'agriculture',
            'retail': 'retail',
            'telecommunications': 'telecom',
            'aerospace': 'aerospace'
        }
        
        # Find topics and fields
        found_topic = None
        found_field = None
        
        for keyword, label in topic_keywords.items():
            if keyword in text:
                topics.append(label)
                found_topic = label
                break
        
        for keyword, label in field_keywords.items():
            if keyword in text:
                fields.append(label)
                found_field = label
                break
        
        if found_topic and found_field:
            topic_field_pairs.append((found_topic, found_field))
    
    # Analyze distributions
    topic_dist = Counter(topics)
    field_dist = Counter(fields)
    pair_dist = Counter(topic_field_pairs)
    
    print("Top Technologies:")
    for topic, count in topic_dist.most_common(10):
        print(f"  {topic}: {count} episodes")
    
    print("\nTop Application Fields:")
    for field, count in field_dist.most_common(10):
        print(f"  {field}: {count} episodes")
    
    print("\nMost Common Technology-Field Combinations:")
    for (topic, field), count in pair_dist.most_common(10):
        print(f"  {topic} → {field}: {count} occurrences")
    
    return topic_dist, field_dist, pair_dist

def analyze_graph_structure(graph):
    """Analyze graph connectivity patterns"""
    print("\n=== Graph Structure Analysis ===\n")
    
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.size(1)
    
    # Calculate node degrees
    edge_index = graph.edge_index.numpy()
    node_degrees = np.bincount(edge_index[0])
    
    avg_degree = np.mean(node_degrees)
    max_degree = np.max(node_degrees)
    min_degree = np.min(node_degrees)
    
    print(f"Graph Statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Average degree: {avg_degree:.2f}")
    print(f"  Max degree: {max_degree}")
    print(f"  Min degree: {min_degree}")
    print(f"  Graph density: {num_edges / (num_nodes * (num_nodes - 1)):.4f}")
    
    # Find hub nodes (highly connected)
    hub_threshold = avg_degree + 2 * np.std(node_degrees)
    hub_nodes = np.where(node_degrees > hub_threshold)[0]
    
    print(f"\nHub nodes (degree > {hub_threshold:.0f}): {len(hub_nodes)}")
    
    return node_degrees, hub_nodes

def analyze_embeddings(graph, episodes):
    """Analyze embedding clusters to find knowledge patterns"""
    print("\n=== Embedding Analysis ===\n")
    
    # Get embeddings
    embeddings = graph.x.numpy()
    
    # Perform clustering
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Analyze cluster contents
    cluster_topics = defaultdict(list)
    
    for idx, cluster in enumerate(clusters):
        if idx < len(episodes):
            text = episodes[idx]['text']
            cluster_topics[cluster].append(text)
    
    print(f"Knowledge Clusters (K={n_clusters}):")
    for cluster_id in range(n_clusters):
        texts = cluster_topics[cluster_id]
        print(f"\nCluster {cluster_id} ({len(texts)} documents):")
        if texts:
            # Show sample texts
            for text in texts[:3]:
                print(f"  - {text[:60]}...")
    
    # PCA visualization prep
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return clusters, embeddings_2d

def find_insights(episodes, graph, node_degrees, hub_nodes):
    """Extract key insights from the data"""
    print("\n=== Key Insights ===\n")
    
    insights = []
    
    # Insight 1: Technology convergence
    texts = [ep['text'] for ep in episodes]
    multi_tech_count = sum(1 for text in texts if 
                          sum(tech in text.lower() for tech in 
                              ['artificial intelligence', 'machine learning', 
                               'deep learning', 'neural network']) >= 2)
    
    convergence_rate = multi_tech_count / len(texts) * 100
    insights.append(f"1. Technology Convergence: {convergence_rate:.1f}% of episodes mention multiple AI technologies")
    
    # Insight 2: Dominant application domains
    healthcare_count = sum(1 for text in texts if 'healthcare' in text.lower())
    finance_count = sum(1 for text in texts if 'finance' in text.lower())
    
    insights.append(f"2. Healthcare ({healthcare_count}) and Finance ({finance_count}) are leading AI adoption")
    
    # Insight 3: Graph connectivity
    avg_degree = np.mean(node_degrees)
    insights.append(f"3. High Knowledge Connectivity: Average {avg_degree:.0f} connections per concept")
    
    # Insight 4: Hub identification
    if len(hub_nodes) > 0:
        hub_texts = [episodes[idx]['text'][:50] for idx in hub_nodes[:3] if idx < len(episodes)]
        insights.append(f"4. Knowledge Hubs: {len(hub_nodes)} central concepts identified")
    
    # Insight 5: Emerging patterns
    recent_texts = texts[-50:]  # Last 50 episodes
    quantum_recent = sum(1 for text in recent_texts if 'quantum' in text.lower())
    blockchain_recent = sum(1 for text in recent_texts if 'blockchain' in text.lower())
    
    insights.append(f"5. Emerging Tech: Quantum ({quantum_recent}) and Blockchain ({blockchain_recent}) in recent additions")
    
    for insight in insights:
        print(f"• {insight}")
    
    return insights

def generate_recommendations():
    """Generate actionable recommendations"""
    print("\n=== Recommendations ===\n")
    
    recommendations = [
        "1. Focus on Healthcare and Finance sectors for immediate AI impact",
        "2. Invest in cross-technology integration (AI + ML + DL)",
        "3. Monitor emerging technologies (Quantum, Blockchain) for future opportunities",
        "4. Leverage highly connected knowledge hubs for faster learning",
        "5. Build domain-specific AI solutions for underserved sectors"
    ]
    
    for rec in recommendations:
        print(f"→ {rec}")

def main():
    """Run complete insight analysis"""
    print("=== InsightSpike-AI Knowledge Base Analysis ===\n")
    
    # Load data
    episodes, graph = load_data()
    print(f"Loaded {len(episodes)} episodes and graph with {graph.num_nodes} nodes\n")
    
    # Analyze topics
    topic_dist, field_dist, pair_dist = analyze_topics(episodes)
    
    # Analyze graph structure
    node_degrees, hub_nodes = analyze_graph_structure(graph)
    
    # Analyze embeddings
    clusters, embeddings_2d = analyze_embeddings(graph, episodes)
    
    # Find insights
    insights = find_insights(episodes, graph, node_degrees, hub_nodes)
    
    # Generate recommendations
    generate_recommendations()
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()