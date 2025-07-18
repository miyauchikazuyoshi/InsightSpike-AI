#!/usr/bin/env python3
"""
Rebuild proper database from english_definitions.json
"""

import json
import networkx as nx
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_proper_database():
    """Build proper knowledge base from english definitions"""
    logger.info("🔨 Building proper database from english_definitions.json...")
    
    # Load definitions
    input_file = Path(__file__).parent.parent / "data" / "input" / "english_definitions.json"
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    definitions = data['definitions']
    
    # Create graph
    G = nx.Graph()
    
    # RAT problems for answer identification
    rat_problems = {
        ("COTTAGE", "SWISS", "CAKE"): "CHEESE",
        ("CREAM", "SKATE", "WATER"): "ICE",
        ("DUCK", "FOLD", "DOLLAR"): "BILL",
        ("NIGHT", "WRIST", "STOP"): "WATCH",
        ("RIVER", "NOTE", "ACCOUNT"): "BANK"
    }
    
    # All answer words
    answer_words = set(rat_problems.values())
    
    # Add word nodes
    for word in definitions.keys():
        G.add_node(word, 
                  node_type="word",
                  is_answer=word in answer_words,
                  importance=0.7 if word in answer_words else 0.5)
    
    # Add semantic relationships based on co-occurrence in definitions
    for word1 in definitions:
        for word2 in definitions:
            if word1 < word2:  # Avoid duplicates
                # Check if words appear in each other's definitions
                word1_in_word2 = any(word1.lower() in defn.lower() for defn in definitions[word2])
                word2_in_word1 = any(word2.lower() in defn.lower() for defn in definitions[word1])
                
                if word1_in_word2 or word2_in_word1:
                    weight = 0.8 if (word1_in_word2 and word2_in_word1) else 0.6
                    G.add_edge(word1, word2, weight=weight, edge_type="semantic")
    
    # Add problem-based connections
    for problem_words, answer in rat_problems.items():
        if answer in G:
            for word in problem_words:
                if word in G and not G.has_edge(word, answer):
                    G.add_edge(word, answer, weight=0.7, edge_type="rat_association")
    
    # Calculate centrality
    betweenness = nx.betweenness_centrality(G)
    for node in G.nodes():
        G.nodes[node]['centrality'] = G.degree(node) / len(G.nodes())
        G.nodes[node]['betweenness'] = betweenness[node]
    
    # Create episodes from definitions
    episodes = []
    episode_id = 0
    
    for word, defn_list in definitions.items():
        for priority, definition in enumerate(defn_list):
            # Check if this definition contains any answer words
            contains_answer = any(ans.lower() in definition.lower() for ans in answer_words)
            
            # Check if this is a RAT-relevant definition
            is_rat_relevant = False
            for problem_words, answer in rat_problems.items():
                if word in problem_words and answer.lower() in definition.lower():
                    is_rat_relevant = True
                    break
            
            episode = {
                "text": definition,
                "source_word": word,
                "priority": priority,
                "contains_answer": contains_answer,
                "is_rat_relevant": is_rat_relevant,
                "episode_id": episode_id
            }
            episodes.append(episode)
            episode_id += 1
    
    # Save outputs
    output_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save graph
    graph_data = nx.node_link_data(G)
    with open(output_dir / "proper_rat_knowledge_graph.json", 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    # Save episodes
    episodes_data = {
        "metadata": {
            "source": "english_definitions.json",
            "total_episodes": len(episodes),
            "total_words": len(definitions),
            "answer_words": list(answer_words)
        },
        "episodes": episodes
    }
    
    with open(output_dir / "proper_rat_episodes.json", 'w') as f:
        json.dump(episodes_data, f, indent=2)
    
    logger.info(f"✅ Database built successfully!")
    logger.info(f"   - Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"   - Episodes: {len(episodes)} total")
    logger.info(f"   - RAT-relevant episodes: {sum(1 for e in episodes if e['is_rat_relevant'])}")
    logger.info(f"   - Answer-containing episodes: {sum(1 for e in episodes if e['contains_answer'])}")
    
    # Show sample connections
    logger.info("\n📊 Sample connections:")
    for word in ["CHEESE", "ICE", "BILL"]:
        if word in G:
            neighbors = list(G.neighbors(word))[:5]
            logger.info(f"   {word} → {', '.join(neighbors)}")


if __name__ == "__main__":
    build_proper_database()