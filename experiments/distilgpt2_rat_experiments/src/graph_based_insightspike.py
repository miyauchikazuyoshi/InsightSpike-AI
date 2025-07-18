#!/usr/bin/env python3
"""
Graph-based InsightSpike with message passing
Uses word meanings and connections for insight detection
"""

import json
import networkx as nx
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class GraphBasedInsightSpike:
    def __init__(self):
        print("🚀 Initializing Graph-based InsightSpike...")
        
        # Load knowledge graph
        kg_path = Path(__file__).parent.parent / "data" / "input" / "rat_knowledge_graph.json"
        with open(kg_path, 'r') as f:
            self.knowledge = json.load(f)
        
        # Load RAT problems
        rat_path = Path(__file__).parent.parent / "data" / "input" / "rat_100_problems.json"
        with open(rat_path, 'r') as f:
            self.rat_data = json.load(f)
        
        # Build NetworkX graph
        self.G = self._build_knowledge_graph()
        print(f"✅ Built knowledge graph with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
        
    def _build_knowledge_graph(self):
        """Build NetworkX graph from knowledge base"""
        G = nx.Graph()
        
        # Add word nodes with meanings
        for word, data in self.knowledge['words'].items():
            G.add_node(word, 
                      node_type='word',
                      meanings=data['meanings'])
            
            # Add meaning nodes
            for i, meaning in enumerate(data['meanings']):
                meaning_node = f"{word}_meaning_{i}"
                G.add_node(meaning_node, 
                          node_type='meaning',
                          text=meaning,
                          parent_word=word)
                G.add_edge(word, meaning_node, weight=1.0)
            
            # Add connection nodes and edges
            for connected_word, connection_phrases in data['connections'].items():
                # Direct word-to-word connection
                if connected_word in self.knowledge['words']:
                    G.add_edge(word, connected_word, 
                              weight=0.8,
                              connection_type='direct',
                              phrases=connection_phrases)
                
                # Add connection phrase nodes
                for phrase in connection_phrases:
                    phrase_node = f"phrase_{phrase.replace(' ', '_')}"
                    if phrase_node not in G:
                        G.add_node(phrase_node, 
                                  node_type='phrase',
                                  text=phrase)
                    G.add_edge(word, phrase_node, weight=0.7)
                    if connected_word in self.knowledge['words']:
                        G.add_edge(connected_word, phrase_node, weight=0.7)
        
        return G
    
    def message_passing(self, start_words, num_iterations=3):
        """
        Perform message passing to find convergence points
        Returns nodes with highest activation after propagation
        """
        # Initialize activation scores
        activation = defaultdict(float)
        
        # Start with high activation for input words
        for word in start_words:
            if word in self.G:
                activation[word] = 1.0
        
        # Message passing iterations
        for iteration in range(num_iterations):
            new_activation = defaultdict(float)
            
            # Propagate activation through edges
            for node, score in activation.items():
                if score > 0:
                    neighbors = self.G.neighbors(node)
                    for neighbor in neighbors:
                        edge_data = self.G.get_edge_data(node, neighbor)
                        weight = edge_data.get('weight', 0.5)
                        new_activation[neighbor] += score * weight * 0.7  # Damping factor
            
            # Normalize and update
            if new_activation:
                max_activation = max(new_activation.values())
                for node in new_activation:
                    activation[node] = new_activation[node] / max_activation
        
        return activation
    
    def detect_insight(self, words):
        """
        Detect insight by finding convergence through message passing
        """
        # Run message passing
        activation = self.message_passing(words)
        
        # Find candidate answers (high activation word nodes)
        candidates = []
        for node, score in activation.items():
            if (node in self.G and 
                self.G.nodes[node].get('node_type') == 'word' and
                node not in words and
                score > 0.5):
                candidates.append((node, score))
        
        # Sort by activation score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Check for insight spike
        if candidates and candidates[0][1] > 0.7:
            answer = candidates[0][0]
            confidence = candidates[0][1]
            
            # Verify connection to all input words
            connected_count = 0
            for word in words:
                if word in self.G and nx.has_path(self.G, word, answer):
                    path_length = nx.shortest_path_length(self.G, word, answer)
                    if path_length <= 3:  # Close connection
                        connected_count += 1
            
            if connected_count == len(words):
                return {
                    'answer': answer,
                    'confidence': confidence,
                    'spike_detected': True,
                    'activation_score': candidates[0][1],
                    'connected_words': connected_count
                }
        
        # No strong insight found
        return {
            'answer': candidates[0][0] if candidates else 'UNKNOWN',
            'confidence': candidates[0][1] if candidates else 0.0,
            'spike_detected': False,
            'activation_score': candidates[0][1] if candidates else 0.0,
            'connected_words': 0
        }
    
    def solve_rat(self, problem_words):
        """Solve a RAT problem using graph-based insight detection"""
        # Convert to uppercase to match knowledge base
        words = [w.upper() for w in problem_words]
        
        # Check if all words are in our knowledge base
        known_words = [w for w in words if w in self.G]
        if len(known_words) < len(words):
            print(f"⚠️  Missing words in knowledge base: {set(words) - set(known_words)}")
        
        # Detect insight
        result = self.detect_insight(known_words)
        return result
    
    def run_experiment(self):
        """Test on RAT problems"""
        print("\n🧪 Running Graph-based InsightSpike Experiment")
        print("=" * 60)
        
        # Test on problems we have in knowledge base
        test_problems = []
        for problem in self.rat_data['problems'][:20]:  # First 20 problems
            # Check if we have knowledge for these words
            if all(word in self.knowledge['words'] for word in problem['words']):
                test_problems.append(problem)
        
        print(f"Testing on {len(test_problems)} problems with complete knowledge")
        
        correct = 0
        spike_count = 0
        results = []
        
        for problem in tqdm(test_problems, desc="Testing"):
            result = self.solve_rat(problem['words'])
            
            is_correct = result['answer'] == problem['answer']
            if is_correct:
                correct += 1
                print(f"\n✅ Solved: {problem['words']} → {problem['answer']} (confidence: {result['confidence']:.2f})")
            
            if result['spike_detected']:
                spike_count += 1
            
            results.append({
                'problem': problem,
                'result': result,
                'correct': is_correct
            })
        
        # Summary
        total = len(test_problems)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"Spike Detection Rate: {spike_count}/{total} = {spike_count/total*100:.1f}%")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "outputs"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"graph_insightspike_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'experiment': 'Graph-based InsightSpike with Message Passing',
                    'num_problems': total,
                    'timestamp': timestamp
                },
                'summary': {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'spike_rate': spike_count / total if total > 0 else 0
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Analysis
        print("\n💡 INSIGHTS:")
        if accuracy > 30:
            print("✓ Message passing successfully finds conceptual connections!")
            print("✓ Graph structure enables insight detection")
        else:
            print("✗ Even with graph structure, some connections are too abstract")
            print("✗ May need richer semantic representations")

if __name__ == "__main__":
    insightspike = GraphBasedInsightSpike()
    insightspike.run_experiment()