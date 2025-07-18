#!/usr/bin/env python3
"""
Proper InsightSpike implementation for RAT experiments
No mocks, no cheating - real insight detection
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
from transformers import pipeline, set_seed
from tqdm import tqdm
from collections import defaultdict
import re


class ProperInsightSpike:
    """
    Real InsightSpike implementation with:
    - Dynamic knowledge graph construction
    - geDIG-based spike detection
    - No hardcoded answers or cheating
    """
    
    def __init__(self):
        print("Initializing Proper InsightSpike...")
        
        # LLM for generation
        self.llm = pipeline('text-generation', model='distilgpt2', device=-1)
        set_seed(42)
        
        # Dynamic knowledge graph
        self.knowledge_graph = nx.Graph()
        self.concept_similarity_cache = {}
        
        # geDIG parameters
        self.ged_threshold = -0.5  # Significant graph simplification
        self.ig_threshold = 0.3    # Information gain threshold
        
        print("✅ InsightSpike initialized properly")
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Calculate simple word similarity based on common substrings"""
        w1, w2 = word1.lower(), word2.lower()
        
        # Check if one word contains the other
        if w1 in w2 or w2 in w1:
            return 0.8
        
        # Check common substrings (length 3+)
        common_score = 0
        for i in range(len(w1) - 2):
            if w1[i:i+3] in w2:
                common_score += 0.2
        
        return min(common_score, 0.6)
    
    def _calculate_ged(self, graph_before: nx.Graph, graph_after: nx.Graph) -> float:
        """
        Calculate Graph Edit Distance (simplified version)
        Negative values indicate graph simplification (good for insights)
        """
        nodes_before = len(graph_before.nodes())
        edges_before = len(graph_before.edges())
        nodes_after = len(graph_after.nodes())
        edges_after = len(graph_after.edges())
        
        # Normalized change
        node_change = (nodes_after - nodes_before) / max(nodes_before, 1)
        edge_change = (edges_after - edges_before) / max(edges_before, 1)
        
        # Negative values indicate simplification
        ged = (node_change + edge_change) / 2
        return ged
    
    def _calculate_ig(self, concepts_before: List[str], concepts_after: List[str]) -> float:
        """
        Calculate Information Gain
        Measures how much new understanding emerges
        """
        # Concept diversity before
        if len(concepts_before) > 1:
            # Calculate average dissimilarity
            diversity_before = 0
            pairs = 0
            for i in range(len(concepts_before)):
                for j in range(i+1, len(concepts_before)):
                    diversity_before += (1 - self._calculate_word_similarity(
                        concepts_before[i], concepts_before[j]
                    ))
                    pairs += 1
            diversity_before = diversity_before / max(pairs, 1)
        else:
            diversity_before = 0
        
        # Concept diversity after
        if len(concepts_after) > 1:
            diversity_after = 0
            pairs = 0
            for i in range(len(concepts_after)):
                for j in range(i+1, len(concepts_after)):
                    diversity_after += (1 - self._calculate_word_similarity(
                        concepts_after[i], concepts_after[j]
                    ))
                    pairs += 1
            diversity_after = diversity_after / max(pairs, 1)
        else:
            diversity_after = 0
        
        # Information gain is reduction in diversity (concepts converge)
        ig = diversity_before - diversity_after
        return ig
    
    def _build_dynamic_graph(self, words: List[str]) -> nx.Graph:
        """
        Build knowledge graph dynamically from word associations
        No hardcoding - uses semantic similarity
        """
        G = nx.Graph()
        
        # Add base words
        for word in words:
            G.add_node(word.upper())
        
        # Generate associations using LLM
        associations = {}
        for word in words:
            prompt = f"List 5 things closely related to {word}:"
            result = self.llm(prompt, max_new_tokens=30, temperature=0.7)
            
            # Extract associations
            text = result[0]['generated_text']
            # Simple extraction: split by common delimiters
            assoc_text = text.split(":")[-1] if ":" in text else text
            assoc_list = [a.strip().upper() for a in assoc_text.split(',')[:5] 
                         if len(a.strip()) > 2]
            
            associations[word.upper()] = assoc_list
            
            # Add to graph
            for assoc in assoc_list:
                G.add_edge(word.upper(), assoc, weight=0.8)
        
        # Find semantic connections between associations
        all_concepts = set()
        for assocs in associations.values():
            all_concepts.update(assocs)
        
        concepts_list = list(all_concepts)
        if len(concepts_list) > 1:
            # Calculate pairwise similarities
            for i in range(len(concepts_list)):
                for j in range(i+1, len(concepts_list)):
                    similarity = self._calculate_word_similarity(
                        concepts_list[i], concepts_list[j]
                    )
                    
                    # Add edge if similar
                    if similarity > 0.5:
                        G.add_edge(concepts_list[i], concepts_list[j], 
                                 weight=similarity)
        
        return G, associations
    
    def _find_insight_path(self, graph: nx.Graph, words: List[str]) -> Optional[str]:
        """
        Find the concept that best connects all words
        Uses graph algorithms, not hardcoded answers
        """
        words_upper = [w.upper() for w in words]
        
        # Method 1: Find nodes with paths to all words
        candidates = defaultdict(float)
        
        for node in graph.nodes():
            if node not in words_upper:
                # Check connectivity to input words
                connected_count = 0
                total_distance = 0
                
                for word in words_upper:
                    if word in graph:
                        try:
                            distance = nx.shortest_path_length(graph, node, word)
                            connected_count += 1
                            total_distance += distance
                        except nx.NetworkXNoPath:
                            pass
                
                if connected_count == len(words):
                    # Score based on average distance (lower is better)
                    avg_distance = total_distance / connected_count
                    candidates[node] = 1.0 / (avg_distance + 1)
        
        # Method 2: Betweenness centrality
        centrality = nx.betweenness_centrality(graph)
        for node, cent_score in centrality.items():
            if node in candidates:
                candidates[node] *= (1 + cent_score)
        
        # Return best candidate
        if candidates:
            best = max(candidates.items(), key=lambda x: x[1])
            return best[0], best[1]
        
        return None, 0
    
    def solve_rat(self, words: List[str]) -> Dict:
        """
        Solve RAT problem using proper InsightSpike approach
        """
        # Build initial graph
        graph_before, associations = self._build_dynamic_graph(words)
        
        # Find insight path
        answer, confidence = self._find_insight_path(graph_before, words)
        
        if answer and confidence > 0.3:
            # Build graph after insight
            graph_after = nx.Graph()
            for word in words:
                graph_after.add_edge(word.upper(), answer)
            
            # Calculate geDIG metrics
            delta_ged = self._calculate_ged(graph_before, graph_after)
            
            # Get concepts for IG calculation
            concepts_before = list(graph_before.nodes())
            concepts_after = [word.upper() for word in words] + [answer]
            delta_ig = self._calculate_ig(concepts_before, concepts_after)
            
            # Detect spike
            spike_detected = (delta_ged < self.ged_threshold and 
                            delta_ig > self.ig_threshold)
            
            return {
                'answer': answer,
                'confidence': confidence,
                'spike_detected': spike_detected,
                'metrics': {
                    'delta_ged': delta_ged,
                    'delta_ig': delta_ig,
                    'graph_nodes_before': len(graph_before.nodes()),
                    'graph_nodes_after': len(graph_after.nodes()),
                    'associations_found': len(associations)
                }
            }
        
        # Fallback: Use enhanced generation
        # Build context from graph
        context_parts = []
        for word, assocs in associations.items():
            if assocs:
                context_parts.append(f"{word} relates to: {', '.join(assocs[:3])}")
        
        context = "; ".join(context_parts)
        prompt = f"Given these relationships: {context}. What single word connects {', '.join(words)}? Answer:"
        
        result = self.llm(prompt, max_new_tokens=5, temperature=0.5)
        generated = result[0]['generated_text']
        
        # Extract answer
        if "Answer:" in generated:
            answer_part = generated.split("Answer:")[-1].strip().upper().split()
            answer = answer_part[0] if answer_part else "UNKNOWN"
        else:
            words = generated.split()
            answer = words[-1].upper() if words else "UNKNOWN"
        
        return {
            'answer': answer,
            'confidence': 0.2,
            'spike_detected': False,
            'metrics': {
                'delta_ged': 0,
                'delta_ig': 0,
                'method': 'generation_fallback'
            }
        }


def run_proper_comparison():
    """Run comparison with proper InsightSpike implementation"""
    print("🔬 Running Proper InsightSpike RAT Experiment")
    print("=" * 60)
    
    # Load RAT problems
    data_path = Path(__file__).parent.parent / "data" / "input" / "rat_100_problems.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Initialize systems
    base_llm = pipeline('text-generation', model='distilgpt2', device=-1)
    set_seed(42)
    insight_spike = ProperInsightSpike()
    
    # Test on subset for demo (use all 100 for full test)
    test_problems = data['problems'][:20]
    
    results = []
    base_correct = 0
    insight_correct = 0
    spike_count = 0
    
    print(f"\nTesting {len(test_problems)} RAT problems...")
    print("-" * 60)
    
    for problem in tqdm(test_problems, desc="Processing"):
        # Base LLM
        base_prompt = f"What word connects {', '.join(problem['words'])}?"
        base_result = base_llm(base_prompt, max_new_tokens=5)
        base_answer = base_result[0]['generated_text'].split()[-1].upper()
        
        # Proper InsightSpike
        insight_result = insight_spike.solve_rat(problem['words'])
        
        # Check correctness
        correct_answer = problem['answer']
        base_is_correct = base_answer == correct_answer
        insight_is_correct = insight_result['answer'] == correct_answer
        
        if base_is_correct:
            base_correct += 1
        if insight_is_correct:
            insight_correct += 1
        if insight_result['spike_detected']:
            spike_count += 1
        
        results.append({
            'problem': problem,
            'base': {'answer': base_answer, 'correct': base_is_correct},
            'insight': insight_result
        })
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    total = len(test_problems)
    print(f"Base LLM      : {base_correct}/{total} = {base_correct/total*100:.1f}%")
    print(f"InsightSpike  : {insight_correct}/{total} = {insight_correct/total*100:.1f}%")
    print(f"Spike Rate    : {spike_count}/{total} = {spike_count/total*100:.1f}%")
    
    # Analyze spikes vs correctness
    spike_when_correct = sum(1 for r in results 
                           if r['insight']['spike_detected'] and 
                           r['insight']['answer'] == r['problem']['answer'])
    
    print(f"\nSpike Analysis:")
    print(f"Spikes when correct: {spike_when_correct}/{insight_correct if insight_correct > 0 else 1} = "
          f"{spike_when_correct/max(insight_correct, 1)*100:.1f}%")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"proper_insightspike_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'experiment': 'Proper InsightSpike RAT Test',
                'num_problems': total,
                'timestamp': timestamp
            },
            'summary': {
                'base_accuracy': base_correct/total,
                'insight_accuracy': insight_correct/total,
                'spike_rate': spike_count/total,
                'spike_correctness_correlation': spike_when_correct/max(insight_correct, 1)
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_proper_comparison()