#!/usr/bin/env python3
"""
Comprehensive geDIG Experiment
=============================

A large-scale evaluation of the geDIG framework with 100 knowledge items and 20 questions.
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ComprehensiveGEDIG:
    """Comprehensive GED-IG implementation with real embeddings"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.knowledge_graph = {}
        self.embeddings = {}
        self.model = SentenceTransformer(embedding_model)
        self.connections = {}
        self.phase_info = {}
        
    def add_knowledge(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Add knowledge item with semantic embedding"""
        text = item['text']
        item_id = item['id']
        
        # Get semantic embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        self.knowledge_graph[item_id] = {
            'text': text,
            'embedding': embedding,
            'category': item.get('category', 'general'),
            'phase': item.get('phase', 1),
            'connections': []
        }
        self.embeddings[item_id] = embedding
        self.phase_info[item_id] = item.get('phase', 1)
        
        # Connect to semantically similar nodes
        connection_count = 0
        for other_id, other_embedding in self.embeddings.items():
            if other_id != item_id:
                similarity = self._cosine_similarity(embedding, other_embedding)
                
                # Dynamic threshold based on phases
                phase_diff = abs(self.phase_info[item_id] - self.phase_info[other_id])
                threshold = 0.4 - (phase_diff * 0.05)  # Lower threshold for distant phases
                
                if similarity > threshold:
                    self.knowledge_graph[item_id]['connections'].append({
                        'id': other_id,
                        'similarity': float(similarity)
                    })
                    self.knowledge_graph[other_id]['connections'].append({
                        'id': item_id,
                        'similarity': float(similarity)
                    })
                    connection_count += 1
        
        return {
            'item_id': item_id,
            'connections': connection_count,
            'phase': item.get('phase', 1)
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def detect_spike(self, question: str) -> Dict[str, Any]:
        """Detect insight spike based on graph structure analysis
        
        TRANSPARENCY NOTE - Algorithm Implementation Details:
        1. Node Selection: cosine similarity > 0.3 (empirically chosen)
        2. Top-N Selection: exactly 10 nodes (not adaptive)
        3. Induced Subgraph: includes ALL existing edges between selected nodes
        4. No filtering or pruning of edges is performed
        5. This implementation may favor densely connected regions
        """
        # Get question embedding
        q_embedding = self.model.encode(question, convert_to_numpy=True)
        
        # Find relevant nodes with similarity scores
        relevant_nodes = []
        for node_id, node in self.knowledge_graph.items():
            similarity = self._cosine_similarity(q_embedding, node['embedding'])
            if similarity > 0.3:  # Lower threshold for better coverage
                relevant_nodes.append({
                    'id': node_id,
                    'similarity': similarity,
                    'text': node['text'],
                    'phase': node['phase'],
                    'category': node['category']
                })
        
        # Sort by similarity
        relevant_nodes.sort(key=lambda x: x['similarity'], reverse=True)
        relevant_nodes = relevant_nodes[:10]  # Consider top 10
        
        if len(relevant_nodes) < 3:
            return {
                'has_spike': False,
                'spike_confidence': 0.0,
                'response': "Insufficient knowledge to generate insights.",
                'relevant_count': len(relevant_nodes),
                'metrics': {}
            }
        
        # Analyze graph structure
        metrics = self._analyze_graph_structure(relevant_nodes)
        
        # Spike detection based on multiple factors
        spike_score = self._calculate_spike_score(metrics)
        has_spike = spike_score > 0.5
        
        # Generate response
        if has_spike:
            response = self._generate_insight_response(relevant_nodes, metrics)
        else:
            top_node = relevant_nodes[0]
            response = f"Based on the knowledge base: {top_node['text']}"
        
        return {
            'has_spike': has_spike,
            'spike_confidence': spike_score,
            'response': response,
            'relevant_count': len(relevant_nodes),
            'metrics': metrics
        }
    
    def _analyze_graph_structure(self, relevant_nodes: List[Dict]) -> Dict[str, float]:
        """Analyze the graph structure of relevant nodes"""
        node_ids = [n['id'] for n in relevant_nodes]
        
        # Cross-connectivity
        cross_connections = 0
        total_possible = len(node_ids) * (len(node_ids) - 1) / 2
        
        for i, node_id in enumerate(node_ids):
            node_connections = [c['id'] for c in self.knowledge_graph[node_id]['connections']]
            for j in range(i + 1, len(node_ids)):
                if node_ids[j] in node_connections:
                    cross_connections += 1
        
        connectivity_ratio = cross_connections / max(total_possible, 1)
        
        # Phase diversity
        phases = [n['phase'] for n in relevant_nodes]
        phase_diversity = len(set(phases)) / 5.0  # Normalized by total phases
        
        # Category diversity
        categories = [n['category'] for n in relevant_nodes]
        category_diversity = len(set(categories)) / len(categories)
        
        # Average similarity
        avg_similarity = np.mean([n['similarity'] for n in relevant_nodes])
        
        return {
            'connectivity_ratio': connectivity_ratio,
            'phase_diversity': phase_diversity,
            'category_diversity': category_diversity,
            'avg_similarity': avg_similarity,
            'cross_connections': cross_connections,
            'node_count': len(node_ids)
        }
    
    def _calculate_spike_score(self, metrics: Dict[str, float]) -> float:
        """Calculate spike score from multiple metrics"""
        # Weighted combination of factors
        weights = {
            'connectivity': 0.3,
            'phase_diversity': 0.3,
            'category_diversity': 0.2,
            'similarity': 0.2
        }
        
        score = (
            weights['connectivity'] * metrics['connectivity_ratio'] +
            weights['phase_diversity'] * metrics['phase_diversity'] +
            weights['category_diversity'] * metrics['category_diversity'] +
            weights['similarity'] * (metrics['avg_similarity'] - 0.3) / 0.7  # Normalize
        )
        
        # Bonus for high connectivity with diversity
        if metrics['connectivity_ratio'] > 0.3 and metrics['phase_diversity'] > 0.4:
            score += 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _generate_insight_response(self, relevant_nodes: List[Dict], metrics: Dict) -> str:
        """Generate an insight response based on connected concepts"""
        # Group by phase
        phase_groups = {}
        for node in relevant_nodes[:6]:  # Use top 6
            phase = node['phase']
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(node['text'][:50] + "...")
        
        # Build response
        response = "This question bridges multiple conceptual levels:\n"
        
        for phase in sorted(phase_groups.keys()):
            phase_name = ["Foundational", "Relational", "Integrative", "Exploratory", "Transcendent"][phase-1]
            response += f"\n{phase_name}: {', '.join(phase_groups[phase][:2])}"
        
        response += f"\n\nThe high connectivity (ratio: {metrics['connectivity_ratio']:.2f}) "
        response += f"and phase diversity ({metrics['phase_diversity']:.2f}) suggest "
        response += "an emergent insight at the intersection of these concepts."
        
        return response


def run_comprehensive_experiment():
    """Run the comprehensive geDIG evaluation"""
    print("=" * 80)
    print("Comprehensive geDIG Evaluation")
    print("100 Knowledge Items | 20 Test Questions")
    print("=" * 80)
    
    # Initialize
    gedig = ComprehensiveGEDIG()
    
    # Load knowledge base
    kb_path = Path(__file__).parent.parent / "data" / "input" / "knowledge_base_100.json"
    with open(kb_path, 'r') as f:
        knowledge_data = json.load(f)
    
    print("\nLoading knowledge base...")
    start_time = time.time()
    
    for i, item in enumerate(knowledge_data['knowledge_items']):
        result = gedig.add_knowledge(item)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/100] Loaded... (connections: {result['connections']})")
    
    load_time = time.time() - start_time
    print(f"\nKnowledge base loaded in {load_time:.2f}s")
    
    # Graph statistics
    total_connections = sum(len(node['connections']) for node in gedig.knowledge_graph.values()) / 2
    avg_connections = total_connections / len(gedig.knowledge_graph)
    print(f"Graph statistics: {len(gedig.knowledge_graph)} nodes, {int(total_connections)} edges")
    print(f"Average connections per node: {avg_connections:.2f}")
    
    # Load test questions
    q_path = Path(__file__).parent.parent / "data" / "input" / "test_questions_20.json"
    with open(q_path, 'r') as f:
        questions_data = json.load(f)
    
    # Run evaluation
    print("\n" + "=" * 80)
    print("Running evaluation...")
    print("=" * 80)
    
    results = []
    spike_count = 0
    
    for q_data in questions_data['questions']:
        print(f"\nQuestion {q_data['id']}: {q_data['question']}")
        print(f"Type: {q_data['type']} | Difficulty: {q_data['difficulty']}")
        
        start_time = time.time()
        result = gedig.detect_spike(q_data['question'])
        processing_time = time.time() - start_time
        
        if result['has_spike']:
            print(f"  ✨ SPIKE DETECTED! (confidence: {result['spike_confidence']:.3f})")
            spike_count += 1
        else:
            print(f"  📝 No spike detected (confidence: {result['spike_confidence']:.3f})")
        
        print(f"  Relevant nodes: {result['relevant_count']}")
        print(f"  Processing time: {processing_time:.3f}s")
        
        # Store result
        results.append({
            'question_id': q_data['id'],
            'question': q_data['question'],
            'type': q_data['type'],
            'difficulty': q_data['difficulty'],
            'has_spike': bool(result['has_spike']),
            'spike_confidence': float(result['spike_confidence']),
            'response': result['response'],
            'metrics': {k: float(v) for k, v in result['metrics'].items()},
            'processing_time': processing_time
        })
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\nOverall Performance:")
    print(f"  Spikes Detected: {spike_count}/20 ({spike_count/20*100:.1f}%)")
    
    # By difficulty
    for difficulty in ['easy', 'medium', 'hard']:
        diff_results = [r for r in results if r['difficulty'] == difficulty]
        diff_spikes = sum(1 for r in diff_results if r['has_spike'])
        print(f"  {difficulty.capitalize()}: {diff_spikes}/{len(diff_results)} ({diff_spikes/len(diff_results)*100:.1f}%)")
    
    # By type (top 3)
    type_counts = {}
    for r in results:
        q_type = r['type']
        if q_type not in type_counts:
            type_counts[q_type] = {'total': 0, 'spikes': 0}
        type_counts[q_type]['total'] += 1
        if r['has_spike']:
            type_counts[q_type]['spikes'] += 1
    
    print("\nBy Question Type (top 3):")
    sorted_types = sorted(type_counts.items(), 
                         key=lambda x: x[1]['spikes']/x[1]['total'], 
                         reverse=True)[:3]
    for q_type, counts in sorted_types:
        rate = counts['spikes']/counts['total']*100
        print(f"  {q_type}: {counts['spikes']}/{counts['total']} ({rate:.1f}%)")
    
    # Processing stats
    avg_time = np.mean([r['processing_time'] for r in results])
    print(f"\nProcessing Statistics:")
    print(f"  Average time per question: {avg_time:.3f}s")
    print(f"  Total processing time: {sum(r['processing_time'] for r in results):.2f}s")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent.parent / "results" / "outputs" / f"comprehensive_results_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'comprehensive_gedig_evaluation',
            'timestamp': timestamp,
            'configuration': {
                'knowledge_items': 100,
                'test_questions': 20,
                'embedding_model': 'all-MiniLM-L6-v2'
            },
            'summary': {
                'spike_rate': spike_count/20,
                'avg_processing_time': avg_time,
                'graph_nodes': len(gedig.knowledge_graph),
                'graph_edges': int(total_connections)
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Example insights
    print("\n" + "=" * 80)
    print("EXAMPLE INSIGHTS")
    print("=" * 80)
    
    spike_results = [r for r in results if r['has_spike']]
    if spike_results:
        example = max(spike_results, key=lambda x: x['spike_confidence'])
        print(f"\nHighest confidence spike ({example['spike_confidence']:.3f}):")
        print(f"Question: {example['question']}")
        print(f"\nResponse:\n{example['response']}")
    
    print("\n✅ Comprehensive evaluation complete!")


if __name__ == "__main__":
    run_comprehensive_experiment()